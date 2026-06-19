# GLM-5.2 per-expert cold offload — 独自ビルド構想

## 律速の確定事実 (実測)
- decode 26 t/s。律速 = CPU に置いた expert を毎token **RAM帯域(50GB/s)で読む** 1.9GB/token。CPU compute でなく帯域。GPU は手待ち
- expert routing ほぼ均一 (cover90≈208/256)、強い hot/cold 偏り無し
- 物理: expert 222GB > VRAM 192GB、最低30GB GPU外不可避 (量子化削減は却下)

## 逆転の発想
**cold expert を CPU で計算するのをやめる。計算は全部 GPU。cold の "重み" を GPU にどう届けるかだけの問題にする。**
→ 律速が「CPU RAM帯域 + GPU idle」から「重み転送(隠せる)」に変わる。

## ハック梯子
### Phase 0 — `-ot ...=CUDA_Host` 直読み → ❌ボツ (Codex検証 2026-06-19)
- Q1: discrete GPU の CUDA backend は CUDA_Host を supports_buft しない(integrated限定, ggml-cuda.cu:5421)。op は CPU backend に落ちる
- Q4致命: 通しても MMVQ が量子化重みを kernel内で細粒度 load(mmvq.cu:396/vecdotq.cuh:817) → PCIe越しに miss storm、device比 7-30x 遅化。**直読みは原理的に死亡**

### ★生き筋 (Codex Q4-2b 発掘) — used-expert を VRAM staging へ先読みコピー
- master に既存: host weight の **「使う expert だけ device コピー→GPU計算」経路** = ggml-backend.cpp:1576-1635 (used ids抽出1604, expert範囲copy1623)。これが土台
- アーキ: per-token、route 確定後に cold used expert だけ host→VRAM staging へ async H2D、GPUは VRAM full帯域で計算
- Q3制約: 動的copyをCUDA graph本体に注入不可。**graph静的維持 + H2Dはgraph外専用stream + staging slot固定アドレス + event-wait node**(ggml-cuda.cu:4244/4330)。routeはgraph launch前にhost確定が前提
- Q4-4: 42GB丸pin危険(cudaMallocHost失敗でfallback) → **小pinned staging ring + 選択copy**
- Q2: hot/cold分割は **GGUF前処理で物理2分割**(hot0-207/cold208-255)+router id permute(id<208→hot, else→cold-208 + mask加算)。in-memory分割は正しさ表面積大でボツ。mask方式は両branch全slot計算=無駄、最大性能には後でcompact/scatter

### Phase 1 — per-expert hot/cold split (独自ビルド)
- hot~208/層を VRAM、cold~48/層を CUDA_Host。PCIe を通るのは cold ~10%活性のみ → 0.19-0.75GB/token
- 改造点: build_moe_ffn (llama-graph.cpp:1310-) で top_k(:1458) 後に selected_experts を hot/cold に分岐 → 2本の mul_mat_id
- tensor split: ffn_*_exps を _hot/_cold に。loader (llama-model-loader.cpp:1156-) or GGUF 前処理 + router id 再マップ

### Phase 2 — prefetch during attention (本命のクールさ)
- router gate は軽い → top_k を attention 前に確定 → **attention 計算中に cold expert を host→VRAM staging へ async H2D** (別stream) → 転送を隠蔽
- 素材: streams/events/cudaMemcpyAsync 全て存在 (common.cuh:1365 / ggml-cuda.cu:3137,3221)
- リスク: 静的グラフ + CUDA graph capture に動的 prefetch を注入できるか

### Phase 3 — dual-GPU + CPU 3レーン (最大)
- 8 active expert を GPU0常駐 / GPU1常駐 / host-stream の3経路に分散、総帯域を足す

## 実測検証ログ
- ❌ flag-only quick win (`GGML_OP_OFFLOAD_MIN_BATCH=1 --no-mmap -ot exps=CPU`): **18.12 t/s** (baseline 26 から悪化)。used-expert→GPU copy 経路は発火するが Q5-4 通り copy↔compute 直列 + ids D2H sync(ggml-backend.cpp:1606) が隠れず、CPU計算に負ける。**naive路線は負け=overlapが勝ち筋と実証**

## 確定: 独自ビルドの本体 (2点)
1. **hot/cold split (Q2, GGUF前処理)**: cold ~19%だけcopy → 1.7→~0.34GB/token
2. **prefetch overlap (Q3)**: graph外prefetch stream + 固定staging slot + event-wait で attention の影に cold copy を隠す
→ PCIeコピーを不可視化 → decode が GPU-resident速度に接近 (~50 t/s射程)

## 次の実装ステップ
1. GGUF前処理スクリプト: ffn_*_exps を hot(0-207)/cold(208-255) 物理分割 + router permute (要 expert頻度profile)
2. build_moe_ffn 2分岐 + cold側を staging経由GPU計算、attention裏prefetch (Codex実装/Opus設計)
3. expert頻度profile: GLM_DUMP_EXPERTS で hot/cold集合を確定 (routing均一なので僅差、要実データ)
