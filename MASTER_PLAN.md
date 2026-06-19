# GLM-5.2 を限界まで速くする — ハッカーの戦闘計画書

> ミッション: 手元のマシンパワーを1ワットも遊ばせず、GLM-5.2 (754B MoE, UD-Q2 254GB) の
> **prefill(encode) と decode を物理限界まで速くする**。量子化は落とさない。やれる手段はなんでもやる。

最終更新: 2026-06-19 / 関連メモリ: `glm52-speedup`, `glm52-ram-offload-gotchas`, `glm52-dsa-llamacpp`
設計詳細: `CONCEPT_per_expert_offload.md` / 計測 harness: `harness/`

---

## 0. 結論サマリ (忙しい人向け)
- **現在地: decode 26 t/s** (7.9 → 26 へ改善済、`--no-mmap` + 層単位offload)
- **B2 de-risk 完了 (2026-06-19, `harness/poc_overlap.cu`)**: overlap の物理は成立。pinned H2D 実測 **57.7 GB/s/GPU** (Gen5の90%)、**Max-Q も同速** (idle降速は負荷時に解消)、**両GPU同時 115 GB/s** (PCIe 2本足し算可)、**compute はコピー裏に完全隠蔽** (重ね=max(c,copy) ≠ sum)。新天井: 片GPU運搬 ~39 t/s / **両GPU分担 ~77 t/s 射程**。残risk は純工学 (llama.cpp CUDA graph 組込 + route事前確定 + event-wait)
- **profile 知見**: 116token では尾を解像不能 (層あたり平均89 expert が0-count)。routing均一+VRAM-fit が cold比を決めるので hot/cold集合の中身はほぼ任意 → 恒等split (cold=id208-255) で十分 (router permute不要)
- **decode 律速 = CPU offload した expert を毎token RAM帯域(50GB/s)で読む**。CPU計算でなく帯域。GPUは手待ち
- **物理天井: 全VRAM化は不可能** (expert 222GB > VRAM 192GB、最低30GB はGPU外、量子化削減却下のため)
- **唯一の本命レバー = per-expert cold offload を attention裏でprefetch overlap** → ~50 t/s 射程
- 安易な近道 (CUDA_Host直読み / flag-only offload) は**実測とコードで両方死亡確認済**。勝ち筋は overlap一本

---

## 1. 戦場 (ハード/モデル — 確定値)
### マシン
- **GPU0**: RTX PRO 6000 Blackwell **Workstation** (SM120, 96GB, PCIe **Gen5 x16 = 64GB/s**)
- **GPU1**: RTX PRO 6000 Blackwell **Max-Q** (SM120, 96GB, 電力制限, idle時PCIe Gen1降速)
- **VRAM合計 192GB** / VRAM帯域 ~1.7TB/s (Max-Q込み実効 ~0.9TB/s)
- **CPU**: Threadripper PRO 9965WX 24C (AVX512/AVX512_BF16/AVX_VNNI/AMX相当, REPACK=1)
- **RAM 125GB** (空き ~108GB, 帯域 ~50GB/s) / Optane 343GB (~6GB/s, 現状満杯)
- P2P は `iommu=pt` 前提

### モデル (GGUFメタ実測)
- glm-dsa: **79層** (先頭3 dense + 76 MoE)、**256 expert中 top-8 + shared 1**
- d_model 6144 / expert FFN 2048 / **MLA (KV head 1)** / context 1M
- **DSA indexer** top_k=2048 / 32head / key128 (GGUFに格納済、llama.cpp未対応)
- **MTP/NextN head** (`blk.78.nextn.*`) もGGUFに実在 (llama.cpp では unused)
- 量子化 UD-Q2_K_XL = **254GB / 7分割** (`/mnt/data/models/GLM-5.2-GGUF/UD-Q2_K_XL/`)
- 全routed expert = 734B = **222GB** (モデル本体の主体)、**per-token active = 34.5B = 10.4GB/token**

---

## 2. ルーフライン (天井の物理)
decode は **完全に weight 帯域律速** (compute はガラ空き)。1token時間 ≈ 触る重みバイト / 帯域。

| シナリオ | 計算 | 天井 |
|---|---|---|
| 全VRAM resident (1.7TB/s) | 10.4GB ÷ 1.7TB/s | **152 t/s** |
| Max-Q込み実効 (0.9TB/s) | 10.4GB ÷ 0.9TB/s | **80 t/s** |
| naive offload (溢れをPCIe 25GB/s) | +87ms/token | **~8 t/s に崩落** |

**核心の物理制約**: expert 222GB > VRAM 192GB。量子化を落とさない以上、**最低30GBは常にGPU外**。
→ 全VRAM (80-152 t/s) は**原理的に到達不能**。現実的上限は overlap が効いて **~50 t/s**。

---

## 3. 律速の解剖 (実測, 決定的)
- decode 38ms/token ≈ **CPU に置いた expert を RAM帯域で読む** (1.9GB/token ÷ 50GB/s)
- CPU compute(~11ms) ではなく **RAM帯域律速** → **CPUを速くしても無駄** (AVX/AMX は効かない)
- 効くのは **per-token に動かす expert バイトを減らすこと**、それだけ
- NVTOPでGPU 0%に見えるのは故障でなく**この症状** (CPUがRAM帯域待ちでstall、GPUは手待ち)
- **expert routing はほぼ均一** (dump 120tok全76層): cover80≈160 / cover90≈208 / cover95≈232 (/256)
  - 強いhot/cold偏りは無い (MoE負荷分散の性質)。cold ~48/256 が活性の~10%

---

## 4. 実測の階段
| 版 | 構成 | decode | prefill |
|---|---|---:|---:|
| BASELINE | 均等-ot配置, mmap有効 | **7.9 t/s** | 3.1 |
| OPT-1 | **--no-mmap** + CPU=19層offload | **26.15 t/s** | 35.5 |
| flag試験 | OFFLOAD_MIN_BATCH=1 (used-expert GPU copy) | ❌**18 t/s** | 30.3 |

- `--no-mmap` が真犯人を排除 (mmap だと CPU expert を disk-backed mmap から再fault読みしていた)
- CPU=19層が VRAM実限界 (94/92GB、CPU=18層は device0 OOM)

---

## 5. 武器庫 (全ハック — decode & prefill、状態と根拠つき)

### ✅ 採用 / 検証中
| # | ハック | 対象 | 効果見込み | 状態 |
|---|---|---|---|---|
| A1 | **--no-mmap** (CPU expert を実RAM常駐) | decode | 7.9→26 (3.3x) | ✅完了 |
| A2 | **VRAM配置最適化** (-ot で hot expert最大常駐, CPU=19層) | decode | 配置の床 ~26-30 | ✅完了 |
| **B1** | **per-expert hot/cold split** (GGUF前処理, cold~19%のみCPU) | decode | copy 1.7→0.34GB/token | 🔨本命・実装へ |
| **B2** | **attention裏 prefetch overlap** (graph外stream+固定staging+event-wait) | decode | copyを隠蔽→GPU-resident接近 | 🔨本命・実装へ |
| C1 | **KV量子化 (q8_0/q4)** | prefill+長context | KVバイト1/2-1/4, DSAと直交 | ⏳即試せる |
| C2 | **DSA lightning indexer** (graph-mask, glm52-dsa) | 長context | KV読み出し削減 | ⏳設計済(別proj) |
| C3 | **dual-GPU 負荷分散調整** (Max-Q電力/PCIe考慮の層配置) | both | 通信ボトル緩和 | ⏳要profile |

### 🔬 要検証 (条件付き)
| # | ハック | 懸念 |
|---|---|---|
| D1 | **MTP/NextN 投機** (GGUFにhead実在) | partial offload下はverifyがCPU expert増やし逆効果。**B1+B2で全GPU化が前提** |
| D2 | **SM120 MMVQ/MoE GEMV カーネル最適化** | GPU常駐部は既に速く律速でない。**decodeには効かない**。prefill/throughput向け |
| D3 | **prefill専用最適化** (batch大→GPU offload閾値超で自動GPU計算) | OFFLOAD_MIN_BATCH=32 既定。大batch prefill は既にGPU。要実測 |
| D4 | **megakernel / CUDA graph 最適化** | launch overhead削減、最後の数% |

### ❌ 却下 (実測/コード根拠で死亡)
| # | ハック | 死因 |
|---|---|---|
| X1 | 量子化を落として全VRAM化 | 品質崩壊 (ユーザ決定で却下) |
| X2 | KVをOptaneに置く | hot path毎token読む、Optane遅い、逆効果 |
| X3 | **CUDA_Host直読み** (GPUがhost expertをUVAで直読み計算) | discrete GPUは非supports_buft(ggml-cuda.cu:5421) + MMVQ細粒度loadがPCIeでmiss storm 7-30x遅化 |
| X4 | **flag-only used-expert GPU copy** | copy↔compute直列+ids D2H sync で隠れず、18 t/sに悪化。**overlapが無いと負ける** |
| X5 | CPU側 expert計算高速化 (AVX/AMX) | 律速はRAM帯域でcompute非律速、無駄 |

---

## 6. 本丸アーキテクチャ — per-expert offload + prefetch overlap
**発想**: cold expert を「CPUで計算」するのをやめる。計算は全部GPU。cold重みをどうGPUに届けるかだけの問題にする。
naive な GPU-copy (X4) は直列で負けた → **勝敗を分けるのは overlap**。

### 設計 (Codex壁打ち5問で確定)
1. **hot/cold split (Q2)**: GGUF前処理で `ffn_{gate,up,down}_exps` を hot[0-207]/cold[208-255] に物理分割。
   `ffn_gate_inp` / `exp_probs_b` / per-expert scale も同permute。
   router id remap: `hot_ids = id<208?id:0` / `cold_ids = id>=208?id-208:0`、各branch出力をmaskで潰して加算。
   weight取得 (`get_rows(probs, selected_experts)`) は global id のまま。
2. **prefetch overlap (Q3)**: 動的copyはCUDA graph本体に注入不可 (graph再利用がdata/ne/nbのmemcmp判定)。
   → **graphは静的維持**。routeをgraph launch前にhost確定 → cold used expertを**graph外のprefetch streamで host→VRAM staging へ async H2D** → graph内のexpert消費直前に**固定event-wait node**。
   staging slot は**固定アドレス** (graph再利用を壊さないため)。
3. **pinned管理 (Q4)**: 42GB丸pinは危険 (cudaMallocHost失敗でfallback) → **小pinned staging ring + 選択copy**。

### 期待
cold ~19% (0.34GB/token) だけ PCIe Gen5(64GB/s)を通り、それを attention の影に隠蔽
→ PCIeコピー不可視化 → decode が GPU-resident 速度に接近 → **~50 t/s 射程** (現状の~2x)。

---

## 7. ロードマップ
**Phase 1 — profile & 前処理 (独立, 先行可)**
- [ ] expert頻度profile確定 (`GLM_DUMP_EXPERTS` + `40_analyze_reuse.py`) → hot/cold集合を実データで
- [ ] GGUF前処理スクリプト: expert物理分割 + router permute + メタ書換

**Phase 2 — 独自ビルド本体 (Codex実装 / Opus設計)**
- [ ] build_moe_ffn 2分岐 (llama-graph.cpp:1310- / top_k:1458 / mul_mat_id:1519,1543,1561)
- [ ] cold側 staging経由GPU計算
- [ ] attention裏 prefetch stream + 固定staging + event-wait (ggml-cuda.cu:4244,4330)
- [ ] `--dsa`同様の opt-inフラグ下、off で回帰ゼロ

**Phase 3 — 全GPU化が解錠する追撃**
- [ ] MTP投機 (D1, 全GPU化前提) / KV量子化(C1) / DSA(C2)
- [ ] SM120カーネル(D2) は prefill/throughput 向けに別途

**Phase 4 — prefill(encode)側**
- [ ] 大batch prefillのGPU offload経路確認 / KV量子化 / DSAで長context prefill加速

---

## 8. 運用の罠 (踏んだ地雷, `glm52-ram-offload-gotchas` 詳細)
- **同時ロード衝突**: 複数プロセスが237GB同時loadで `failed to load model` (OOM明示なし)。**1ドライバ1ロード厳守**、起動前 `pgrep llama` + `nvidia-smi --query-compute-apps`
- **237GB reload ~1GB/s = 4分地獄** → 配置探索は server常駐 + API推奨
- **-ot 構文**: 複数フラグ=最後のみ採用 / セミコロン=unknown buffer → **comma区切り単一引数**が正
- **llama-server は `--fit off`** 必須 (-ngl 999 と auto-fitが衝突しabort)
- **CPU override時 `--no-mmap`** 必須 (mmapだと推論中ディスク再fault)
- **`-no-cnv` 非対応** (新build): llama-cli=対話落ち、bench=llama-bench、単発=llama-completion
- swap 4GB極小、二重ロードで即満杯

---

## 9. 計測プロトコル (実測ドリブン厳守、憶測でカーネル書かない)
- harness: `harness/` (env.sh / 10_baseline / 20_nsys / 30_ncu / 40_analyze_reuse.py)
- baseline: llama-completion + `-ot` 3デバイス明示 + `--no-mmap`, eval tok/s を読む
- 律速確認: nsys で律速カーネル + 達成帯域比 (dram pct_of_peak)、両GPU>70%飽和を nvidia-smi 30s で確認
- カーネル多ファイル改修は Codex委譲 (env `llamacpp-cu131`, SM120, gcc-15 host compiler の罠)

---

*我々はハッカーだ。物理の天井までは行く。その先は天井を動かす (overlapで律速を隠す) しかない。*

## ⚠️ 2026-06-19 STREAM実測でRAM帯域の前提が崩れた
- STREAM triad: **全48スレッド=135.3 GB/s / 1スレッド=46.5 GB/s** (8ch DDR5 実効)
- 現decode は 1.7GB/token を 38ms = 実効~45GB/s = **ほぼ単スレッド帯域しか使えてない**
- → 律速は「RAM帯域(50)」でなく、(A)CPU expert読みのスレッド並列不全 か (B)Q2_K/Q3_K dequant計算律速 のどちらか。要thread-sweep判定
- (A)なら CPU側修正だけで ~79 t/s 射程(GPUコピー不要)。(B)ならGPUコピー正当

## ★2026-06-19 thread-sweep で律速の正体が判明(計画の中核診断を訂正)
- decode tg96 vs threads: 12→18.4 / **24→23.9** / 36→24.3(頭打ち) / **48→10.5(SMT崩壊)**
- 24スレッド(物理コア)でスケール頭打ち、実効~40GB/s(STREAM同条件152GB/sの26%)= **帯域飽和でなくCPU計算(dequant)律速**
- **訂正: §3「RAM帯域律速」は誤診。真因=Q2_K/Q3_K dequant が24コアを食い潰す compute律速**
- 戦略的含意: (1)GPUコピー正当化強化(GPU dequant>>CPU、copy払っても勝つ。dual-link 15msでoverlap無でも~50t/s) (2)X5(CPU dequantカーネル高速化)理屈上復活だが24コア飽和済で頭打ち (3)**SMT(>24thread)厳禁=無料gotcha**

## ★★2026-06-19 spike実測 + 律速の最終確定 → 勝ち筋を「overlapped copy」から「CPU∥GPU並列」へ転換
### spike結果(pinned CUDA_Host overflow + OFFLOAD_MIN_BATCH=1, llama-completion ctx8192)
- decode **17.94 t/s**(baseline 21.79@ctx8192 / 26.15@ctx4096 を下回る)。dmon: GPU0 PCIe RX **max 51.7GB/s**(=pinned copy正常発火) / sm util 10% / GPU1 rx ほぼ0(=single-link)
- arg.cpp 改修で -ot に `CUDA_Host`(pinned host buft)を露出済(llama-bench は独自parserで非対応、completion/cli/serverは可)
### 「overlapped copy(ゴール当初の形)」が当たる物理の壁
1. **strict layer依存**: layer L+1 の expert ids は layer L の MoE 出力後にしか判明 → 未来 expert の prefetch / dual-link 同時コピー(2 layer分同時)が原理的に不可能
2. **PCIe は VRAM の 1/30**: コピー(1.7GB=30ms@single)を隠せる独立 GPU 計算は resident層の~6msしかない → 大半が露出。layer内で copy と独立なのは shared expert(~0.5ms/層)のみ=焼け石
→ **overlapped copy 単独では CPU計算(24-26)を超えられない**(spike 17.94 が実証)
### 真の勝ち筋(転換): CPU∥GPU 並列 expert 分割
- 律速は**CPU dequant計算**(thread-sweep確定)で GPU と両PCIeは遊休。**同一MoE層の8 used expertを位置でCPU部分集合/GPU部分集合に分割し2本のMUL_MAT_IDで並列計算**(disjoint idスライス=2倍計算の無駄なし、bit-ident:全expert計算と数学的同値)
- copy は「未来GPU計算」でなく「同層のCPU計算」の裏に隠れる→依存の壁を回避。GPU部分を両GPUに割れば両PCIe同時稼働
- scheduler は split を async 発行(ggml-backend.cpp:1678 callback無し経路で非同期)=backend並列が原理上可能
- 試算: CPU3/GPU5(両GPU)で balanced → 30ms→~11ms → **~50 t/s 射程**。これがゴール(dual-GPU overlap)の実現可能形
