# GLM-5.2 decode 高速化 — CPU∥GPU 並列 expert 分割 実装要件書

> このドキュメントは **確定情報** として書く。記載の事実はすべて 2026-06-19 セッションで
> **実コード参照 or 実測** で裏取り済み。`[確定]` = 検証済 / `[未解決]` = 次セッションで検証要。
> 次セッションはこれを読めば再導出不要で実装に入れる。
>
> 対象リポ: `~/src/llama.cpp` (build: `cmake --build build -j`, env `conda llamacpp-cu131`, host `g++-15`, CUDA 13.1, SM120/compute_120a)
> モデル: `/mnt/data/models/GLM-5.2-GGUF/UD-Q2_K_XL/GLM-5.2-UD-Q2_K_XL-00001-of-00007.gguf` (7分割自動ロード)
> 関連: `MASTER_PLAN.md`(全経緯), メモリ `glm52-speedup`

---

## 0. TL;DR
- **狙い**: decode を現状 **26.15 t/s → ~50 t/s**(物理上限の現実解)。量子化は UD-Q2 固定(劣化不可、ユーザ決定)。
- **手段**: 同一 MoE 層の top-8 expert を **位置で CPU 担当 / GPU 担当に2分割し、両 backend で並列計算**。
  コピーを「未来の GPU 計算」でなく「**同じ層の CPU 計算**」の裏に隠す → 後述の依存の壁を回避。
- **これが唯一の勝ち筋**。「コピーを GPU 計算裏に overlap」する当初案は物理の壁で頭打ち(実測 17.94 t/s で死亡確認)。

---

## 1. 律速の正体 [確定・実測]
**decode は CPU の dequant **計算**律速。RAM 帯域律速ではない**(←計画の旧診断は誤り、訂正済)。

根拠(実測, harness/out/):
| 計測 | 結果 | 出所 |
|---|---|---|
| thread-sweep (decode tg96) | 12→18.4 / **24→23.9** / 36→24.3(頭打ち) / **48→10.5(SMT崩壊)** | `sweep_threads.log` |
| STREAM triad RAM帯域 | 1t=46.5 / 12t=101 / **24t=152(ピーク)** / 48t=136 GB/s | `/tmp/stream.c` |
| 現 decode 実効帯域 | 1.7GB/token ÷ ~42ms ≈ **40-45 GB/s = STREAM同条件(152)の26%** | 算出 |

→ 24スレッド(物理コア数)で頭打ち + 帯域の26%しか使えてない = **dequant(Q2_K/Q3_K展開)が24コアを食い潰す compute 律速**。
**含意**: (a) CPU を速くする道は「より速い dequant カーネル」だけ(24コア飽和済でスレッド増は無効)。(b) **GPU は dequant が桁違いに速い + 両 PCIe 遊休** → GPU に計算を分担させれば勝てる。
**無料 gotcha**: **48スレッド(SMT)は 10.5 t/s と壊滅。24スレッド固定が正解**。

---

## 2. なぜ「overlapped copy(当初ゴール)」は死んだか [確定・実測+コード]
spike 実測: offload expert を pinned host(`CUDA_Host`)に置き `GGML_OP_OFFLOAD_MIN_BATCH=1` で GPU へ強制 copy+compute
→ **decode 17.94 t/s**(baseline 21.79@ctx8192 / 26.15@ctx4096 を下回る)。
dmon: **GPU0 PCIe RX max 51.7 GB/s**(=pinned copy 正常発火) / sm util 10% / GPU1 ほぼ0(single-link)。`spike_pinned_offload.log` + `spike_dmon.log`。

**当たった2つの物理の壁**:
1. **層の厳密直列依存** — layer L+1 がどの expert を使うかは layer L の MoE 出力後にしか判明しない。
   → 未来 expert の prefetch も、2層分を両GPUに同時コピー(=dual-link 並列)も **原理的に不可能**。
2. **PCIe は VRAM の 1/30** — コピー(1.7GB=~30ms@single link 57GB/s)を隠せる独立 GPU 計算は resident層の~6msしかない。layer内で copy と独立なのは shared expert(~0.5ms/層)のみ=焼け石。

→ **「コピーを GPU 計算の裏に隠す」路線は構造的に頭打ち**。これがゴール当初形の限界。

---

## 3. 勝ち筋アーキテクチャ — CPU∥GPU 並列 expert 分割
### 発想
律速は CPU 計算で GPU・両PCIe は遊休。**同層の8 used expert を「CPUで k 個・GPUで 8-k 個」に分けて同時計算**する。
- GPU 担当分の **コピーは、CPU 担当分の **計算**(同じ層・常に存在)の裏に隠れる** → §2 の依存の壁を回避。
- GPU 担当分を CUDA0/CUDA1 に割れば **両 PCIe link 同時稼働**(これが「dual-GPU」の実現形)。
- disjoint な id スライス(位置 0..k-1 を CPU、k..7 を GPU)で2本の MUL_MAT_ID → **2倍計算の無駄なし**、全 expert 計算と **数学的に同値=bit-identity 可能**。

### ルーフライン試算 [未解決・要実測]
balanced(例 CPU 3 / GPU 5、GPU分を両link)で offload 層の時間 ~1.6ms→~0.6ms/層 → decode 38ms→~20ms ≈ **~50 t/s 射程**。
※ これは projection。実測で確定要(特に scheduler 並列実行が効くか=§6)。

---

## 4. モデル/メモリの確定値 [確定・GGUF実測]
全7shard を gguf-py で集計(ロード不要):
| 項目 | 値 |
|---|---|
| total | 236.4 GB |
| routed_experts | 221.8 GB |
| **非routed合計** | **14.6 GB**(attn 10.5 + shared 2.0 + embed 1.1 + router 0.45) |
| 1 expert (gate+up+down) | **11.53 MB** |
| dtype | gate/up = **Q2_K**, down = **Q3_K**(一部早層 IQ系混在) |
| アーキ | glm-dsa, **79層(先頭3 dense + 76 MoE)**, expert **256中 top-8 + shared 1**, d_model 6144, expert FFN 2048, MLA(KV head 1) |
| VRAM 配置の天井 | 192GB - 非routed15 - KV余裕8 = hot に 169GB = **expert の 76%(195/256)が常駐限界。24%(53GB)は常にGPU外** |
| per-token offload traffic | 8 expert × offloaded層分 ≈ **1.7 GB/token**(現 19層 offload 構成) |

**ハード**: GPU0 = RTX PRO 6000 Blackwell WS (96GB), GPU1 = Max-Q (96GB)。両 PCIe Gen5 x16 **実測 57.7 GB/s/GPU・両同時 115 GB/s**(`poc_overlap.cu`)。Max-Q は負荷時 Gen5 で同速(idle降速は杞憂)。RAM 125GB(空き~108)。P2P は `iommu=pt`。

---

## 5. 実装挿入点 [確定・全て実コード行番号で裏取り]
### graph(MoE 本体)
- `glm-dsa` の graph builder = **deepseek2 の graph を共有**
  (`src/models/models.h`: `struct llama_model_glm_dsa { using graph = llama_model_deepseek2::graph; }`)。
  → MoE block は **`src/models/deepseek2.cpp:388`** で `build_moe_ffn` を呼ぶ(1st overload)。
  **deepseek2/deepseek2ocr/mistral4 も同 graph 共有** → 改造は **opt-in(env flag)で cold/split off 時に原挙動完全維持**が必須。
- `build_moe_ffn` 本体 = **`src/llama-graph.cpp`**:
  - 1st overload 宣言/実装: `:1310`(`:1339` で 2nd へ委譲)
  - 2nd overload 実装: **`:1352`–`:1700+`**。鍵となる行:
    - top_k(selected_experts 生成): **`:1458`** `ggml_argsort_top_k(selection_probs, n_expert_used)` → `[n_expert_used, n_tokens]`
    - weights(正規化済): **`:1471`** get_rows → **`:1482-1496` で full top-8 上で正規化(split より前=順序救える)**
    - up: **`:1543`** `build_lora_mm_id(up_exps, cur, selected_experts)` → `[n_ff, n_used, n_tok]`
    - gate: **`:1561`** 同上
    - swiglu: `:1607-1609`
    - down: **`:1651`** `build_lora_mm_id(down_exps, cur, selected_experts)` → `[n_embd, n_used, n_tok]`
    - weights 乗算: `:1669` `experts = ggml_mul(experts, weights)`
    - **集約**: `:1680-1696` n_used 個の view を `ggml_add` で総和 → `moe_out [n_embd, n_tok]`
- `deepseek2.cpp:388` の呼び出し引数(1st overload): `cur, ffn_gate_inp, ffn_up_exps, ffn_gate_exps, ffn_down_exps, ffn_exp_probs_b, n_expert, n_expert_used, LLM_FFN_SILU, norm_w, w_scale, gating_func, il, nullptr(probs_in), ffn_gate_up_exps`。
  - **glm-dsa は別々 gate/up**(`ffn_gate_up_exps` は未作成=nullptr)→ build_moe_ffn の **separate gate/up path** を通る[確定]。

### expert tensor 作成
- **`src/models/glm-dsa.cpp:128-130`**: `ffn_gate_exps {n_embd,n_ff_exp,n_expert}`, `ffn_down_exps {n_ff_exp,n_embd,n_expert}`, `ffn_up_exps {n_embd,n_ff_exp,n_expert}`。
  router: `:117` `ffn_gate_inp {n_embd,n_expert}`, `:118` `ffn_exp_probs_b {n_expert}`。
- **per-expert scale `_exps_s` は glm-dsa に存在しない [確定]**(`grep -c exps_s glm-dsa.cpp` = 0)。
  → build_moe_ffn の `up_exps_s/gate_exps_s/down_exps_s` は nullptr。scale ブロック(`:1552/1573/1660`)は全 skip。実装簡素化可。

### backend 強制 + offload + copy 経路
- **backend 手動ピン API [確定・既存]**: `ggml_backend_sched_set_tensor_backend(sched, node, backend)`
  (宣言 `ggml/include/ggml-backend.h:334`, 実装 `ggml/src/ggml-backend.cpp:1960`)。**単一 node を backend に固定**(接続ノードへは scheduler が伝播)。既存使用例 `src/llama-graph.cpp:2063`(attention を CPU 固定)。
- graph context メンバ: **`sched`(llama-graph.h:540/753), `backend_cpu`(:541/755) は利用可**。`backend_gpu` 直メンバは**無い** → §6 参照。
- offload 判定: `ggml-backend.cpp:918-921`(`op_offload && src が host buffer && GPU が supports & offload_op が true`)。
  - CUDA offload_op = `get_op_batch_size(op) >= GGML_OP_OFFLOAD_MIN_BATCH`(既定 **32**, env で変更, `ggml-cuda.cu:5622`)。
  - `get_op_batch_size(MUL_MAT_ID) = op->ne[2] = n_tokens`(`ggml-cuda.cu:5435`)。**decode は n_tokens=1 < 32 → 既定では offload せず CPU 計算**(これが現 26 t/s の正体)。
- **used-expert だけ copy する既存経路 [確定]**: `ggml-backend.cpp:1576-1660`。発火条件 = `node->op==MUL_MAT_ID && input が host WEIGHTS buffer && GPU split`。
  copy は `set_tensor_async`(`:1630`)で **stream0**(=compute と同 stream → 直列)。
  **`GGML_ASSERT(id>=0 && id<n_expert)`(`:1615`)** ← 分割で id 範囲を変える場合の即死点。本案は **id 値を変えない(位置スライスのみ)ので非該当**。

### -ot で pinned host を指定可能にする改修 [確定・実施済]
- **`common/arg.cpp:260-266` に追加済**(このセッションで実装、ビルド確認済): `-ot ...=CUDA_Host` で pinned host buft へ配置可能に。
  ```
  auto * host_buft = ggml_backend_dev_host_buffer_type(dev);
  if (host_buft) buft_list[ggml_backend_buft_name(host_buft)] = host_buft;
  ```
- **注意**: `llama-bench` は独自 -ot パーサで arg.cpp を使わない(`CUDA_Host` 不可)。**テストは `llama-completion`/`llama-cli`/`llama-server`** で。

---

## 6. backend 強制の確定方針 [確定方針 + 一部未解決]
**CPU branch を `backend_cpu` に明示ピン + `GGML_OP_OFFLOAD_MIN_BATCH=1`** で GPU branch を自動 offload、が最小機構:
- CPU branch の down 出力 node を `set_tensor_backend(sched, node, backend_cpu)` で CPU 固定(pin は offload より優先)。
- GPU branch: expert tensor を host(`CUDA_Host` pinned)に置き、min_batch=1 で offload 発火 → 既存 copy 経路で GPU 計算。
- **利点**: graph context に既にある `backend_cpu` だけで足り、`backend_gpu` ハンドル取得が不要。
- **[未解決]** dual-GPU(GPU branch を CUDA0/CUDA1 に二分)するには各 device backend ハンドルが要る。
  取得法候補: `ggml_backend_sched_get_tensor_backend(sched, <CUDA0常駐の既存node>)` で device backend を引く / sched->backends を辿る。次セッションで確定。

### 並列実行が成立する条件 [確定機構 + 未解決の検証]
- scheduler は split を **async 発行**(`ggml-backend.cpp:1678`, callback無し経路は per-split 同期しない)。
- **CPU backend の `graph_compute_async` は同期(ブロック)実行、CUDA backend は非同期(即 return)** [確定: ggml の backend 実装]。
- → **GPU split を CPU split より「先に」発行**すれば、GPU が走ってる間に CPU split がブロック実行 = **並列**。逆順だと直列。
- split 順は graph のノード構築順に依存 → **build_moe_ffn で GPU branch のノードを CPU branch より先に構築**して順序を担保する。
- **[未解決・最重要]** この並列が実際に効くか(壁時計が max(CPU,GPU) になるか sum か)は **実測で確認必須**。
  nsys か nvidia-smi dmon でGPU稼働とCPU稼働が時間的に重なることを確認。**ここが本案の生命線**。

---

## 7. 実装増分(順序固定・各 bit-identity gate)[計画]
各増分は単独でビルド+テスト可能に。正しさ最優先。

| # | 内容 | files | gate |
|---|---|---|---|
| **A** | build_moe_ffn に env-gated CPU∥GPU 分割の **配線のみ**: `selected_experts[n_used,n_tok]` を位置で `[0:k]`/`[k:n_used]` に view スライス → up/gate/down を各 subset で MUL_MAT_ID → 出力を `ggml_concat`(dim=expert軸)で `[.,n_used,.]` に戻す → 既存 weights乗算/集約は不変。**k=n_used(全CPU)/k=0(全GPU) で原経路と数学的同値** | `src/llama-graph.cpp`(+ 引数 or env), `src/models/deepseek2.cpp`(呼び出しに k 伝播, glm-dsa時のみ) | env off で従来と byte一致 / k=n_used で byte一致(=分割実行だが全CPU=同値) |
| **B** | CPU branch を `set_tensor_backend(...,backend_cpu)` ピン + GPU branch を `GGML_OP_OFFLOAD_MIN_BATCH=1` で offload。expert tensor は `-ot ...=CUDA_Host` で pinned host へ | `src/llama-graph.cpp` | byte一致継続 + dmon で GPU PCIe RX 上昇(GPU branch が動いてる証跡) |
| **C** | **並列実証**: GPU branch ノードを先構築し split 順を担保。k を振って(CPU2/GPU6 ～ CPU6/GPU2)decode t/s を計測、26.15 超えを確認。nsys で CPU∥GPU 時間重なり確認 | `src/llama-graph.cpp` | byte一致継続 + **t/s > 26.15** + dmon/nsys で CPU・GPU 同時稼働 |
| **D** | dual-GPU: GPU branch を更に CUDA0/CUDA1 に二分(各 device backend ピン)、両 PCIe 同時稼働 | `src/llama-graph.cpp` | byte一致継続 + 両GPU PCIe同時 + t/s 最大化(目標 ~50) |
| **E** | `--expert-cpu-split k` CLI 化(env hack 卒業)、最適 k の自動/既定値 | `common/arg.cpp`, `common/common.h` | 既定で従来動作 / 指定で C,D の効果 |

**owner 方針**: 実装サブエージェントは **Opus/Sonnet**(Codex不使用、ユーザ指定 [[feedback-implementers-opus-sonnet]])。graph 数式は Opus、機械的部分は Sonnet。

---

## 8. テストプロトコル [確定手順]
- **bit-identity gate(全増分必達)**: `llama-completion -m <model> -ngl 999 -fa 1 --fit off --no-mmap -ot <配置> -c 4096 --no-warmup --seed 1234 --temp 0 -n 64 -p "<固定prompt>"` を増分前後で2回、出力 token 列(text)を `diff` でゼロbyte一致。
  - env off(従来経路)で必ず byte一致 → 回帰ゼロ保証。
  - 分割 on で k=n_used(全CPU)でも byte一致(分割実行が同値の証明)。
- **性能 gate**: 同配置で decode t/s を **26.15(ctx4096)/ 21.79(ctx8192)** と比較。**24スレッド固定**(`-t 24`、SMT厳禁)。
- **並列の実証**: `nvidia-smi dmon -s put -d 1 -o DT` を 30s 観測 → GPU sm util と PCIe RX が **CPU 計算と時間的に重なる**こと。可能なら nsys で copy/compute stream overlap。
- **回帰ガード**: deepseek2/ocr/mistral4 系 GGUF があれば env off で byte一致確認(共有 graph 保護)。手元に無ければ glm-dsa の env off で代替。
- **運用罠**(`glm52-ram-offload-gotchas`): 1ドライバ1ロード厳守(起動前 `pgrep llama`)。ロード~4-5分(237GB)→ 反復は最小化。swap 4GB 極小、二重ロード厳禁。`--no-mmap` 必須(offload時 mmap は推論中ディスク再fault)。`--fit off` 必須(-ngl 999 と auto-fit 衝突)。

---

## ★§9.1 RESOLVED (2026-06-19) — scheduler overlap は成立。動作レシピ確定 [確定・spike実測]
**結論: ggml_backend_sched は CPU split ∥ GPU split を実際に並列実行する(壁時計=max)。但し正しい配線が要る。**
spike: `harness/sched_overlap.cpp`(独立split=97%), `sched_overlap2.cpp`(共有入力+join), `manual_overlap.cpp`(backend直叩き=108%=基礎機構の証明)。ログ=`harness/out/sched_overlap_RECIPE.log`。
- **基礎機構[確定]**: CUDA backend の `graph_compute` は末尾 sync 無し=async(ggml-cuda.cu:4454-4511)、CPU backend は blocking。GPU split を先発行→CPU split が block 実行→重なる。完全独立 split は **97% overlap**。
- **直列化の真犯人 = JOIN**: sched は最終 `add`(2 partial 結合)を **CPU 枝と同一 split に grouping**。その add が partial_gpu(GPU出力)を cross-backend input に要求 → split 冒頭で `synchronize(GPU)`(`cpy_tensor_async(GPU→CPU)` は dst非CUDAで即false → ggml-backend.cpp:1665 の full sync に落ちる)→ **CPU 枝が GPU 枝完了を待つ**。これが naive 実装が serial になる理由(17.94 spike の構造的死因と同根)。
- **勝ちレシピ[実測 86-99.7% overlap]**:
  1. `selected_experts`[n_used,n_tok] を **位置スライス**(GPU=前 8-k / CPU=後 k)。同じ 256-expert tensor で 2 本の MUL_MAT_ID → **2倍計算なし**。ids は実 expert-id のまま=copy_experts が used subset だけコピー、**id remap 不要**(前 hot/cold 案の致命傷を回避)。`ggml_mul_mat_id` 制約は `ids->ne[1]==b->ne[2]` と `ids->ne[0]%b->ne[1]==0` のみ=スライス合法(ggml.c:3291)。decode(n_tok=1)では view 連続=安全。
  2. **inp を CPU へ pre-stage**: `inp_cpu = ggml_cont(inp)` を `backend_cpu` ピン + **GPU 枝より前に `ggml_build_forward_expand`**。CPU 枝が inp の GPU→CPU コピー sync を踏まない(早期 split で attention だけ待つ=安価)。
  3. **CPU 枝**(up_c/gate_c/down_c の mul_mat_id)を `set_tensor_backend(sched, node, backend_cpu)` でピン。
  4. **GPU 枝を先に構築** + `set_tensor_backend(sched, node, backend_gpu)`。backend_gpu = `ggml_backend_sched_get_backend(sched, 0)`(0=最優先GPU)。host weight → copy_experts 発火(used 8-k だけ copy)。
  5. **★ join(moe_out の add)を `backend_gpu` にピン** → 独立した trailing split になり、CPU 枝 split に GPU 入力が混入しない。**これが決定打**(無いと 23%、有ると 96%)。moe_out は下流 attention(GPU)へ流れるので GPU 配置は自然。
- **天井試算**: balance k≈4 で CPU_k≈GPU_(8-k)。単link copy 57GB/s で **~50 t/s**、dual-link 115GB/s で **~60 t/s** 射程(現状26)。GPU 枝 copy が律速になり得るので k は実測で最適化。
- **必要メモリ**: offloaded experts を pinned host(CUDA_Host)に置くと copy が async 化(arg.cpp 改修済 `-ot ...=CUDA_Host`)。~19層×2.92GB=~55GB pinned。RLIMIT_MEMLOCK 要確認(pageable fallback は async copy を同期化し overlap を不可視に潰す=要 LOUD assert)。

## 9. 旧・未解決事項 [§9.1 は上で決着済]
1. ~~[最重要] scheduler が CPU split ∥ GPU split を実際に並列実行するか~~ → **上で RESOLVED**。
2. **dual-GPU 用の device backend ハンドル取得法**(§6)。
3. **最適 k**(CPU/GPU 担当数の balance)。CPU compute rate vs GPU copy+compute rate を実測してから。
   - GPU branch 1 expert = copy 11.53MB/57GB/s ≈ 0.2ms + GPU compute微小。CPU branch 1 expert = ?ms(実測要)。
4. **concat の正しさ/コスト**: `ggml_concat` が expert 軸(ne1)で正しく繋ぐか、view スライスが MUL_MAT_ID の ids として有効か。
5. **CUDA graph 整合**: 本案は graph 構造を変える(分割 on 時)。`ggml-cuda.cu:3260-3268` の MUL_MAT_ID graph 無効化条件に触れないか。split buffer は使わない方針(graph 維持)。

---

## ★実装ステータス (2026-06-19 後半セッション)
- **増分A+B+C 実装済**: `src/llama-graph.cpp` build_moe_ffn の cur reshape 直後(weights計算後)に env-gated `LLAMA_MOE_CPU_SPLIT=k` ブロック挿入。レシピ①〜⑤を全実装(id位置スライス/inp pre-stage cont/CPU枝pin/GPU枝先構築+pin/join GPU pin)。glm-dsa構造ガード(separate gate/up・bias無・scale無・SILU・experts_host・n_tok==1)で他arch保護、env-off は1ノードも足さず byte一致(構造的保証)。
- **proxy smoke-test PASS**: GLM-4.7-Flash(arch=deepseek2=同graph, n_expert64/used4, 13.8GB, 15秒ロード)で `LLAMA_MOE_CPU_SPLIT=2` → `[MOE_CPU_SPLIT] active: n_cpu=2 n_gpu=2` 確認、出力"Paris"= baseline一致、**crash/assert/NaN 無 = グラフ数式は正しい**。proxyでは小モデルゆえ速度は遅化(64.8 vs 79.3、CPU expert非律速+split overhead)=想定内、性能は GLM-5.2 のみで意味。
- **pinned host 確保 OK**: `cudaHostAlloc 60GB` 成功(memlock 8MB を CUDA driver が bypass)。`-ot ...=CUDA_Host` で offloaded experts を pinned 配置可(GPU枝 copy が async 57GB/s に)。
- **実測中(GLM-5.2)**: `harness/test_glm52_split.sh` で baseline vs split k=4 を同config(host=blk3-24 / CUDA0=25-49 / CUDA1=50-78)で連続ロード測定。runtime は rpath で conda CUDA 解決(conda activate 不要)。

## 10. 確定済の地ならし(このセッションで完了)
- `common/arg.cpp` に **`-ot ...=CUDA_Host`(pinned host 配置)を露出する改修済**(ビルド確認済)。
- 計測ハーネス `harness/`: `build.sh`(増分ビルド), `iter_test.sh`(1ロードで正しさ+性能), `derisk_d1.sh`, `sweep_threads.sh`, `poc_overlap.cu`(PCIe/overlap PoC), `spike_pinned_offload.log`(GPU-copy 死亡確認), `out/` に全ログ。
- 物理 PoC `poc_overlap.cu`: pinned H2D 57.7GB/s/GPU・両同時115GB/s・compute/copy overlap 成立(=コピー機構の物理は健全、問題は decode の依存構造)。

---

## 付録: 既に死亡確認済みの道(再検討不要)[確定]
- ❌ 量子化を落として全VRAM化 — 品質崩壊(ユーザ決定)
- ❌ CUDA_Host 直読み(UVA) — discrete GPU 非supports_buft + MMVQ miss storm 7-30x 遅化
- ❌ flag-only / pinned GPU-copy(overlap無) — **17.94 t/s 実測で死亡**(§2)
- ❌ overlapped copy(コピーを GPU 計算裏に隠す) — 層依存 + PCIe 1/30 で構造的頭打ち(§2)
- ❌ CPU dequant のスレッド増 — 24コア飽和済、48(SMT)は逆に 10.5 t/s 崩壊
- △ CPU dequant カーネル高速化(X5) — compute律速なので理屈上有効だが 24コア飽和済で頭打ち、本案と直交(併用可)
- △ MTP/投機 — verify が batch増で expert traffic 増、partial offload と相性悪(保留)

---

## ★★ 2026-06-19 後継セッション: 同期削減 本実装 着手 — WAR guard が主犯と確定 → events で除去

### 計測手段(新規): env-gated per-site sync counter
`ggml-backend.cpp` の `ggml_backend_sched_compute_splits` に `GGML_SCHED_SYNC_COUNT=1` で
**GPU stream を flush する host-blocking synchronize を site 別に集計**する instrumentation を追加
(WAR guard / INPUT copy / MoE input_backend / MoE ids 読み / generic fallback ib・sp)。結果に影響なし、off で zero-overhead。
→ commit `f92b3be` (llama.cpp-glm52 repo)。

### proxy(GLM-4.7-Flash, 46 MoE層全 offload, k=2)で内訳確定 [実測]
**1 decode token あたりの GPU-flush sync(steady-state cumulative/call):**

| site | baseline(split off) | split ON | split が追加 | per-layer 追加 |
|---|---|---|---|---|
| **war**(WARガード 1573) | 49 | 275.5 | **+226.5** | **+4.9/層 ← 最大主犯** |
| moe_ids(ids 読み) | 0 | 67.95 | +67.95 | +1.5/層(1/層は必要) |
| gen_ib(pre-stage GPU→CPU) | 92 | 137.3 | +45.3 | +1/層 |
| gen_sp(join CPU→GPU) | 47 | 47 | 0 | 0(1/層は必要 join 待ち) |
| **TOT** | **194** | **533.8** | **+339.8** | **+7.4/層** |

→ 手記の「split が層あたり ~6 sync 追加」を裏取り。**WAR guard が圧倒的主犯**。
真因: `-ot` で `pipeline_parallel` 無効 → `n_copies=1` → WAR guard が `ggml_backend_event_wait`(stream wait)でなく
host-blocking `ggml_backend_synchronize(split_backend)` に落ち、**GPU split ごとに pipeline flush** → overlap 死。

### Step 1 [実装済・commit `6552024`]: n_copies=1 でも events を確保 → WAR を stream-wait 化
`ggml_backend_sched_new` の event 確保を `n_copies>1` 限定から **`n_copies>1 || LLAMA_MOE_CPU_SPLIT 有効`** に拡張。
- WAR guard が `event_wait`(GPU stream の自己 event 待ち=同一 stream FIFO で既に保証される no-op、**host 非ブロック**)に。
- **`LLAMA_FORCE_PP`(n_copies=4)と違い staging buffer を 4 倍にしない** → GLM-5.2 OOM 回避(これが決定的差)。events は小さい CUDA handle のみ。
- CPU backend は `event_new`→NULL で安全に従来 sync 路。env off の非 split 運用は events 確保せず=stock 挙動・回帰ゼロ。
- **proxy 実測: war 275.5 → 0、TOT 533.8 → 258.3、出力 byte 一致(coherent)**。proxy eval 53→63 t/s(CPU 非律速でも改善)。

### 残る同期(Step 1 後, proxy 値)= 大半が本質的に必要
- moe_ids ~1.5/層(GPU 上 ids 読み, manual も 1/層)、gen_ib ~3/層(pre-stage GPU→CPU, 先頭1個が attention 待ち=必要・残りは idle GPU で安価)、gen_sp ~1/層(join の GPU expert 待ち=必要)。
- **pre-stage は events 後も必須**: 廃すると CPU 枝が GPU-resident inp/sel/weights を読む→CPU split で synchronize(GPU) が
  **issue 済の GPU expert 枝を待つ**→overlap 死。pre-stage は GPU→CPU sync を expert 枝 issue **前**(attention 待ちのみ)に移す役。
- 次手候補(GLM-5.2 で Step1 が break-even 止まりなら): (a) pre-stage 3 input を pack して gen_ib 3→1、(b) join を CPU↔CUDA async copy 化(pinned 前提)。ただし gen_sp/moe_ids は本質。

### GLM-5.2 実測 [進行中]: `harness/test_glm52_events.sh 4`(baseline vs split k=4 + SYNC_COUNT + dmon, ~12分)

### 単一GPU k-sweep 結果 [実測, 2026-06-19 後継]
GPU枝が単一link copy律速 → CPU寄り(k↑=GPU expert↓=copy↓)が効くと予測 → 実測で確認:
| k (CPU experts) | n_gpu | decode t/s | vs baseline 23.74 |
|---|---|---|---|
| 4 | 4 | 23.62 | 0.995x (break-even) |
| 5 | 3 | 24.68 | 1.04x |
| **6** | **2** | **26.99** | **1.14x** ← 単一GPU最適付近 |
| 7 | 1 | (測定中) | |
**含意**: 単一link では GPU expert を 2個程度に絞るのが最適(copy ~0.5GB/token=~10ms が CPU枝6expert と均衡)。+14% は出るが goal 43 には dual-GPU 必須。dual なら copy 帯域2倍で GPU expert を増やせる(k↓)→ さらに上。

### timing 確定値 [GLM-5.2 split k4, GGML_SCHED_SYNC_COUNT=3, steady-state/token]
in_gpu(GPU待ち,join含)=18ms / cmp_cpu(CPU枝4exp)=10.5ms / cmp_gpu(GPU async発行)=0.5ms / in_cpu(prestage)=5ms。
→ overlap成立(GPU async)、だが GPU枝が copy律速(~1GB/token/50GB/s)。**dual-GPU 実装済 (commit 8037eeb, env LLAMA_MOE_DUAL_GPU=1)、test_dual_gpu.sh で実測予定**。
