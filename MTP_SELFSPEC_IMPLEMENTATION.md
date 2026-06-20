# GLM-5.2 MTP self-spec 完成 — 実装方法とゴール(handoff / 設計図)

> 生成 2026-06-20。決定 = **A(cheap draft + 1-token verify を完成させ +34〜46% lossless を取る)**。
> このドキュメントを Read 1回で実装を引き継げる。前提 memory: `project_glm52_speedup.md`(★8〜★11 が最新)、
> `project_step37_mtp.md`(Step3.7 MTP の前例)。ソース = `~/src/llama.cpp`(branch glm52-cpu-gpu-moe-split)。

---

## 0. 一行サマリー
GLM-5.2 内蔵の nextn(MTP)head は次トークンを **α=0.72 で当てる(実証済・trusted)**。これを実 decode 速度に変える
self-spec ループの **naive 実装は失敗(16.17 t/s, −36% regression, lossless も FAIL)**。**正しい実装 = cheap
layer-78-only draft + 真の 1-token verify** で **production 27.58 比 +20〜51%(中央 ~+35%, 要 cheap draft 実測, lossless)** が予測。
本体(cheap draft mode)は**未着手**で、**敵対レビューが3 BLOCKER を発見(§3.5 必読・最重要)**:
① hidden 注入は新 input class 必須(Option A-direct は構造的に不可) ② 1-token verify は真の 1-token batch 必須(seqlen-2 re-commit は union tax 残す)
③ graph reuse topology flag 必須(無いと silent corruption)。工数 ~3-5日。

## 1. ゴール(Endgame)
GLM-5.2 754B(UD-Q2_K_XL 236GB, expert 224GB > VRAM 192GB ゆえ 19層 CPU offload 不可避)を 2× RTX PRO 6000 +
Threadripper 9965WX で限界まで速く。**量子化は UD-Q2 固定(劣化不可)**。確定 baseline = **27.58 t/s(A1+A3, push 済)**。
**MTP self-spec で +34〜46% = 37-40 t/s を lossless(greedy byte-identical)で達成する**のが本タスクのゴール。
論文化興味なし、X 記事化は候補。repo は public hunt-log。

## 2. これまでの確定成果(数字 — 盛らず)

| 項目 | 値 | 状態 |
|---|---|---|
| **probe α**(MTP head の次トークン一致率) | **0.7212** | **trusted**(offline numpy cross-check cos≈1.0, negative control 判別力証明) |
| spec-OFF baseline(tool, -t 24, A3 pin 無し) | 25.24 t/s | 実測 |
| **production 確定 baseline**(A1+A3, push 済) | **27.58 t/s** | 確定(比較の基準) |
| **broken self-spec(naive fused)** | **16.17 t/s(−36%)** | honest negative + **lossless FAIL** |
| └ model forwards / tokens | 277 / 256 = **0.92 tok/fwd** | (n_fwd/tok≈2.0 予想は外れ) |
| └ real-loop α | **0.44**(chain bug で probe 0.72 から低下) | 想定内 |
| └ drafts proposed / accepted | 177 / 78 | |
| verify forward cmp_cpu(2-token, steady) | **~30-34 ms/forward** | SYNC_TIME |
| base wall / verify wall | 39.6 ms/token / 57.1 ms/forward | |
| **union tax(2-token verify の追加コスト, wall)** | **~17.5 ms/forward** | (= 57.1 − 39.6) |

**regression の主因 = ① union tax(2-token verify が 1-token base より +17.5ms)② real-α 低下(chain bug)**。
n_fwd/tok は ~1.08(2.0 でない)— broken でも forward 数はほぼ 1/token だが、各 forward が重く・α が低い。

**★さらに lossless GATE = FAIL(最重要・broken の数字を二重に無価値化)**: spec-ON vs spec-OFF を手動 token diff →
token 0-172 は byte 一致、**token 173(0-idx)で分岐**(spec-OFF=`355,11,279,...` / spec-ON=`355,53296,13,6156,355,572,...`
= 余分トークン挿入 + 順序乱れ)。spec-ON text 自体は coherent だが greedy 列が spec-OFF と非一致 = **lossless 違反**。
これは「1ズレ chain bug」とは別の **broken loop の受理/KV 会計バグ**(構成上 v==d 時のみ d を残す設計なのに分岐 → 会計がズレ)。
**最有力容疑 = 的中時の `out += w`(bonus w = logits@p+1 の argmax だが KV position / push 順序がズレ)**。
→ **16.17 t/s は lossless FAIL ゆえ二重に無価値**。**この会計バグは追跡不要 — cheap draft 実装で正しい会計に書き直す**。
教訓: lossless は **構成的保証で死守**(v==d 時のみ d を残す、bonus w は logits@p+1 argmax = target greedy と一致するはず)。
毎 run token diff(`/tmp/mtp_spec_on.txt` vs `S[n_prompt:]`)で 1 token でもズレたら STOP。

### 予測(正しい実装 = cheap draft + 1-token verify)
```
1-token verify forward ≈ base decode(union tax 無し)        = ~39.6 ms
cheap draft(layer-78 のみ, base cmp_cpu の 1/19)            = ~3 ms
chain 修正で real-α 0.44 → 0.6〜0.72(probe 値に接近)

ms/token = (39.6 + 3) / (1 + real-α)
  real-α 0.60 → 26.6 ms = 37.6 t/s   (production 27.58 比 +36%)
  real-α 0.72 → 24.8 ms = 40.4 t/s   (production 27.58 比 +46%)
```
**予測 37-40 t/s = +34〜46%(lossless)**。不確実性 = real-α の戻り具合 + cheap draft 実 cost + 実装リスク。

## 3. 正しい実装の設計(cheap draft + 1-token verify)

### なぜ naive fused が失敗したか(構造的、再発防止)
naive fused は verify batch `[a, d]` の中で next draft を col0 = `nextn(h_a, emb(d))` として計算した。だが:
- **chain が延びない**: hit 後に bonus token `w`(logits@p+1)を取ると、次に必要な draft = `nextn(h_{p+1}, emb(w))`。
  **`w` は forward の OUTPUT で入力 batch に存在しない** → col0(入力 emb(d) 使用)では作れない(EAGLE chicken-egg)。
  miss 時も recovery forward が要り、結局 forward 数が膨らむ。
- **union tax**: verify が 2-token batch で expert を ~1.8-2.0× 読む(mul_mat_id は gemv per-token、batch 償却ゼロ)。

### 正しい設計(2点)
**(A) cheap draft step を独立に持つ(layer-78 only graph)**
各サイクルで、verify forward が出した **h_prenorm(最後に accept した position の main hidden)** + **emb(last token)** を
**layer-78 の nextn branch だけ**に通して次 draft を生成。layers 0-77 をスキップ → **~1-3ms(base cmp_cpu の 1/19、
layer-78 expert は 92MB vs 全 offload 1.75GB)**。これが EAGLE 標準(draft model = 1層)。GLM の nextn block が 1層
だから安い、が成立条件。full graph で draft すると 36ms で 2-forward に戻り負ける(実証済: broken の recovery がこれ)。

**(B) 1-token verify(union tax を回避)**
verify batch を 2-token `[a,d]` から **1-token `[d]`** に。**前サイクルの logits@a を再利用**して d を verify
(d == argmax(prev logits@a)?)。これで verify forward が 1-token = base 同等(cmp_cpu 1×)。union tax(+17.5ms)を払わない。

### chain(1 forward/cycle で 1+α token)
```
state: 最後に確定した token、その h_prenorm、次 draft d
cycle:
  1. verify forward [d] @ pos p (1-token, all-logits)
     → logits@p (= d の次 = v), h_d (= d の main hidden)
  2. accept: d == argmax(prev_logits)?  (前サイクルが出した「d の verify 基準」)
     hit  → d 確定(+ v も確定なら 2 token)。
     miss → d 破棄、KV seq_rm。確定は prev の正解。
  3. cheap draft step (layer-78 only): next_d = nextn(h_d, emb(d))  ← h_d を host 経由で注入
  4. next cycle で next_d を verify
lossless: d を「prev logits の argmax と一致」した時だけ残す → spec-OFF と byte-identical(構成的保証)
```

## 3.5 ★敵対レビュー反映(必読・実装前に最重要 — 本文 §3 の楽観を訂正)

敵対レビュー(4 critic, 396k tok, 現状コードと照合)が §3 に **3 BLOCKER + 予測の楽観**を発見。
**本文の「Option A-direct」「cheap draft 3ms」「予測 +34-46%」は以下で上書きされる。**

### ★BLOCKER 1(最大の欠け): hidden 注入は「新 input class 必須」、Option A-direct は不可能
- §3-A の「build_inp_embd の dual-path(ubatch.embd で h + ubatch.token で emb、新 class 不要)」は **構造的に不可能**。
  build_inp_embd(`llama-graph.cpp:1937`)は `ggml_build_forward_select(.., ubatch.token?0:1)` で **バッチ全体で token XOR
  embd を build 時に1つ選択**(per-token でない)。両方は無理。さらに `tok_emb_all = inpL = build_inp_embd`(glm-dsa.cpp:190-193)
  なので、embd path で h を注入すると **emb_shift が h の関数になり nextn math が壊れる**(silent、α 崩壊、crash 無し)。
- **正しい実装**: 新 graph input class `llm_graph_input_mtp_hidden`({n_embd,N} F32, `ggml_set_input`, set_input override で
  host の h_d を `ggml_backend_tensor_set`)。draft-only mode で: **h = この injected tensor / emb_shift = token path の
  build_inp_embd(維持)**。h_d は verify forward の `nextn_h_prenorm-78`(glm-dsa.cpp:430 で命名済)を cb_eval で readback
  → context-side ポインタで次サイクルに注入(llama_batch.embd 経由でなく)。

### ★BLOCKER 2: 1-token verify は「真の 1-token batch」でだけ cmp_cpu 1×(seqlen-2 re-commit と矛盾)
- gotcha #4「1-token verify でも KV 上 prev 再commit して seqlen-2」は **union tax を消さない**(2-token で expert 2× 読む、
  cmp_cpu ~30ms のまま = broken と同じ)。+34-46% は cmp_cpu 1× が前提なので、これでは利得が消える。
- **正しい実装**: verify = **真の 1-token batch `common_batch_add(b, d, p, {0}, true)`、prev 再commit なし**
  (prev は前サイクルの verify forward で position p-1 に commit 済が前提)。これで cmp_cpu 1×。
- **prev logits は volatile**: `logits.data` は毎 decode で offset 0 から上書き(`llama-context.cpp:1750`)。
  **argmax(prev) を host 変数に保存**してから次 decode(greedy は token id 1個で十分)。accept = `d == t_acc_prev`。
- `common_sampler_sample_and_accept_n` は使えない(`idxs.size()==draft+1` を assert, sampling.cpp:621)→ tool の raw
  argmax accept(mtp-alpha.cpp:366-394)を維持。lossless は argmax 決定性(strict `>`, first-index-wins)に依存、temp/seed 無関係。

### ★BLOCKER 3(silent corruption): graph reuse topology aliasing
- topology flag が無いと、process_ubatch(`llama-context.cpp:1192`)が full-verify graph を draft-only forward に
  (or 逆)**同じ n_tokens で silently 再利用**。allow_reuse(`llama-graph.h:573`)は ubatch shape のみ key。
  → wrong layers 実行、crash 無し、lossless diff のみが検出。
- **正しい実装**: `llm_graph_params` に topology discriminator(uint8 graph_variant)追加 → llama_decode 毎に set →
  glm-dsa.cpp で 0-77 loop の skip gate に使う → **allow_reuse の key に入れる**。検証で `graph_reuse_disable` を立てた
  control run と比較(lossless が reuse-disable でだけ通れば aliasing バグ)。

### KV recycle invariant の正しい条件(gotcha #1 を強化)
- 正しい invariant = **「layers 0-77 が attend する全 position は full forward で commit 必要」**(seq_rm-before-reread だけでは
  不十分)。hit 時に accepted position の 0-77 K が draft-only でしか書かれてないと、後の full forward が stale 0-77 K を読む
  (silent drift, kq_mask は pos/seq のみで stale を検知不能)。**accept path は accepted position を full forward で再commit してから読む**。
- MLA ゆえ V cache 無し(`has_v=false`, llama-kv-cache.cpp:208)→ rollback は seq_rm(cell metadata)+ layer-78 K 上書きのみ。
  0-77 は seq_rm で cell 解放 → 次 full forward が `cpy_k` で上書き(implicit)。
- seq_rm order: draft-only forward 前に `seq_rm(0, p_draft, -1)`(layer-78 cell を新鮮化)。miss 時 `seq_rm(0, p+1, -1)`(wrong d 破棄)。
- **n_outputs**: draft-only forward は全 position logits=true(n_outputs==n_tokens)必須(guard glm-dsa.cpp:406-407)。
  「1-token verify」は verify batch 幅の話で、draft forward の n_outputs を減らす意味ではない。

### 予測の訂正(pinned baseline + cheap draft 実測、楽観を下方修正)
- §2 の base 39.6ms は **un-pinned tool baseline**、production 27.58(**pinned 36.3ms**)と比較は apples-to-oranges → 膨張。
- cheap draft「3ms / 1/19」は楽観: CPU offload layer は **23(19 でない: blk.0-21 + blk.78, inc3_selfspec.sh:16)**、
  base cmp_cpu 16.3ms。layer-78 attention は **full KV を読む(1/19 でない)** + head GEMM(model.output 6144×151552 on GPU ~1ms)
  + 固定 per-forward overhead(~20ms の残差)。**cheap draft cost は実測必須、5-12ms 見込み**。
- **正直な再予測** `ms/token = (36.3 + draft_ms) / (1 + real-α)`:
  - 楽観(draft 5ms, α 0.72): 24.0ms = 41.6 t/s = **+51%**
  - 中央(draft 8ms, α 0.65): 26.8ms = 37.3 t/s = **+35%**
  - 悲観(draft 12ms, α 0.60): 30.2ms = 33.0 t/s = **+20%**
- **正直な幅 = +20〜51%(中央 ~+35%)**。支配的不確実性 = cheap draft cost + real-α。**cheap draft を実測してから確定**。

### memory ★182-183 との reconcile(必須 — でないと「死んだ案」と誤読される)
- memory project_glm52_speedup.md ★182-183 は「BW-bound で cheap-MTP は ~0、27.6 が天井、超えるには GEMM(週単位)」と結論。
  本ドキュメントの「+20-51%」と矛盾に見える。**reconcile**: memory の批判は「batch=K は 1 forward で expert を K回読む
  (bytes/token 不変)」。だが **1-token verify は 1 full forward(1.75GB expert 読み 1回)を 1+α token に償却**するので
  bytes/token が 1/(1+α) に下がる = **BW 壁の loophole**。これが GEMM 不要で天井を超える理由。**この loophole の成立が本タスクの賭けの核心** —
  cheap draft が本当に安く(<< verify)、real-α が高ければ成立、どちらか崩れれば memory の天井に回帰。

### 改訂後の正直な工数・期待値
新 input class + topology flag + KV re-commit recipe + prev-logits 保存 = 当初「2-3日」より重い、**~3-5日**。
成功(+51%)/中央(+35%)/悲観(+20%)/失敗(silent drift で lossless 落ち撤退=0)。lossless 保証ゆえ出れば確実だが、
**day-1 blocker(hidden 注入 input class)を超えるまで t/s は不明**。最初に作るべき = 新 input class + 真の 1-token verify、
そして毎変更で lossless token diff を死守。

## 4. 実装手順(段階的 — 一度に変えない、各段で数字を切り分け)

- **段階A(完了)**: broken full-graph spec(LLAMA_MTP_SPEC)動作確認 = 16.17 t/s(honest negative)。
- **段階B(次, 本体)**: cheap draft mode `LLAMA_MTP_DRAFT_ONLY` を glm-dsa.cpp graph に追加。
  - layers 0-77 を skip、layer-78 nextn branch のみ実行。
  - h_prenorm を `build_inp_embd` の **vector path(ubatch.embd)** で注入、last token は `ubatch.token`(dual path, 新 input class 不要 = Option A-direct)。
  - **まず 2-token verify のまま** cheap draft で chain を修正 → real-α が probe 0.72 に接近 + lossless PASS を確立。
  - これだけで「chain 修正単独の利得」が見える(union tax は残るので t/s は中間)。
- **段階C**: 1-token verify 最適化(前サイクル logits 再利用 + KV 整合)→ union tax 1.8×→1× → t/s 37-40 目標。
- **各段の gate**: ① lossless token diff PASS(非交渉)② real-α ③ wall-clock t/s ④ SYNC_TIME per-forward。

## 5. 難所と gotcha(全部 — ここで詰まった/詰まる)

1. **★KV recycle invariant(最難所, silent-drift リスク)**: draft-only forward は layer-78 K だけ書く。layers 0-77 K は
   position p で stale。不変条件 = **「draft-only が書いた position は、layers 0-77 KV を読む前に必ず full forward で
   seq_rm + 再commit」**。間違えると **crash せず silently logits drift**(layer-78 は valid history を読むので動くが微妙にズレる)。
   **検出は lossless token diff のみ** → これを毎 run 死守。1 token でもズレたら即 STOP。
2. **hidden = h_prenorm(main layers 0-77 出力)、`nextn_block` 出力ではない**。nextn 定義 `draft=nextn(h_i, emb(t_{i+1}))`
   の h_i は main hidden(probe の入力 `nextn_h_prenorm-78`)。間違えると real-α が過小評価される。
3. **out_ids 縮約**: nextn branch は **n_outputs == n_tokens の forward だけ engage**(guard `h_prenorm->ne[1]==tok_emb_all->ne[1]`
   が glm-dsa.cpp:407 に入済)。prompt prefill(最後だけ logits, n_outputs<n_tokens)では h_prenorm が縮んで emb_shift と
   concat shape mismatch → ggml.c:2634 abort。
4. **seqlen-1 禁止**: seed/recovery を seqlen-1 でやると `ggml_view_2d(ne=0)` → ggml_concat assert。**全 forward seqlen-2 統一**
   (draft `nextn(h_prev,emb(tok))` = `[prev,tok]` の col0 を使う)。1-token verify でも KV 上は prev を再commit して 2-token batch。
5. **nextn テンソルは `LLM_TENSOR_LAYER_REPEATING`**(LAYER_OUTPUT でない)。LAYER_OUTPUT だと loader が「output 層に layer
   number」で abort(llama-arch.cpp:760-767, 修正済)。
6. **col0 が valid draft**(emb(batch[1]) 使用)、**col1 は junk pad**(batch[2] 不在)。cb_eval は col0 固定。
7. **math**(vLLM `deepseek_mtp.py`/`glm4_moe_mtp.py` で検証済): concat order **e=enorm(emb) THEN hn=hnorm(h)**;
   x = eh_proj·concat; o = **shared_head_norm**(block)[output_norm でない]; head/embed は **tied to model.output/tok_embd**
   (nextn.embed_tokens / nextn.shared_head_head は GGUF に absent)。
8. **GEMM は verify では効かない**(検証済): `ggml_gemm_q2_K_8x8`(repack.cpp:1986、配線のみ欠)は 8-row tile。verify batch=2
   では tile が埋まらず、routing overlap 0.125 で共有 expert がほぼ無く dedup 相手がいない。GEMM が効くのは prefill(batch 大)
   で decode t/s に直接効かない。**union tax を潰すのは GEMM でなく 1-token verify**。GEMM/Q3_K カーネル工事は不要。
9. **lossless は greedy 限定**: `--temp 0 --seed 固定`。token diff は `/tmp/mtp_spec_on.txt` vs `S[n_prompt:]`。
10. **t/s は wall-clock(perf eval-time でなく)**: spec では eval-time が forward 数を数え、生成 token 数でない。
11. モデルロード ~3分(--no-mmap 236GB)。1ロード制約。layer-78 experts は CPU 配置(-ot, CUDA1 tight)。GPU0 desktop <3.6GB。

## 6. 現状のコード状態(working tree, 未コミット)

`git diff --stat`: 9 files +341/-11 + 新規 `tools/mtp-alpha/`(untracked)。**未コミット**。

| ファイル | 状態 | 内容 |
|---|---|---|
| `include/llama.h`, `common/{arg,common}.{h,cpp}`, `src/llama-model.cpp` | ✅ 完成 | `--enable-mtp` flag plumbing(INC-1) |
| `src/llama-arch.cpp:760-767` | ✅ 完成 | nextn → LAYER_REPEATING(abort 修正) |
| `src/models/glm-dsa.cpp` | 部分 | INC-1 load gate + INC-2 own graph + nextn branch。**env mode = LLAMA_MTP_PROBE / LLAMA_MTP_SPEC の2つ**。out_ids guard(:407)入済。**`LLAMA_MTP_DRAFT_ONLY`(layers 0-77 skip)は未実装** ← 段階B の本体 |
| `src/models/models.h:1008-1016` | ✅ 完成 | glm_dsa own graph subclass(deepseek2 から decouple) |
| `tools/mtp-alpha/mtp-alpha.cpp` | 部分 | probe mode(α 測定, 完成)+ self-spec loop(broken full-graph 版, fused/seed/recovery, cb_eval col0, seq_rm 再commit)。**lossless FAIL(会計バグ)**, **1-token verify 未, cheap draft 注入 未** |

**broken self-spec loop の現状構造**(mtp-alpha.cpp ~226-, 次の実装者が cheap draft で書き直す元):
- seed: `draft_only(prompt_last, p-1, a)` = seqlen-2 `[prev,a]@[p-1,p]` 両 logits、col0 cb で d 取得。
- fused(毎ループ): `seq_rm(0,p,-1)` → `[a,d]@[p,p+1]` 両 logits → v=argmax(@0), w=argmax(@1), dn=col0。
- 受理 v==d: out+=d, out+=w, a=w, d=dn, p+=2。外れ: `seq_rm(0,p+1,-1)`, a=v, p+=1, out+=v, recovery `draft_only(a_prev,p-1,v)`。
- **これが lossless FAIL の loop**。cheap draft + 1-token verify で正しい会計に書き直す(構成的 lossless を死守)。

**= 残り作業 = 段階B(cheap draft mode)+ 段階C(1-token verify)**。INC-1/INC-2/broken は完了。

## 7. 測定インフラ

- 起動 config(server.sh 由来): `GGML_CUDA_NO_PINNED=1 GGML_CPU_HUGEPAGE=1`、`-ngl 999 -fa 1 --fit off --no-mmap
  -ctk q8_0 -ctv q8_0 -ot "<offload, blk.78 exps=CPU>" -t 24`。A3 pin(`--cpu-mask FFFFFF --cpu-strict 1`)は tool 未適用
  → 最終測定では足すと +3.4%。
- spec-ON/OFF A/B: `harness/inc3_selfspec.sh`(STEP A spec-OFF gen + t/s、STEP B spec-ON self-spec、GATE lossless diff)。
- timing: `GGML_SCHED_SYNC_COUNT=3` で `[SYNC_TIME]`(in_gpu/cmp_cpu/cmp_gpu/in_cpu per forward)。
- α probe: `LLAMA_MTP_PROBE=1` + `llama-mtp-alpha`(N=256 all-logits prefill)。offline math: `harness/inc2_mathcheck.py`
  (conda 罠回避: `export NVCC_PREPEND_FLAGS=""` してから conda activate、私の Bash は set -u 無しで OK)。
- broken 数字ログ: `harness/out/inc3_specon5.log`(16.17 t/s, real-α 0.44, SYNC_TIME)。

## 8. 必読資料(最小限)

| Path | なぜ |
|---|---|
| `~/projects/glm52-speedup/INC2_nextn_probe_design.md` | nextn graph 設計(N列 FIX, hidden, math, env-gate, gotcha) |
| `~/projects/glm52-speedup/DECISION_alpha_to_route.md` | α → route 予測モデル(27.58 baseline, breakeven α≈0.36) |
| `memory: project_glm52_speedup.md`(★8〜★11) | A1/A3 勝ち、split死、INC-0.5、BW天井の全結論 |
| `memory: project_step37_mtp.md` | Step3.7 MTP 前例(K=1 最適、NVFP4 非対応の罠) |
| `src/models/deepseek2.cpp:196-423` | per-layer body(run_layer ラムダの元、glm_dsa が継承) |
| `src/models/glm-dsa.cpp:385-460` | nextn branch(probe/spec mode、guard、cheap draft を足す場所) |
| `src/llama-graph.cpp:1877-1959` | build_inp_embd vector path(h_prenorm 注入の mechanism) |
| `tools/mtp-alpha/mtp-alpha.cpp:226-` | self-spec loop(broken 版、cheap draft + 1-token verify に拡張) |
| `harness/out/inc3_specon5.log` | broken 実測(16.17, real-α 0.44, SYNC_TIME) |

## 9. 正直な期待値(賭けの大勝ち)

- **成功**(real-α 0.7, 実装 OK)→ +46%(40 t/s)。
- **部分**(real-α 0.6, cheap draft cost 高)→ +34%(37 t/s)。
- **失敗**(KV invariant で silent drift, lossless 落ちる)→ 0(撤退、27.58 維持)。
- 工数 ~2-3日。**lossless(品質劣化ゼロ)が保証**ゆえ、出れば確実な大勝ち。支配的リスク = KV recycle invariant の silent-drift。
- 完成後: A3 pin を tool に足す + production config で最終測定 → 正直な最大 t/s を確定し、壁の式とともに記録。
