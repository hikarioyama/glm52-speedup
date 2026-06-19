# PLAN C — full roadmap: bandwidth quick-wins → MTP self-speculation

> Status: **strategic roadmap.** Phase 0 = PLAN A (do first, certain ~+10–15%).
> Phase 1+ = GLM-5.2 native MTP self-speculation. **Read §2 "Honest payoff" before committing** —
> the MTP upside is materially lower than the first optimistic model, for two independently-measured
> reasons. MTP is lossless (greedy byte-identical) so it cannot hurt quality, but the throughput math
> is now "marginal-to-modest, acceptance-dependent, multi-day risky build", not the "+50% slam-dunk".
>
> Generated 2026-06-19 from the MTP design+payoff+bandwidth workflow. Repo: `~/src/llama.cpp`
> (branch `glm52-cpu-gpu-moe-split`). Companion: `PLAN_A_bandwidth_quickwins.md`.

---

## Phase 0 — PLAN A (prerequisite)
Do `PLAN_A_bandwidth_quickwins.md` first. It is certain, cheap, orthogonal, and a faster `cmp_cpu`
**directly shrinks MTP's dominant verify term** (the MTP verify forward is `cmp_cpu × union_mult` +
amortized constants). Target after Phase 0: ~28–30 t/s.

---

## 1. What MTP is and why it was the candidate
Autoregressive decode = 1 forward / 1 token, and each forward reads ~1.75 GB of offloaded experts from RAM
(bandwidth-bound). **Speculative decoding** lets a cheap *draft* propose K tokens and the expensive *target*
**verify all K in ONE batched forward** — lossless via rejection sampling (greedy ⇒ byte-identical output).
**MTP (Multi-Token Prediction)** makes the draft the model's **own** built-in `nextn` head (DeepSeek-V3/EAGLE
style), no separate draft model. GLM-5.2 ships this head (the `blk.78.nextn.*` tensors).

**The hoped mechanism:** at batch=K the CPU is far from compute-bound (stays BW-bound until ~K=16), so extra
verify tokens are "free on FLOPs"; if they route to the **same experts**, the expert dequant is amortized →
per-token RAM bytes drop → break the bandwidth wall **without touching quality**.

---

## 2. HONEST PAYOFF — measured, and weaker than first modeled

### 2a. Expert-routing overlap is LOW (INC-0, **measured**, the decisive gate)
Analyzed `harness/out/expert_ids.jsonl` (120-token real generation, top-8/layer, offloaded layers 3–21):

| metric | measured | model's assumption |
|---|---|---|
| consecutive-token top-8 shared fraction `o` | **median 0.125 / mean 0.231** | low=0.30 / med=0.50 / high=0.70 |
| union_mult(K=1) | **1.77×** | low 1.70 |
| union_mult(K=2) | **2.45×** | low 2.33 |
| layers with median `o` < 0.4 | **19/19** | gate wanted o>0.4 |

**GLM-5.2's router spreads consecutive tokens across nearly-disjoint experts** (only ~1 of 8 shared) — MoE
load-balancing working as designed. This **fails the workflow's own GO gate (o>0.4)** and lands at/below its
"low overlap" row. (Re-measure on the *target* workload before building — but note code/JSON/structured output
is predicted *lower* overlap, so prose is already near the best case.)

### 2b. The current code cannot amortize shared experts anyway (**code-referenced**)
Even where experts ARE shared, today's `ggml_compute_forward_mul_mat_id`:
- **generic path** re-fetches the weight per `vec_dot` call (`ggml-cpu.c:~1507`) — per (row,col), so batch>1 does **not** reuse the dequantized weight across tokens;
- **repack path** loops `gemv` per token row (`repack.cpp:4498-4514`) — re-streams the weight per token;
- the real reuse kernel **`ggml_gemm_q2_K_8x8` (4 rows × 8 cols) EXISTS (`repack.cpp:1986`) but is NOT wired into `mul_mat_id`**; and **Q3_K has no repack/GEMM kernel at all**.

> ⚠️ **UNRESOLVED DISAGREEMENT to settle first.** The synthesis agent claimed the generic `one_chunk` path
> already amortizes across a 16-row block (`blck_0=blck_1=16`, `ggml-cpu.c:1480-1508`); the bandwidth agent
> claimed per-`vec_dot` refetch with no reuse. **These contradict and decide whether MTP can work without a
> GEMM rewrite.** Resolve by reading `ggml_compute_forward_mul_mat_id` end-to-end (does the 16-row tile reuse
> the dequantized `src0` tile across the 16 token columns, or re-dequantize per column?) BEFORE writing MTP code.

### 2c. Resulting honest expectation
Plugging **measured** union into the model (constants amortize across accepted tokens; in_cpu prestage 4.3 ms is per-decode so amortizes outright):

| K | accept | est t/s | vs 25 |
|---|---|---|---|
| 1 | 0.5 | ~27 | +5% |
| 1 | 0.7 | ~30 | +19% |
| 1 | 0.85 | ~33 | +30% |
| 2 | (union 2.45×) | hurt by union tax | marginal beyond K=1 |

- **Cannot regress at K=1** (the 19.4 ms of per-forward constants amortize faster than union grows).
- Upside (+30%) needs **acceptance ≥0.85**, which is **unmeasured** and only knowable after INC-2/3.
- **K≥2 is union-taxed** (2.45×) → K=1 is the safe ceiling on this routing.
- And **all of the above assumes the amortization is realized in code** (§2b) — otherwise MTP buys ~0.

**Net:** MTP = lossless +5–30% (K=1, acceptance-dependent), **conditional on** (i) the GEMM-amortization being
built/confirmed (§2b) and (ii) acceptance clearing ~0.45. The "+50–126%" of the first model assumed routing
overlap the data refutes. Decide with eyes open.

---

## 3. Feasibility (workflow finding): hard-but-feasible, all pieces present
- The 6 `nextn` tensors + a **full** layer-78 attn+MoE block are **physically in the UD-Q2_K_XL GGUF** (confirmed from load logs): `eh_proj, enorm, hnorm, shared_head_norm` present; `embed_tokens`/`shared_head_head` **absent because tied** to `tok_embd`/`output` (standard DeepSeek-V3 MTP) — **no re-quant needed**.
- Tensors are loaded-as-ignored today (`TENSOR_SKIP`, `glm-dsa.cpp:139-148`; mappings `llama-arch.cpp:448-453,760-767`; hparam `nextn_predict_layers` in `llama-hparams.h:93`).
- `common/speculative.cpp` has a pluggable impl framework (`common_speculative_impl_draft_simple` as template; an `eagle3` slot is a TODO stub at ~:343-365).
- **Genuinely hard:** no graph path consumes the nextn tensors; the pre-norm hidden-state handoff for EAGLE-class self-spec is unimplemented.

### nextn head math (GLM/DeepSeek-V3 MTP, infer-confirm against shapes)
Layer 78 = `n_layer-1`, the single `nextn_predict_layer`. To propose t+1 given the target just produced t at position p with **pre-output-norm** hidden `h_p`:
1. `e  = enorm(embed_tokens[t])`  (embed tied to `tok_embd` if absent)
2. `hn = hnorm(h_p)`
3. `x  = eh_proj(concat(e, hn))`   `{2*n_embd}→{n_embd}`
4. run **one full transformer block** using layer-78's OWN MLA attn (`wq_a/wq_b/wkv_a_mqa/wk_b/wv_b/wo`, `attn_norm`) + OWN MoE (`ffn_gate_inp`+`ffn_*_exps`+`shexp` via `build_moe_ffn`); KV in a private layer-78 slot
5. `o = shared_head_norm(block_out)`; `logits = shared_head_head(o)` (head tied to `output`/`tok_embd` if absent)
6. argmax/sample → draft t+1. For K>1: iterate the same module EAGLE-style (only 1 MTP block ships).

---

## 4. Implementation roadmap (gate-able increments)

- **INC-0 — routing-overlap validation — DONE.** Result: median o=0.125, union 1.77/2.45 → **fails the o>0.4 gate** (see §2a). Re-run on the *target* workload if it differs (`harness/40_analyze_reuse.py` + the consecutive-pair script used in this session). **Decision rule:** o>0.4 ⇒ strong build; 0.3–0.4 ⇒ K=1-only marginal build; <0.3 (current) ⇒ build only if acceptance is expected very high AND §2b amortization is confirmed, else ship PLAN A and stop.
- **INC-0.5 — resolve §2b (code read, ~0.5d, NEW hard gate):** read `ggml_compute_forward_mul_mat_id` end-to-end; determine whether batch>1 reuses the dequantized expert tile across token columns. If NO reuse ⇒ MTP also needs wiring `ggml_gemm_q2_K_8x8` into mul_mat_id **and** a new Q3_K repack/GEMM kernel (large). This gate can kill MTP regardless of acceptance.
- **INC-1 — load + GGUF verify (~0.5d, low risk):** `gguf-dump` the model, confirm `blk.78.nextn.*` (done from logs). Add a runtime `enable_mtp` flag clearing `TENSOR_SKIP` at `glm-dsa.cpp:80-85,139-148` for `i>=n_layer-nextn_predict_layers`. **Gate:** all nextn + layer-78 tensors non-null when on; base decode byte-identical with MTP off.
- **INC-2 — nextn graph + hidden handoff (~2d, the real driver / main risk):** factor the deepseek2 per-layer body (`deepseek2.cpp:199-417`) into a reusable helper; give `glm_dsa` its own graph (not `using graph = deepseek2::graph`, `models.h:1013`); **expose the PRE-output-norm hidden** (`inpL` at `deepseek2.cpp:425`, NOT `res->t_embd` which is post-norm at :430) as a graph output + public accessor; build `build_nextn(hidden, token_id, pos)`. **Gate:** offline numpy/torch reference over the 6 tensors + a captured hidden matches in-graph nextn top-1 token and full-logit cosine>0.999.
- **INC-3 — spec integration + verify loop (~1–1.5d, easy half):** add `COMMON_SPECULATIVE_TYPE_NEXTN` modeled on `draft_simple` (no 2nd model/vocab check); `process()` captures target hidden at the **last accepted** position, `draft()` iterates nextn K×, `accept()` rolls the private layer-78 KV by `n_accepted`. Wire a **STUB draft (echo id_last)** first to exercise batch-(K+1) plumbing + KV rollback without the graph. **Gate (lossless):** greedy spec-ON output token-for-token identical to spec-OFF.
- **INC-4 — measure + tune (~0.5d):** A/B vs 40 ms/token; log `n_accepted/n_generated` + expert-union bytes/token at batch=1 vs verify-batch (confirm INC-0 in-graph). Default K=2 (or K=1 given §2a); ship enabled-by-default only on a net win; else off-by-default flag. Add low-acceptance auto-reset (like `ngram_mod`, `speculative.cpp:600-621`).

## 5. Hardest unknowns / risks
1. **Expert-routing overlap** — measured LOW (§2a); the decisive perf risk, now mostly answered (negative-ish).
2. **§2b amortization in current code** — must resolve (INC-0.5); may force a GEMM-tile + Q3_K-repack rewrite.
3. **MTP acceptance at Q2_K target** — nextn trained in BF16; agreement at Q2_K unmeasured; breakeven ~0.45.
4. **Pre-norm hidden plumbing** — threading a new graph output + accessor through llama-context/graph is the biggest core-touching change.
5. **Layer-78 private KV rollback on partial-accept** — `n_layer_kv_from_start` (`glm-dsa.cpp:40`) caps KV to 78 layers; enabling layer-78 KV + capturing hidden at the *last accepted* (not drafted) position is subtle.
6. **Generic-vs-repack path** — the win assumes staying on the generic CPU path; if experts land in the repack extra-buffer, gemv-per-token re-streams the weight and the win disappears; Q3_K has no repack kernel.

## 6. Files to change (workflow inventory)
`src/models/glm-dsa.cpp:79-148` (drop TENSOR_SKIP under flag) · `src/models/models.h:1008-1014` (own graph subclass) · `src/models/deepseek2.cpp:199-417` (factor reusable layer body) · `src/llama-graph.{cpp,h}` (expose pre-norm hidden output + new inputs) · `common/common.h:158-167` (`COMMON_SPECULATIVE_TYPE_NEXTN`) · `common/speculative.cpp:~25,343-365,822` (impl + register) · `src/llama-model.cpp:1794` (`llama_model_has_mtp` accessor) · `tools/server/server.cpp` + `common/sampling` (form verify ubatch, greedy accept) · plus the §2b GEMM wiring if INC-0.5 says so.

---

## 7. Recommendation (honest)
1. **Do PLAN A now** (certain +10–15%, low risk).
2. **Before any MTP code:** resolve INC-0.5 (§2b code read) and re-run INC-0 on the *target* workload.
3. **MTP go/no-go:** build only if (INC-0.5 shows in-code amortization OR you accept building the GEMM/Q3_K-repack) AND expected acceptance is high. On the current data MTP is a **+5–30% (K=1), acceptance-dependent, ~4-day, core-touching** bet — lossless, but no longer the headline lever. PLAN A delivers comparable gain far cheaper.
4. If MTP is pursued anyway, follow INC-1 → INC-2 → INC-3 → INC-4 with the **lossless greedy-equality** gate as non-negotiable.
