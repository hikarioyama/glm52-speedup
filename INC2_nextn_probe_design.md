# INC-2 design: GLM-5.2 nextn graph + single-prefill α-probe

> Goal: build the GLM-5.2 nextn (MTP) head into the graph just far enough to measure **acceptance α**
> (how often the nextn head's greedy draft == the target's real next token) FAITHFULLY, with the
> minimum graph surgery. This is the decisive GO/NO-GO gate for the whole MTP investment — α governs
> everything (see PLAN_C §2c: K=1 gives +5% @α0.5 / +19% @α0.7 / +30% @α0.85, all GEMM-free).
> Source tree: `~/src/llama.cpp` (branch glm52-cpu-gpu-moe-split). Arch = `glm_dsa`. n_embd=6144,
> n_layer=79, nextn_predict_layers=1 ⇒ layer 78 is the single nextn block. INC-1 (the `--enable-mtp`
> load flag) is already landed and verified; the layer-78 attn+MoE+nextn tensors load non-null when on.

## The key insight: measure α with ONE prefill forward, no autoregressive lockstep

A naive α probe runs the target + nextn head autoregressively in lockstep, carrying the pre-norm
hidden across forwards and juggling a private layer-78 KV with rollback. **We avoid all of that.**

In a **prefill** the entire token sequence S = [t0..t_{N-1}] is the input ubatch, so within ONE graph:
1. run the main layers 0..77 → produces the pre-output-norm hidden `h_i` at every position i (in-graph, no handoff);
2. in the SAME graph run the nextn block (layer 78) over all positions, where the nextn input at
   position i uses `h_i` (in-graph) and `emb(t_{i+1})` (a shift-by-one view of the input embeddings,
   available because the whole sequence is the input);
3. output the nextn draft logits at each position.

The nextn block's MLA attention runs causally over the prefill positions — and prefill-with-causal-mask
is numerically identical to autoregressive decode for a causal block, so this α is faithful. The
layer-78 KV is allocated/written ONCE inside this single ubatch (same find_slot as layers 0..77) — no
cross-forward conflict, no rollback. **α from this probe ≈ real spec-loop α** (modulo tiny prefill-vs-
decode numerical batch effects; note the caveat, don't pretend it's exact).

### α definition (over S = the target's OWN greedy generation, so it is on-distribution)
- For greedy S, the target's real token after position i+1 IS `S[i+2]` (greedy ⇒ argmax(main logits at i+1)==S[i+2]).
- nextn draft at logical position i = `argmax(nextn_logits_i)` computed from `h_i` and `emb(S[i+1])`.
- **α = mean over i∈[0, N-3] of 1[ argmax(nextn_logits_i) == S[i+2] ]**.
- Also report per-position and the top-1 match plus a histogram of draft rank of the true token (diagnostic).

## nextn math (DeepSeek-V3 MTP, confirm against shapes)
At position i, to draft the token that follows `t_{i+1}`:
1. `e  = enorm(emb(t_{i+1}))`            enorm = RMSNorm, weight `model.layers[78].nextn.enorm` {n_embd}
2. `hn = hnorm(h_i)`                      hnorm = RMSNorm, weight `model.layers[78].nextn.hnorm` {n_embd}
3. `x  = eh_proj · concat(e, hn)`         eh_proj `model.layers[78].nextn.eh_proj` {2*n_embd, n_embd}; concat along dim0 → {2*n_embd}; result {n_embd}
   - **CONFIRM concat ORDER** (e first then hn, i.e. [token-embed ; hidden]) against the reference; getting it backwards silently tanks α. PLAN_C §3 step 3 says concat(e, hn).
4. run ONE full layer-78 transformer block on `x` using layer-78's OWN weights (attn_norm, wq_a/wq_b,
   wkv_a_mqa, wk_b, wv_b, wo + the MLA path; ffn_norm; build_moe_ffn over ffn_gate/up/down_exps +
   ffn_exp_probs_b + shared expert) → `block_out`. This is EXACTLY the deepseek2 per-layer body with il=78.
5. `o = shared_head_norm(block_out)`      RMSNorm, weight `model.layers[78].nextn.shared_head_norm` {n_embd}
   - NOTE: nextn uses its OWN `shared_head_norm`, NOT the model's `output_norm`.
6. `draft_logits = output · o`            head tied to `model.output` (nextn.shared_head_head ABSENT/tied; fall back to model.output). embed in step 1 ties to `model.tok_embd` (nextn.embed_tokens ABSENT).

## Exact code changes

### A. hparams: give layer 78 a KV slot when MTP on (`src/models/glm-dsa.cpp:39-40`)
Today `hparams.n_layer_kv_from_start = hparams.n_layer - hparams.nextn_predict_layers` (=78), so
`has_kv(78)==false` and `map_layer_ids.at(78)` would throw in build_attn. Change to:
```cpp
hparams.n_layer_kv_from_start = params.enable_mtp ? hparams.n_layer
                                                  : hparams.n_layer - hparams.nextn_predict_layers;
```
With enable_mtp ⇒ n_layer_kv_from_start=79 ⇒ has_kv(78)=true ⇒ the KV ctor allocates layer-78's latent-K
(MLA ⇒ has_v=false) and registers `map_layer_ids[78]`. **MTP-off path unchanged (byte-identical gate holds).**
(`params` is a member of llama_model — already used at glm-dsa.cpp:81.)

### B. glm_dsa gets its OWN graph subclass (`src/models/models.h:1008-1016`)
Replace `using graph = llama_model_deepseek2::graph;` (models.h:1013) with an own nested struct mirroring
deepseek2's pattern (models.h:989-991):
```cpp
struct graph : public llm_graph_context { graph(const llama_model & model, const llm_graph_params & params); };
```
Why: editing deepseek2::graph in place would perturb deepseek2 / deepseek2ocr / mistral4 (which alias it)
and risk their lossless gates. Add the ctor body in `src/models/glm-dsa.cpp`.

### C. glm_dsa::graph ctor = copy deepseek2 body, factor the per-layer body, capture h, append nextn
Copy the deepseek2::graph ctor body (`src/models/deepseek2.cpp:~150-439`) into `glm-dsa.cpp`. Then:

1. **Factor the per-layer body** (`deepseek2.cpp:196-423`, from `inpSA=inpL` through `inpL=cur`) into a
   lambda `auto run_layer = [&](ggml_tensor * inpL_in, int il, bool apply_out_ids) -> ggml_tensor*`,
   capturing the locals it needs (kq_scale, inp_pos, inp_attn_k/kv, inp_attn_scale, is_mla, is_ocr,
   n_head, the n_embd_head_* dims, n_expert*, freq_base/scale, rope params). Return the post-build_cvec `cur`.
   - The `inp_out_ids` tail-reduction (`deepseek2.cpp:369-372`) is gated by `apply_out_ids` (true ONLY for the
     last MAIN layer il==77; FALSE for the nextn call so it operates on all positions).
   - For glm_dsa: `is_ocr=false`, `is_mla=true`, `is_lite=false` (it loads wq_a/wq_b, not wq). The DSA
     indexer tensors are loaded but UNUSED by this graph (plain MLA) — keep it that way; do NOT add indexer.
2. **Main loop**: `for il in 0..77: inpL = run_layer(inpL, il, il==76? ...)`. Use effective_n_layers
   = n_layer - nextn_predict_layers (=78), so il∈0..77; apply_out_ids on the last (il==77).
   (Keep the exact existing semantics; this must stay byte-identical to deepseek2 when the nextn branch is OFF.)
3. **Capture h**: after the loop, `ggml_tensor * h_prenorm = inpL;` (== deepseek2.cpp:425 `cur=inpL`, the
   PRE-output-norm hidden). **DO NOT use res->t_embd (post-norm).** Keep building the normal main path
   (output_norm → t_embd, lm_head → t_logits) UNCHANGED so the base graph is identical when probe is off.
4. **nextn probe branch** (env-gated to keep it a pure measurement add, matching the repo idiom of
   `LLAMA_MOE_CPU_SPLIT` / `GGML_CPU_HUGEPAGE`): `if (getenv("LLAMA_MTP_PROBE") && model.layers[78].nextn.eh_proj)`:
   - **★P0 BLOCKER FIX — everything MUST be N columns, never N-1.** (review-confirmed: an N-1 draft causes a
     host-readback OOB — for an all-logits prefill `n_outputs == n_tokens == N` so the readback copies
     `N*n_vocab` floats from t_logits and `output_reserve` only sizes `n_vocab*N`; AND a row-misaligned
     attn mask / `ggml_view_3d(n_tokens=N)` mismatch inside run_layer. So build the shift as N columns and
     let the harness ignore the last 2.)
   - `tok_emb_all` = the input token embeddings {n_embd, N} (the tensor build_inp_embd returns at graph top;
     it is `build_inp_embd(model.tok_embd)`'s result — usable for views; n_embd_inp==n_embd for glm_dsa).
     Build `emb(t_{i+1})` shifted LEFT by one column, then PAD the last column to keep width N:
     ```cpp
     const int64_t s = tok_emb_all->nb[1];                                  // column stride
     ggml_tensor * drop0 = ggml_view_2d(ctx0, tok_emb_all, n_embd, N-1, s, s);   // cols 1..N-1
     ggml_tensor * lastc = ggml_view_2d(ctx0, tok_emb_all, n_embd, 1,   s, (N-1)*s); // col N-1 (junk pad)
     ggml_tensor * emb_shift = ggml_concat(ctx0, drop0, lastc, 1);          // {n_embd, N}; last col is junk
     ```
   - `h = h_prenorm;`  // {n_embd, N}, used as-is (no view) so ne[1]==N matches emb_shift
   - `e  = build_norm(emb_shift, model.layers[78].nextn.enorm, NULL, LLM_NORM_RMS, -1);`  // enorm→token-embed
   - `hn = build_norm(h,         model.layers[78].nextn.hnorm, NULL, LLM_NORM_RMS, -1);`  // hnorm→hidden
   - `x  = ggml_mul_mat(ctx0, model.layers[78].nextn.eh_proj, ggml_concat(ctx0, e, hn, 0));` // concat order e THEN hn; {n_embd, N}
   - `ggml_tensor * blk = run_layer(x, 78, /*apply_out_ids=*/false);`  // layer-78 MLA(+KV il=78)+MoE, x is N cols
   - `ggml_tensor * o = build_norm(blk, model.layers[78].nextn.shared_head_norm, NULL, LLM_NORM_RMS, -1);`  // nextn norm, NOT output_norm
   - `ggml_tensor * draft = ggml_mul_mat(ctx0, model.output, o);`  // tied head → {n_vocab, N}
   - **Overwrite the output**: `res->t_logits = draft;` and `ggml_build_forward_expand(gf, draft);` so the
     existing logits readback returns DRAFT logits, shape {n_vocab, N} matching n_outputs=N (no OOB).
   - **Position mapping**: draft column i uses h_i + emb(t_{i+1}), predicting t_{i+2}. Harness compares
     `argmax(logits_ith(i)) == S[i+2]` for **i ∈ [0, N-3]** only; columns N-2 (no t_N) and N-1 (junk pad) ignored.
   - math (concat order, enorm/hnorm assignment, shared_head_norm-then-tied-output) is **review-validated
     against vLLM `deepseek_mtp.py` + `glm4_moe_mtp.py`** — but see gate 2: there is NO existing nextn forward
     graph anywhere in this repo, so the offline numpy cross-check is the ONLY correctness guarantee.

### D. α harness (`tools/mtp-alpha/` or a `harness/` script driving an example)
Smallest correct vehicle — a dedicated tool (clone llama-cli/embedding scaffolding):
1. Load model with `--enable-mtp` (LLAMA_MTP_PROBE unset).
2. Greedily generate S = N tokens (e.g. N=256) from a fixed prompt, temp 0 — the ON-DISTRIBUTION sequence.
3. Build ONE batch of all S tokens with `logits=true` at every position; `llama_decode`. **★P1 env-gate:
   run this in a DEDICATED process with `LLAMA_MTP_PROBE=1` set at startup** (graph reuse caches topology by
   input-can_reuse only, so toggling the env mid-process risks a stale non-probe graph — review-confirmed).
   So: process #1 (env unset) generates S and writes it to a file; process #2 (`LLAMA_MTP_PROBE=1`) loads,
   reads S, runs the all-logits prefill, computes α. The byte-identical base A/B (probe-off == HEAD) is also a
   separate env-unset run.
4. For i∈[0,N-3]: `draft_i = argmax(llama_get_logits_ith(ctx, i))`; `n_acc += (draft_i == S[i+2])`.
5. Print **α = n_acc / (N-2)**, plus a rank histogram (how often true token is in draft top-5) and a few
   example (draft, true) pairs for eyeballing.
Reuse the SAME server.sh env/config (GGML_CUDA_NO_PINNED=1, GGML_CPU_HUGEPAGE=1, -ctk/-ctv q8_0, the -ot
offload with blk.78 experts→CPU, -t 24). Run on GPU when free (1 load at a time).

## INC-2 gates (before trusting α)
1. **Build + load**: compiles; loads with --enable-mtp + LLAMA_MTP_PROBE; no crash; layer-78 KV allocated
   (no map_layer_ids throw); coherent (the probe path doesn't NaN).
2. **Offline reference cross-check** (the real correctness gate): dump `model.layers[78].nextn.{enorm,hnorm,
   eh_proj,shared_head_norm}` + capture ONE position's `h_i` and `emb(t_{i+1})` from the graph; recompute
   steps 1-3 + (a stripped attention-less or numpy-replicated) head in numpy/torch; require the in-graph
   nextn top-1 token == reference top-1 AND full-logit cosine > 0.999. (The MLA block is hard to replicate
   in numpy — at minimum verify the eh_proj input `x` matches the reference exactly, integer/float close,
   and that the head (shared_head_norm + output) on a captured block_out matches; this isolates the
   token-shift + concat-order + norm-choice bugs which are the silent α-killers.)
3. **Then** report α.

## Gotchas (from the subsystem maps — do NOT trip these)
- **PRE-norm vs POST-norm**: h_i is `inpL` (deepseek2.cpp:425) BEFORE output_norm. Using res->t_embd (post-norm)
  silently tanks α with no crash. (cosine-looks-fine-but-wrong trap.)
- **concat order** e then hn (token-embed first, hidden second) — verify vs reference.
- **shared_head_norm not output_norm** for the nextn head; head tied to model.output; embed tied to model.tok_embd
  (nextn.embed_tokens / nextn.shared_head_head are ABSENT → use model.tok_embd / model.output).
- **build_cvec(cur, il)** is applied inside run_layer at il=78 too; with no control vector loaded it's identity — fine for the probe (assert no cvec).
- **out_ids tail-reduction** must be OFF for the nextn call (apply_out_ids=false) — it operates on all positions.
- **MLA il=78 KV**: build_attn(inp_attn_k, ..., il=78) requires map_layer_ids[78] (hparams change A). MLA ⇒ latent-K only.
- **rope is ROPE_TYPE_NORM, n_pos_per_embd=1** (NOT mrope — review-corrected; GLM_DSA sets LLAMA_ROPE_TYPE_NORM at
  llama-model.cpp:2241). inp_pos is length N, the shift is simple and safe; no mrope/4D-pos concern. (Earlier
  "rope_sections=4 ⇒ mrope" worry was wrong: rope_sections is read but the rope type is NORM.)
- **attn-mask row alignment**: the self_kq_mask is shaped {n_kv, n_tokens=N, ...}; feeding run_layer(x,78) an
  N-column x keeps q rows == mask rows (the N-column FIX above is what guarantees this — an N-1 x would misalign
  and may trip a CUDA flash-attn assert under -fa 1 + q8_0 KV).
- **byte-identical base gate**: when LLAMA_MTP_PROBE is unset, the glm_dsa::graph must produce the IDENTICAL
  graph as today's deepseek2::graph (same nodes, same t_embd/t_logits). The own-subclass + factored lambda must
  be a behavior-preserving refactor. Verify by an A/B greedy generation (probe-off, MTP-on) == current HEAD output.
- **VRAM**: layer-78 KV + the probe's extra layer-78 forward add a little; the experts are CPU-placed in the probe -ot.

## Why this is enough (and what it is NOT)
This measures α faithfully WITHOUT the production spec loop (INC-3: the COMMON_SPECULATIVE_TYPE_NEXTN impl,
the cross-forward hidden handoff, the KV rollback). Those are only worth building if α clears ~0.45. The probe
reuses the existing logits readback (overwrite t_logits) so no new output-plumbing. GEMM (items 1/2) is
orthogonal and deferred — α decides whether any of it is worth it.
