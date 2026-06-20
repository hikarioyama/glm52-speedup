# K=2 depth-2 self-spec chain — negative result + the per-column cost law

Follow-up to the K=1 in-graph MTP self-spec (30.07 t/s, +18.8%). Goal was 40+ t/s
by drafting deeper. **It regressed. The investigation found a clean cost law that
explains why, and relocates the real bottleneck.**

## Result

| config | verify cols (N) | nextn traversals | per-forward | tok/fwd | t/s |
|---|---|---|---|---|---|
| spec-OFF greedy | 1 | 0 | ~38 ms | 1.00 | 26.3 |
| **K=1 (shipped)** | 2 | 1 | 59 ms | 1.77 | **30.1** |
| K=2 depth-1 only (NO_STEP2) | 3 | 1 | 79 ms | 1.73 | 21.8 |
| K=2 depth-2 chain | 3 | 2 | 85 ms | 1.80 | 21.2 |

- α1 = 0.71 (preserved — the depth-2 step writes no layer-78 KV).
- α2 = 0.108 (the attention-less FFN-only step-2 is a weak predictor).

## The cost law (3 points, exact fit)

```
per-forward ≈ 18 + 20·N  ms        (N = number of verify columns)
t/s = tok_per_fwd / (18 + 20·N)
```

- **18 ms = amortizable fixed overhead** (multi-backend graph-split sync / launch).
  Speculation amortizes *this* — that's the whole K=1 win.
- **20 ms/column = per-token cost on the 22 CPU-offloaded MoE layers**, NOT amortizable.

Adding the depth-2 nextn traversal cost only +5.6 ms (79→85). Adding the 3rd verify
**column** cost +20 ms (59→79). So the bottleneck is the **column**, not the depth.

## Why depth-2 is dead

To beat K=1's 30.1 at N=3 you need tok/fwd > 2.38. Even a perfect α2=1.0 gives 2.44
(barely); a realistic α2≈0.6 gives 2.15 — a loss. Break-even needs **α2 ≥ 0.92**,
impossible for a depth-2 MTP draft. **N=2 (K=1) is the optimum for this offload config.**
Speculation tuning is exhausted here.

## What this relocates the fight to

The "extra verify column is free because decode is latency-bound" assumption is **false**:
each column adds a hard +20 ms. A perf profile of steady-state decode shows ~78%
libcuda cudaSync (spin), ~18% libc memcpy, ~0% libggml-cpu, GPU SM ~1% — i.e. **neither
the CPU nor the GPU is saturated**; the wall is the CPU↔GPU serialization across the
~63 graph splits/token. The open lever is whether that idle time can be **overlapped /
pipelined** (the CPU expert compute hidden under GPU work), which would shrink the 18 ms
fixed term and possibly the 20 ms/column. Under investigation.

Reproduce: `harness/inc3_selfspec.sh` (spec-OFF vs spec-ON + lossless gate),
`harness/perf_decode.sh` (long steady-state decode for profiling). The K=2 code lives on
the `mtp-k2-chain-experiment` branch of the llama.cpp-glm52 fork.
