# DECISION GATE: α → route (data-driven, no hype)

> Filled in once INC-2 reports a TRUSTED α (Gate 2 offline cross-check closed). If Gate 2 is open, α is
> "suspect" and we re-close Gate 2 before deciding — a wrong α misdirects a week of GEMM work.

## The throughput model (anchored to the CONFIRMED 27.58 t/s baseline, not the old 25)

Per-token decode at 27.58 t/s = **36.3 ms/token**, decomposed (A1+A3 timing, BW-bound cmp_cpu):
- `cmp_cpu` (offloaded expert RAM read+dequant, BW-bound 107/152 GB/s) = **16.3 ms** (45%)
- fixed forward overhead (in_gpu attention+resident ~8 + in_cpu prestage ~4.3 + sched/overhead ~7.7) = **20.0 ms** (55%)

A K=1 nextn-MTP verify runs ONE forward over (1 sampled + 1 draft) = 2 token-columns, and on accept yields
(1 + 1) = 2 tokens; on reject yields 1 token. **Expected tokens per forward = 1 + α.** What each term does
in that forward:
- **fixed overhead (20 ms): paid ONCE per forward** → amortized across (1+α) tokens. THIS is the MTP win.
- **cmp_cpu (16.3 ms): NOT amortized without GEMM** (verified §2b: mul_mat_id re-dequants per token-column).
  The 2-column verify reads experts for BOTH columns. With routing union_mult ≈ 1.77 (INC-0 measured, the 2
  columns route to ~1.77× distinct experts, not 2×), the verify cmp_cpu ≈ 16.3 × 1.77 = **28.9 ms**.

### Per-accepted-token time (GEMM-free, the minimal MTP)
`t_forward = 20.0 (fixed) + 28.9 (cmp_cpu union) = 48.9 ms`, producing `(1+α)` tokens:
`ms/token = 48.9 / (1+α)` →  **t/s = 1000 × (1+α) / 48.9**

| α | ms/tok | t/s | vs 27.58 |
|---|---|---|---|
| 0.30 | 37.6 | 26.6 | **−3.5%** (REGRESSION) |
| 0.45 | 33.7 | 29.7 | +7.7% |
| 0.50 | 32.6 | 30.7 | +11.3% |
| 0.60 | 30.6 | 32.7 | +18.6% |
| 0.70 | 28.8 | 34.8 | +26.1% |
| 0.85 | 26.4 | 37.9 | +37.4% |

> NOTE this is slightly MORE pessimistic than PLAN_C's table because it charges the FULL union cmp_cpu
> (28.9 ms) on every forward — i.e. it does NOT assume the fixed overhead is the only cost. Breakeven is
> **α ≈ 0.36** (where (1+α)/48.9 = 1/36.3). Below that, the union cmp_cpu tax exceeds the overhead saving.
> The exact breakeven depends on union_mult; if real union is lower than 1.77 on this workload, breakeven
> drops and all rows improve. Re-derive with the MEASURED union if the probe also logs draft/target expert sets.

### If GEMM (item 1, Q2_K wired into mul_mat_id) lands later
GEMM collapses the 2-column cmp_cpu from 1.77× back toward ~1.0× (dequant the expert tile once, apply to
both columns) — but ONLY for Q2_K experts (gate/up = 2/3 of expert bytes); Q3_K (down = 1/3) stays per-column
until item 2. So GEMM-partial cmp_cpu ≈ 16.3 × (2/3 × 1.0 + 1/3 × 1.77) = 16.3 × 1.257 = 20.5 ms:
`t_forward = 20.0 + 20.5 = 40.5 ms` → `t/s = 1000×(1+α)/40.5`:

| α | GEMM-partial t/s | vs 27.58 |
|---|---|---|
| 0.50 | 37.0 | +34% |
| 0.70 | 42.0 | +52% |

Item 2 (Q3_K GEMM, week-scale) pushes cmp_cpu→16.3 flat: `t_forward=36.3`, t/s=1000×(1+α)/36.3 → α0.7=46.8 (+70%).
**BUT** item 1/2 value is entirely conditional on α clearing breakeven first, and item 2's marginal gain over
item 1 (42→47 @α0.7) costs a week for ~+12% — gated hard by whether item 1's measured gain justifies it.

## ROUTE DECISION (apply once α is trusted)

- **α < 0.36 → MTP is DEAD.** Ship 27.58. Write the wall equation (experts 224GB>VRAM192 → 1.75GB/tok RAM read
  → cmp_cpu 16.3ms floor + 20ms fixed = 36.3ms = 27.58 t/s; MTP can't beat it because union cmp_cpu tax >
  overlap saving at this α). Do NOT build INC-3. This is a clean, honest kill — a measured negative is progress.
- **0.36 ≤ α < 0.55 → MARGINAL.** INC-3 (real spec loop, GEMM-free) for a +5–11% lossless bank IF the
  engineering is cheap (it mostly is — the spec framework + KV rollback already exist). GEMM NOT justified
  (partial gain too small vs the union tax it removes at low α). Bank the small win, stop.
- **0.55 ≤ α < 0.75 → SOLID.** Build INC-3 (+15–26%). THEN evaluate item 1 (Q2_K GEMM wiring, ~1–2 days,
  kernel EXISTS at repack.cpp:1986, just unwired into mul_mat_id at 4240-4244). Item 1 gate: integer-exact vs
  generic + measured cmp_cpu/token drop at the verify batch. Item 2 (Q3_K) ONLY if item 1's measured gain
  extrapolates to justify a week.
- **α ≥ 0.75 → STRONG.** INC-3 + item 1 GEMM both clearly worth it (+34–52%). Consider K=2 ONLY if union at
  K=2 (measured 2.45×) doesn't eat it — likely K=1 stays best on this routing, but measure.

## Non-negotiables regardless of route
- INC-3 lossless gate: greedy spec-ON output == spec-OFF token-for-token identical. Non-negotiable.
- Any GEMM change: integer-exact vs generic on 20k blocks (cosine FORBIDDEN — the fed1a84 lesson), then real-model
  coherence + per-token cmp_cpu drop.
- Re-measure union_mult on the TARGET workload if the probe logs expert sets — code/structured output predicted
  LOWER overlap than prose, which would WORSEN the union tax and RAISE breakeven. Don't assume prose's 1.77.
- Report the honest t/s with config/seed/warm state. No A/B across different conditions.
