# PLAN A — RAM-bandwidth / CPU-efficiency quick-wins for GLM-5.2 decode

> Status: **recommended next work.** Low-risk, mostly few-line changes, ~+10–15% decode
> expected (the first real t/s gain of the effort). Fully orthogonal to MTP and compounds
> with it. Every number below is **measured** or **code-referenced**; expected gains are
> estimates to be confirmed by measurement (do not report them as achieved until measured).
>
> Generated 2026-06-19 from the design+profile workflow. Repo: `~/src/llama.cpp`
> (branch `glm52-cpu-gpu-moe-split`, conda env `llamacpp-cu131`, build `cmake --build build -j --target llama-server`).

## 0. Why (the measured wall)

Decode at 19-offload, split off, batch=1 (real `[SYNC_TIME]` wall-clock, CUMUL/token):

| term | ms | note |
|---|---|---|
| **cmp_cpu** (CPU expert read+Q2_K/Q3_K dequant+dot) | **20.1** | the target of PLAN A |
| in_gpu (attn + resident experts + join wait) | 8.1 | |
| in_cpu (prestage GPU→CPU input copy, per-decode) | 4.3 | |
| overhead (sampling/sync/token-gen) | ~7 | |
| **total** | **~40 → 25.3 t/s** | |

`cmp_cpu` reads 19 layers × 8 experts × 11.53 MB = **1.75 GB/token in 20.1 ms = 87.6 GB/s = 58% of STREAM peak (152 GB/s, 24t)**.
It is **memory-bandwidth bound, NOT ALU-bound** — proven: an AVX-512 widening of the dequant kernel gave
**0% end-to-end** (cmp_cpu 20.1 AVX2 vs 21.5 AVX-512, commit `fed1a84`). So the lever is **lift sustained
bandwidth**, not faster math.

**Root cause of the 42% gap to peak (workflow analysis):**
1. **TLB / page-walk tax (dominant, addressable):** 1.75 GB/token over **4 KB pages** = ~428K translations/token; Zen5 STLB can't cover 1.75 GB → page-walks run concurrent with the streaming loads. Default `mmap=true` gives **file-backed 4 KB pages (THP-ineligible)**.
2. **Reduced memory-level parallelism:** dequant work (shift/mask/maddubs/FMA) between loads lowers in-flight cache-line requests vs a pure-copy STREAM kernel. The x86 q2_K/q3_K path has **NO software prefetch** (only arch/s390 + arch/powerpc do).
3. Access pattern is **benign** (experts read contiguously, 8 jumps/layer) — not the problem. NUMA is **moot** (single NPS1 node). Thread balance is fine at `-t 24`.

---

## A1 — Hugepages for the offloaded-expert buffers (HIGHEST confidence)

**Goal:** back the 1.75 GB/token expert reads with **2 MB pages** instead of 4 KB → ~512× fewer page-table
entries → collapse STLB misses/page-walks.

**State today:**
- `mmap=true` (default) → file-backed 4 KB pages, THP-ineligible.
- `--no-mmap` → anonymous `posix_memalign(64,…)` (`ggml/src/ggml.c` `ggml_aligned_malloc`, ~:367) → THP-eligible **but no `madvise(MADV_HUGEPAGE)` is issued**, and `/sys/kernel/mm/transparent_hugepage/enabled = madvise`, so the buffer stays 4 KB. (`HugePages_Total=0`, none reserved.)

**Step A1-validate (zero code, run FIRST):**
1. Launch the 26-t/s config **with `--no-mmap`** (already the default best config — see `harness/server.sh`).
2. In a separate root shell (user runs; do not paste sudo into Claude): `echo always > /sys/kernel/mm/transparent_hugepage/enabled`
3. Re-launch + measure `cmp_cpu` via `GGML_SCHED_SYNC_COUNT=3` `[SYNC_TIME]`.
4. **GATE:** if `cmp_cpu` drops ~20% (20→~16 ms) and `AnonHugePages` rises by ~the expert footprint → hugepages help → implement A1-impl. If no drop → TLB was not the bottleneck; skip A1, go to A2.
5. Restore `echo madvise > …/enabled` after.

**Step A1-impl (targeted, the proper fix):**
- Add `madvise(ptr, size, MADV_HUGEPAGE)` on the **CPU backend buffer allocation that holds the offloaded experts** (NOT every alloc). First confirm the exact alloc path the `=CPU` expert tensors take at runtime (likely the CPU backend buffer type → `ggml_aligned_malloc`; verify it isn't the mmap path). Gate the hint behind `--no-mmap` (anonymous only).
- Alternative (stronger, bigger): reserve `nr_hugepages` + `mmap(MAP_HUGETLB)` for the expert buffer. Needs up-front reservation, fails hard if unavailable. Only if THP-madvise underdelivers.

**Expected:** 58% → ~68–78% peak ⇒ cmp_cpu 20.1 → ~16 ms ⇒ **~25 → ~28 t/s** (estimate, measure it).
**Effort:** few lines (madvise) / medium (hugetlb). **Risk:** low (THP madvise) / medium (hugetlb reserve + longer load + no page-cache sharing under --no-mmap).

---

## A2 — Software prefetch in x86 Q2_K/Q3_K vec_dot

**Goal:** lift memory-level parallelism on the latency-bound streaming kernel so DRAM latency hides under the dequant ALU work.

- Add `__builtin_prefetch` of the **next super-block of `src0`** (next q2/q3 cache line(s)) a few iterations ahead, in `ggml/src/ggml-cpu/arch/x86/quants.c` `ggml_vec_dot_q2_K_q8_K` / `_q3_K` inner loops.
- **Pattern already exists** in `arch/powerpc/quants.c` and `arch/s390/quants.c` — mirror it.
- **Expected:** +5–12% sustained BW ⇒ cmp_cpu ~20.1 → ~18 ms ⇒ +1–1.5 t/s (HW prefetcher already does most of sequential; gains incremental).
- **Effort:** low (a handful of lines). **Risk:** low (worst case neutral; over-prefetch mildly pollutes L1). Easy A/B. Revalidate kernel correctness vs generic (integer-exact, NOT cosine — see `fed1a84` lesson).

---

## A3 — Thread pinning to physical cores (variance reduction)

- Pin the 24 compute threads 1:1 to physical cores **0–23**, explicitly NOT SMT siblings 24–47, via existing `--cpu-mask` / `--cpu-strict` (no code change).
- Prevents cross-CCX migration jitter (L3 = 32 MB per 6-core CCX × 4). `-t 24` already avoids the SMT collapse (48t → 10.5 t/s), so this is mostly **variance reduction ~+2–5%**.
- **Effort:** flag only. **Risk:** very low.

---

## Do NOT (measured dead-ends)
- ❌ **AVX-512 widening of the dequant** — already measured **0% e2e** (RAM-bound, not ALU-bound). The fix `fed1a84` is a correctness fix only.
- ❌ **Q3_K repack→GEMM** — no Q3_K repack kernel exists in-tree; at batch=1 repack gives ~0 (weight read once anyway).
- ❌ **`--numa`** — single NPS1 node; the flag sets `MADV_RANDOM` + prefetch=0 and would **HURT** this box.

---

## Verification protocol (every change)
1. **Correctness:** greedy (temp 0) output **byte-identical** before/after (bandwidth changes must not alter output). Kernel changes (A2): integer-exact vs `ggml_vec_dot_q*_K_q8_K_generic`, NOT cosine.
2. **cmp_cpu:** `GGML_SCHED_SYNC_COUNT=3`, read `[SYNC_TIME] … cmp_cpu=` CUMUL/token from the server log.
3. **t/s:** A/B vs the 25.3 t/s / 40 ms baseline at the **identical** 19-offload split-off config, `-t 24`, `--no-mmap`, temp 0. Long run (n≥400) for steady-state; short runs are noisy.
4. **Ordering:** A1-validate (gate) → A1-impl → A2 → A3. Stop early if a gate fails; report measured numbers honestly (盛らない).

Expected stacked PLAN-A result: **~28–30 t/s** (from ~25), low risk. Confirm by measurement.

---

## ➡️ NEXT: when PLAN A is complete, proceed to PLAN C
`~/projects/glm52-speedup/PLAN_C_bandwidth_then_mtp.md`

PLAN C is the full roadmap (PLAN A as Phase 0 → MTP self-speculation). **Read C's honest payoff
section first** — the MTP upside is much lower than first hoped (measured low expert-routing overlap +
the current code can't amortize shared experts without a GEMM rewrite). A faster `cmp_cpu` from PLAN A
directly shrinks MTP's dominant verify term, so A is a prerequisite either way.
