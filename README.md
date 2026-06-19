# GLM-5.2 fast local inference — hunting for decode speed

**Goal:** run **GLM-5.2** (754B-parameter MoE, `unsloth/GLM-5.2-GGUF` **UD-Q2_K_XL**, ~236 GB)
as fast as possible for **single-stream decode** on a home-lab rig, *without* dropping the
quantization (smaller quants wreck quality — UD-Q2 is fixed). We are actively chasing every
hack that moves the needle and **would love ideas / PRs** (see [Looking for hacks](#looking-for-hacks)).

This repo is the **research / harness / write-up**. The engine changes live in a llama.cpp fork:
**https://github.com/hikarioyama/llama.cpp-glm52** (branch `glm52-cpu-gpu-moe-split`).

## The rig & the wall

| | |
|---|---|
| GPUs | 2× RTX PRO 6000 Blackwell — **192 GB** VRAM total, SM120, PCIe Gen5 x16 (~57 GB/s/link, ~115 GB/s both), P2P via `iommu=pt` |
| CPU | Threadripper **9965WX** — 24 physical cores, Zen5 (full AVX-512F/BW/VL/DQ/**VNNI**/BF16, **no AMX**) |
| RAM | 125 GB (STREAM triad ~152 GB/s @ 24 threads) |
| Engine | our own llama.cpp fork |

**The hard wall:** routed experts are **224 GB > 192 GB VRAM**. After non-routed weights (~15 GB),
KV, and scratch, only ~57 of the 76 MoE layers fit resident — so **~19+ MoE layers' experts must
live on the host and be consumed every single decode token**. Routing is uniform (no exploitable
hot/cold reuse), so ~1.7–2.0 GB of distinct expert weights are read **per token**.

## What we learned (measured, not guessed)

We built per-token instrumentation (`GGML_SCHED_SYNC_COUNT`) and measured the decode breakdown.
The headline result **overturned our own earlier theories**:

> **~74% of decode time is the CPU dequantizing + multiplying the offloaded `Q2_K`/`Q3_K`
> experts.** Attention + all resident-layer experts (GPU) are only ~19%. And the CPU part is
> **compute-bound** (only ~26% of STREAM bandwidth) — i.e. the dequant ALU is the limiter, not memory.

Per-token, ~40 ms at the ~25 t/s baseline:

| bucket | time | share |
|---|---|---|
| **CPU offloaded-expert dequant+matmul** | **~29.5 ms** | **~74%** |
| GPU work (attention + resident experts) | ~7.6 ms | ~19% |
| cross-backend / pre-stage | ~4.8 ms | ~7% |

## Approaches tried (and honest results)

| Idea | Result |
|---|---|
| **CPU∥GPU expert split** — split a layer's top-8 experts: *k* on CPU, *8−k* copied to GPU, computed concurrently | **Works. Best = k6 → 26.99 t/s (+14% over 23.74 baseline).** GPU branch is PCIe-copy-bound so only ~2 experts are worth offloading. |
| **ggml scheduler events @ n_copies=1** — kill the host-blocking WAR-guard `synchronize` that flushed the GPU pipeline every cross-backend handoff | Sync count dropped to ~0, but wall-clock unchanged → **sync was not the bottleneck.** |
| **dual-GPU split** — split the GPU-side experts across both PCIe links | First version serialized (GPU1 read GPU0-resident inputs → P2P queued on GPU0's stream); fixed by reading host-staged inputs. Still **doesn't beat single-GPU** — coordination/P2P overhead > the 2× copy bandwidth at the optimal (high-*k*) point. |
| **AVX-512 / VNNI `Q2_K`/`Q3_K` dequant kernel** — the stock kernels are **AVX2-only** (0 `zmm`, 0 `vpdpbusd`) despite the CPU having full AVX-512 | A widening port is only **~1.25×** — **Zen5 double-pumps the 512-bit integer datapath**, so `vpmaddubsw`/`vpdpbusd` have the *same* per-cycle throughput as 256-bit; the win is only from halving the count of loads/shifts/masks. (Our first port also had a real-model correctness bug — reverted; re-validation pending.) |
| `CPU_REPACK` (interleaved `q2_Kx8` gemv) via `-ot` | Crashes when forced via tensor-override, and is incompatible with the GPU split. |

## Where it stands

- **Verified best: +14% (24 → 27 t/s)**, output-correct.
- The bottleneck is now pinpointed: **CPU `Q2_K`/`Q3_K` dequant, compute-bound, `maddubs`-count-limited, on a double-pumped Zen5 integer datapath.**
- **Honest ceiling estimate:** ~32–37 t/s by stacking a *correct* AVX-512 kernel (~1.25×) + shrinking the offloaded-layer count (free VRAM → fewer host-read bytes) + the split. **43 t/s (the 1.8× dream) looks physically hard** given quant-is-fixed + experts-exceed-VRAM + the Zen5 integer ceiling — but we're not done.

Full diagnosis: [`CPU_GPU_SPLIT_SPEC.md`](CPU_GPU_SPLIT_SPEC.md). A 16-angle adversarial idea-hunt is in
[`SPEEDUP_HUNT_ROADMAP.json`](SPEEDUP_HUNT_ROADMAP.json).

## Looking for hacks

**This is an open hunt — got a trick? Open an issue / PR.** Especially interested in:

- A `Q2_K`/`Q3_K` CPU dequant-matmul that beats ~1.25× on **Zen5** (reducing `maddubs` *count*, not width — the per-16-element scale between `maddubs` and the horizontal add resists `vpdpbusd`). ik_llama-style restructuring? A one-time re-pack to a friendlier layout?
- Any legitimate way to overlap CPU expert compute across the **strict layer dependency** (L+1's attention needs L's output) — speculative/prefetch tricks that survive dynamic routing.
- Cheaper per-token host→GPU expert transfer (the copy is the GPU-branch limiter), or a smarter on-GPU expert cache despite uniform routing.
- Freeing VRAM to make more layers resident (e.g. the DSA indexer tensors look loaded-but-unread; KV quant).
- Anything we dismissed too quickly.

## Repo layout

| path | what |
|---|---|
| `CPU_GPU_SPLIT_SPEC.md` | the living spec — every fact is measured or code-referenced |
| `SPEEDUP_HUNT_ROADMAP.json` | multi-agent ranked roadmap of untried speedups |
| `harness/` | benchmark + measurement scripts (`server_sweep.sh` = one model load, runtime-tunable `k`/dual via `/tmp/moe_ctl`); `kernel_bench/` = standalone CPU dequant microbench |

Engine changes (the actual code): **https://github.com/hikarioyama/llama.cpp-glm52**.

## License

**Free and unencumbered — public domain** ([Unlicense](UNLICENSE)). Do whatever you want with it.
(The llama.cpp fork keeps upstream's MIT license for upstream code; the GLM-5.2-specific changes there are likewise dedicated to the public domain.)
