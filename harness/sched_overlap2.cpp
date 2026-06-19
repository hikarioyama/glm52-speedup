// §9.1b de-risk: the SHARED-INPUT handoff. The CPU branch needs `inp` (produced on
// GPU). cpy_tensor_async(GPU->CPU) returns false (dst not cuda) => the sched falls to
// ggml_backend_synchronize(GPU), which (if inp is copied AFTER the GPU expert compute
// is queued) serializes CPU behind GPU. Fix: pre-stage inp on CPU as its own split
// BEFORE the GPU compute split (cont(inp) pinned CPU, expanded early).
//
// This spike builds:
//   x0 (gpu leaf) -> shared = Wsh @ x0   (on GPU; this is "inp")
//   GPU chain: h=shared;       for d<depth_g: h = Wg[d] @ h      (GPU resident)
//   CPU chain: h=shared_or_cpu; for d<depth_c: h = Wc[d] @ h     (CPU, pinned)
//   out = add(partial_gpu, partial_cpu)   (partial_gpu = src0 => GPU split first)
//
// mode: 0 = no pre-stage (CPU chain reads `shared` directly -> expect SERIAL)
//       1 = pre-stage   (CPU chain reads cont(shared) pinned CPU, expanded early -> expect OVERLAP)
//
// build: see sched_overlap.cpp
// run:   sched_overlap2 [N] [depth_g] [depth_c] [iters] [threads] [mode]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>

using clk = std::chrono::high_resolution_clock;

int main(int argc, char ** argv) {
    const int64_t N       = argc > 1 ? atoll(argv[1]) : 8192;
    const int     depth_g = argc > 2 ? atoi(argv[2]) : 40;
    const int     depth_c = argc > 3 ? atoi(argv[3]) : 2;
    const int     iters   = argc > 4 ? atoi(argv[4]) : 30;
    const int     threads = argc > 5 ? atoi(argv[5]) : 24;
    const bool    parallel = argc > 6 ? atoi(argv[6]) != 0 : false;

    ggml_backend_t be_gpu = ggml_backend_cuda_init(0);
    if (!be_gpu) { fprintf(stderr, "cuda init failed\n"); return 1; }
    ggml_backend_t be_cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be_cpu, threads);

    ggml_backend_buffer_type_t bt_gpu = ggml_backend_get_default_buffer_type(be_gpu);
    ggml_backend_buffer_type_t bt_cpu = ggml_backend_get_default_buffer_type(be_cpu);

    // run one config: use_gpu chain, use_cpu chain, prestage (only relevant when both)
    auto run = [&](bool use_gpu, bool use_cpu, int prestage, const char * tag) {
        int n_w = depth_g + depth_c + 4;
        ggml_init_params wp{ ggml_tensor_overhead() * (n_w + 8), nullptr, true };
        ggml_context * ctx_wg = ggml_init(wp);
        ggml_context * ctx_wc = ggml_init(wp);
        ggml_init_params cp{ ggml_graph_overhead() + ggml_tensor_overhead() * (n_w * 3 + 32), nullptr, true };
        ggml_context * ctx_c = ggml_init(cp);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx_c, (depth_g + depth_c) * 2 + 64, false);

        std::vector<ggml_backend_buffer_t> bufs;

        // GPU weights: x0, Wsh, Wg[depth_g]
        ggml_tensor * x0  = ggml_new_tensor_2d(ctx_wg, GGML_TYPE_F32, N, 1);
        ggml_tensor * Wsh = ggml_new_tensor_2d(ctx_wg, GGML_TYPE_F32, N, N);
        std::vector<ggml_tensor*> Wg(depth_g);
        for (int d = 0; d < depth_g; d++) Wg[d] = ggml_new_tensor_2d(ctx_wg, GGML_TYPE_F32, N, N);
        ggml_backend_buffer_t buf_g = ggml_backend_alloc_ctx_tensors_from_buft(ctx_wg, bt_gpu);
        ggml_backend_buffer_set_usage(buf_g, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_clear(buf_g, 0);
        bufs.push_back(buf_g);

        // CPU weights: Wc[depth_c]
        std::vector<ggml_tensor*> Wc(depth_c);
        for (int d = 0; d < depth_c; d++) Wc[d] = ggml_new_tensor_2d(ctx_wc, GGML_TYPE_F32, N, N);
        ggml_backend_buffer_t buf_c = ggml_backend_alloc_ctx_tensors_from_buft(ctx_wc, bt_cpu);
        ggml_backend_buffer_set_usage(buf_c, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_clear(buf_c, 0);
        bufs.push_back(buf_c);

        ggml_backend_t backends[2] = { be_gpu, be_cpu };
        ggml_backend_buffer_type_t bufts[2] = { bt_gpu, bt_cpu };
        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 2,
                                        (depth_g + depth_c) * 2 + 128, parallel, true);

        // shared = Wsh @ x0  -> "inp", produced on GPU
        ggml_tensor * shared = ggml_mul_mat(ctx_c, Wsh, x0); // [N,1]

        // pre-stage shared onto CPU as its own EARLY split (the fix)
        ggml_tensor * shared_cpu = shared;
        if (use_cpu && prestage) {
            shared_cpu = ggml_cont(ctx_c, shared);
            ggml_backend_sched_set_tensor_backend(sched, shared_cpu, be_cpu);
            ggml_build_forward_expand(gf, shared_cpu); // force this split FIRST
        }

        ggml_tensor * partial_gpu = nullptr, * partial_cpu = nullptr;
        // GPU chain (built first => its split before CPU compute split)
        if (use_gpu) {
            ggml_tensor * h = shared;
            for (int d = 0; d < depth_g; d++) h = ggml_mul_mat(ctx_c, Wg[d], h);
            partial_gpu = h;
            ggml_build_forward_expand(gf, partial_gpu);
        }
        // CPU chain
        if (use_cpu) {
            ggml_tensor * h = shared_cpu;
            for (int d = 0; d < depth_c; d++) {
                h = ggml_mul_mat(ctx_c, Wc[d], h);
                ggml_backend_sched_set_tensor_backend(sched, h, be_cpu);
            }
            partial_cpu = h;
            ggml_build_forward_expand(gf, partial_cpu);
        }

        // prestage==2: skip the join (both partials are independent outputs) to isolate
        // whether the final add is what serializes.
        if (partial_gpu && partial_cpu && prestage != 2) {
            ggml_tensor * out = ggml_add(ctx_c, partial_gpu, partial_cpu);
            // prestage==3: pin the JOIN to GPU so it becomes its own split AFTER the CPU
            // chain, instead of being grouped into the CPU split (which forces a GPU sync
            // at the start of that split = serialization).
            if (prestage == 3) ggml_backend_sched_set_tensor_backend(sched, out, be_gpu);
            ggml_build_forward_expand(gf, out);
        } else {
            if (partial_gpu) ggml_build_forward_expand(gf, partial_gpu);
            if (partial_cpu) ggml_build_forward_expand(gf, partial_cpu);
        }

        if (!ggml_backend_sched_alloc_graph(sched, gf)) { fprintf(stderr, "alloc fail\n"); exit(1); }
        for (int i = 0; i < 3; i++) ggml_backend_sched_graph_compute(sched, gf);
        ggml_backend_sched_synchronize(sched);

        auto t0 = clk::now();
        for (int i = 0; i < iters; i++) ggml_backend_sched_graph_compute(sched, gf);
        ggml_backend_sched_synchronize(sched);
        auto t1 = clk::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        int n_splits = ggml_backend_sched_get_n_splits(sched);
        fprintf(stdout, "%-22s : %8.3f ms/iter  (splits=%d)\n", tag, ms, n_splits);
        fflush(stdout);

        ggml_backend_sched_free(sched);
        for (auto b : bufs) ggml_backend_buffer_free(b);
        ggml_free(ctx_c); ggml_free(ctx_wg); ggml_free(ctx_wc);
        return ms;
    };

    fprintf(stderr, "N=%lld depth_g=%d depth_c=%d iters=%d threads=%d\n",
            (long long)N, depth_g, depth_c, iters, threads);
    double gpu_only = run(true,  false, 0, "gpu_only");
    double cpu_only = run(false, true,  0, "cpu_only");
    double both_no  = run(true,  true,  0, "both_NO_prestage");
    double both_ps  = run(true,  true,  1, "both_PRESTAGE");
    double both_nj  = run(true,  true,  2, "both_PRESTAGE_NOJOIN");
    double both_jg  = run(true,  true,  3, "both_PRESTAGE_JOIN_GPUPIN");

    double serial = gpu_only + cpu_only;
    double ideal  = gpu_only > cpu_only ? gpu_only : cpu_only;
    auto ov = [&](double both){ return 100.0 * (serial - both) / (serial - ideal); };
    fprintf(stdout, "\n--- analysis ---\n");
    fprintf(stdout, "serial(sum)        = %8.3f ms\n", serial);
    fprintf(stdout, "ideal(max)         = %8.3f ms\n", ideal);
    fprintf(stdout, "no-prestage     overlap = %6.1f%%  (%.3f ms)\n", ov(both_no), both_no);
    fprintf(stdout, "prestage        overlap = %6.1f%%  (%.3f ms)\n", ov(both_ps), both_ps);
    fprintf(stdout, "prestage_nojoin overlap = %6.1f%%  (%.3f ms)\n", ov(both_nj), both_nj);
    fprintf(stdout, "prestage_join_gpupin    = %6.1f%%  (%.3f ms)  <== the real-MoE shape\n", ov(both_jg), both_jg);

    ggml_backend_free(be_gpu);
    ggml_backend_free(be_cpu);
    return 0;
}
