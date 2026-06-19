// §9.1 de-risk: does ggml_backend_sched run an independent CPU split CONCURRENTLY
// with a CUDA split? (the whole CPU∥GPU expert-split thesis hinges on this)
//
// We build ONE cgraph containing two INDEPENDENT matmul chains:
//   - a GPU chain  (weights resident on CUDA0)   -> goes to the CUDA split
//   - a CPU chain  (weights resident on host)    -> goes to the CPU  split
// No data dependency between them. GPU chain nodes are expanded FIRST so the
// scheduler issues the (async) CUDA split before the (blocking) CPU split.
//
// If the scheduler overlaps them, wall(both) ~= max(gpu_only, cpu_only).
// If it serializes,            wall(both) ~= gpu_only + cpu_only.
//
// build:
//   g++ -O2 -std=c++17 -Iggml/include harness/sched_overlap.cpp -o harness/sched_overlap \
//       -Lbuild/bin -lggml -lggml-base -lggml-cpu -lggml-cuda -Wl,-rpath,$PWD/build/bin
// run:
//   harness/sched_overlap [depth] [Ng] [Nc] [iters] [threads]

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

struct chain_cfg {
    bool   use;
    int    depth;
    int64_t N;
    ggml_backend_t backend;
    ggml_backend_buffer_type_t buft;
};

// allocate `depth` weight matrices [N,N] + 1 input [N,1] on `buft`, build a matmul chain,
// expand it into `gf`. weights/inputs are zero-cleared (timing is data-independent).
static ggml_tensor * build_chain(ggml_context * ctx_w, ggml_context * ctx_c,
                                 ggml_cgraph * gf, const chain_cfg & c,
                                 std::vector<ggml_backend_buffer_t> & bufs) {
    // weights live in their own context so we can alloc them on a specific buft
    std::vector<ggml_tensor *> W(c.depth);
    for (int d = 0; d < c.depth; d++) {
        W[d] = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, c.N, c.N);
    }
    ggml_tensor * x = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, c.N, 1);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, c.buft);
    if (!buf) { fprintf(stderr, "alloc failed\n"); exit(1); }
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_clear(buf, 0);
    bufs.push_back(buf);

    // compute chain: h = x; h = W[d] @ h   (mul_mat(W[N,N], h[N,1]) -> [N,1])
    ggml_tensor * h = x;
    for (int d = 0; d < c.depth; d++) {
        h = ggml_mul_mat(ctx_c, W[d], h);
    }
    ggml_build_forward_expand(gf, h);
    return h;
}

int main(int argc, char ** argv) {
    const int     depth   = argc > 1 ? atoi(argv[1]) : 19;
    const int64_t Ng      = argc > 2 ? atoll(argv[2]) : 16384;
    const int64_t Nc      = argc > 3 ? atoll(argv[3]) : 4096;
    const int     iters   = argc > 4 ? atoi(argv[4]) : 30;
    const int     threads = argc > 5 ? atoi(argv[5]) : 24;

    fprintf(stderr, "depth=%d Ng=%lld Nc=%lld iters=%d threads=%d\n",
            depth, (long long)Ng, (long long)Nc, iters, threads);

    ggml_backend_t backend_gpu = ggml_backend_cuda_init(0);
    if (!backend_gpu) { fprintf(stderr, "cuda init failed\n"); return 1; }
    ggml_backend_t backend_cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend_cpu, threads);

    ggml_backend_buffer_type_t buft_gpu = ggml_backend_get_default_buffer_type(backend_gpu);
    ggml_backend_buffer_type_t buft_cpu = ggml_backend_get_default_buffer_type(backend_cpu);

    // run one configuration: which chains are active
    auto run = [&](bool use_gpu, bool use_cpu, const char * tag) {
        // contexts (no_alloc: tensors get backed by explicit buffers)
        size_t meta = ggml_tensor_overhead() * (2 * depth + 8) + ggml_graph_overhead();
        ggml_init_params wp{ meta, nullptr, true };
        ggml_context * ctx_wg = ggml_init(wp);
        ggml_context * ctx_wc = ggml_init(wp);
        ggml_init_params cp{ ggml_graph_overhead() + ggml_tensor_overhead() * (2 * depth + 8) * 2,
                             nullptr, true };
        ggml_context * ctx_c = ggml_init(cp);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx_c, 2 * depth + 16, false);

        std::vector<ggml_backend_buffer_t> bufs;

        // IMPORTANT: expand GPU chain FIRST so its split is issued before the CPU split
        if (use_gpu) {
            chain_cfg cg{ true, depth, Ng, backend_gpu, buft_gpu };
            build_chain(ctx_wg, ctx_c, gf, cg, bufs);
        }
        if (use_cpu) {
            chain_cfg cc{ true, depth, Nc, backend_cpu, buft_cpu };
            build_chain(ctx_wc, ctx_c, gf, cc, bufs);
        }

        ggml_backend_t backends[2] = { backend_gpu, backend_cpu };
        ggml_backend_buffer_type_t bufts[2] = { buft_gpu, buft_cpu };
        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 2,
                                        2 * depth + 32, /*parallel*/ false, /*op_offload*/ true);

        if (!ggml_backend_sched_alloc_graph(sched, gf)) { fprintf(stderr, "alloc graph fail\n"); exit(1); }

        // warmup
        for (int i = 0; i < 3; i++) ggml_backend_sched_graph_compute(sched, gf);
        ggml_backend_sched_synchronize(sched);

        auto t0 = clk::now();
        for (int i = 0; i < iters; i++) {
            ggml_backend_sched_graph_compute(sched, gf);
        }
        ggml_backend_sched_synchronize(sched);
        auto t1 = clk::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

        int n_splits = ggml_backend_sched_get_n_splits(sched);
        fprintf(stdout, "%-10s : %8.3f ms/iter   (splits=%d)\n", tag, ms, n_splits);
        fflush(stdout);

        ggml_backend_sched_free(sched);
        for (auto b : bufs) ggml_backend_buffer_free(b);
        ggml_free(ctx_c); ggml_free(ctx_wg); ggml_free(ctx_wc);
        return ms;
    };

    double gpu_only = run(true,  false, "gpu_only");
    double cpu_only = run(false, true,  "cpu_only");
    double both     = run(true,  true,  "both");

    double serial = gpu_only + cpu_only;
    double ideal  = gpu_only > cpu_only ? gpu_only : cpu_only;
    fprintf(stdout, "\n--- analysis ---\n");
    fprintf(stdout, "serial(sum)   = %8.3f ms\n", serial);
    fprintf(stdout, "ideal(max)    = %8.3f ms\n", ideal);
    fprintf(stdout, "measured both = %8.3f ms\n", both);
    double overlap_frac = (serial - both) / (serial - ideal); // 1.0 = perfect overlap, 0 = fully serial
    fprintf(stdout, "overlap_frac  = %6.1f%%  (100%%=perfect parallel, 0%%=fully serial)\n",
            100.0 * overlap_frac);

    ggml_backend_free(backend_gpu);
    ggml_backend_free(backend_cpu);
    return 0;
}
