// Ground truth: bypass ggml_backend_sched. Drive the two backends MANUALLY to learn
// whether GPU-async + CPU-blocking actually overlap at the backend level, given a
// pre-staged shared input. If THIS overlaps but the sched version doesn't, the sched
// is inserting a sync we must avoid; if THIS doesn't overlap either, it's foundational.
//
// region under test (per iter):
//   be_gpu.graph_compute_async(gpu_chain)   // queue on cuda stream, return
//   be_cpu.graph_compute(cpu_chain)         // blocks host thread
//   be_gpu.synchronize()                    // wait for gpu
// both chains read a pre-staged shared input (gpu copy + cpu copy made once up front).
//
// build: see sched_overlap.cpp ; run: manual_overlap [N] [depth_g] [depth_c] [iters] [threads]

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

// build a standalone chain graph on a single backend; weights+input in `buft`.
struct chain {
    ggml_context * ctx_w;
    ggml_context * ctx_c;
    ggml_cgraph  * gf;
    ggml_backend_buffer_t buf;
    ggml_tensor  * x; // input leaf (fill before compute)
    ggml_tensor  * out;
};

static chain make_chain(int depth, int64_t N, ggml_backend_buffer_type_t buft) {
    chain c;
    ggml_init_params wp{ ggml_tensor_overhead() * (depth + 4), nullptr, true };
    c.ctx_w = ggml_init(wp);
    std::vector<ggml_tensor*> W(depth);
    for (int d = 0; d < depth; d++) W[d] = ggml_new_tensor_2d(c.ctx_w, GGML_TYPE_F32, N, N);
    c.x = ggml_new_tensor_2d(c.ctx_w, GGML_TYPE_F32, N, 1);
    c.buf = ggml_backend_alloc_ctx_tensors_from_buft(c.ctx_w, buft);
    ggml_backend_buffer_clear(c.buf, 0);

    ggml_init_params cp{ ggml_graph_overhead() + ggml_tensor_overhead() * (depth + 8), nullptr, true };
    c.ctx_c = ggml_init(cp);
    c.gf = ggml_new_graph_custom(c.ctx_c, depth + 8, false);
    ggml_tensor * h = c.x;
    for (int d = 0; d < depth; d++) h = ggml_mul_mat(c.ctx_c, W[d], h);
    c.out = h;
    ggml_build_forward_expand(c.gf, h);
    // allocate compute buffers via a gallocr on this buft
    ggml_gallocr_t ga = ggml_gallocr_new(buft);
    ggml_gallocr_alloc_graph(ga, c.gf);
    // leak ga (process-lifetime spike)
    return c;
}

int main(int argc, char ** argv) {
    const int64_t N       = argc > 1 ? atoll(argv[1]) : 8192;
    const int     depth_g = argc > 2 ? atoi(argv[2]) : 4;
    const int     depth_c = argc > 3 ? atoi(argv[3]) : 18;
    const int     iters   = argc > 4 ? atoi(argv[4]) : 30;
    const int     threads = argc > 5 ? atoi(argv[5]) : 24;

    ggml_backend_t be_gpu = ggml_backend_cuda_init(0);
    ggml_backend_t be_cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be_cpu, threads);

    chain cg = make_chain(depth_g, N, ggml_backend_get_default_buffer_type(be_gpu));
    chain cc = make_chain(depth_c, N, ggml_backend_get_default_buffer_type(be_cpu));

    auto time_block = [&](const char * tag, auto fn) {
        for (int i = 0; i < 3; i++) fn();
        ggml_backend_synchronize(be_gpu); ggml_backend_synchronize(be_cpu);
        auto t0 = clk::now();
        for (int i = 0; i < iters; i++) fn();
        ggml_backend_synchronize(be_gpu); ggml_backend_synchronize(be_cpu);
        auto t1 = clk::now();
        double ms = std::chrono::duration<double,std::milli>(t1-t0).count()/iters;
        fprintf(stdout, "%-28s : %8.3f ms/iter\n", tag, ms); fflush(stdout);
        return ms;
    };

    double g = time_block("gpu_only", [&]{
        ggml_backend_graph_compute(be_gpu, cg.gf); // sync version
    });
    double c = time_block("cpu_only", [&]{
        ggml_backend_graph_compute(be_cpu, cc.gf);
    });
    double both = time_block("both_manual_overlap", [&]{
        ggml_backend_graph_compute_async(be_gpu, cg.gf); // queue async, return
        ggml_backend_graph_compute(be_cpu, cc.gf);       // blocks on host thread
        ggml_backend_synchronize(be_gpu);                // wait gpu
    });

    double serial = g + c, ideal = g > c ? g : c;
    fprintf(stdout, "\n--- analysis ---\n");
    fprintf(stdout, "serial(sum) = %.3f  ideal(max) = %.3f  both = %.3f\n", serial, ideal, both);
    fprintf(stdout, "overlap     = %.1f%%\n", 100.0*(serial-both)/(serial-ideal));

    ggml_backend_free(be_gpu);
    ggml_backend_free(be_cpu);
    return 0;
}
