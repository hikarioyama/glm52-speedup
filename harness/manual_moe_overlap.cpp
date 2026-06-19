// De-risk: would a MANUAL CPU∥GPU orchestration (bypassing ggml sched, only 1 sync/layer)
// actually beat CPU-only for the GLM-5.2 decode MoE — including the per-token host->GPU
// expert COPY that the earlier compute-only spike (manual_overlap.cpp, 108%) skipped?
//
// Faithful model of one decode token over the offloaded MoE layers:
//   per layer (sequential): GPU branch = memcpyAsync(n_gpu experts H2D) + kernel(read them)
//                           CPU branch = read+dequant-like compute over n_cpu experts (24 thr)
//   orchestration: issue GPU async on streamG, run CPU (blocks host), ONE stream sync / layer.
//
// Real sizes: expert = 11.53 MB, 22 offloaded layers, top-8. k = n_cpu (split point).
//   CPU per-expert ~ host RAM read + dequant. GPU per-expert ~ 11.53MB / 57GB/s pinned + tiny.
//
// build: nvcc -O3 -Xcompiler -fopenmp -arch=sm_120 manual_moe_overlap.cpp -o manual_moe_overlap
// run:   manual_moe_overlap [n_layers] [k_cpu] [iters] [threads]

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <vector>
#include <omp.h>
#include <cuda_runtime.h>

using clk = std::chrono::high_resolution_clock;
#define CK(x) do{ cudaError_t e=(x); if(e){printf("cuda err %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

static const size_t EXPERT_BYTES = (size_t)(11.53 * 1024 * 1024); // GLM-5.2 expert (gate+up+down)
static const int    N_USED       = 8;

// CPU branch: read n experts from host + a dequant-like multiply-accumulate (compute-bound-ish),
// parallel over threads. Returns a checksum to prevent the compiler optimizing it away.
static int g_ops = 6; // compute ops per element (calibrate so CPU is dequant-compute-bound, not bw-bound)
static double cpu_experts(const float* host, int n_experts, size_t floats_per_expert) {
    // all threads split the TOTAL work (every expert's matmul is parallelized across cores,
    // as in ggml) => time scales with n_experts. The inner FMA loop emulates Q2_K dequant being
    // COMPUTE-bound (real decode hits only ~45GB/s = 26% of STREAM, per thread-sweep).
    const size_t total = (size_t)n_experts * floats_per_expert;
    double acc = 0.0;
    #pragma omp parallel for reduction(+:acc) schedule(static)
    for (size_t i = 0; i < total; i++) {
        float v = host[i];
        float s = v;
        for (int o = 0; o < g_ops; o++) s = s * 1.0009765625f + 0.5f; // dequant-like compute
        acc += s;
    }
    return acc;
}

// GPU kernel: read n experts from device scratch (mimic compute touching the copied weights).
__global__ void gpu_touch(const float* __restrict__ w, size_t total_floats, float* out) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    float s = 0.0f;
    for (; i < total_floats; i += (size_t)gridDim.x * blockDim.x) s += w[i] * 1.0009765625f;
    atomicAdd(out, s);
}

int main(int argc, char** argv) {
    const int n_layers = argc>1?atoi(argv[1]):22;
    const int k_cpu    = argc>2?atoi(argv[2]):4;     // experts computed on CPU
    const int iters    = argc>3?atoi(argv[3]):20;
    const int threads  = argc>4?atoi(argv[4]):24;
    g_ops              = argc>5?atoi(argv[5]):6; // CPU compute intensity (calibration)
    const int n_gpu    = N_USED - k_cpu;
    omp_set_num_threads(threads);

    const size_t fpe = EXPERT_BYTES / sizeof(float);
    // DISTINCT data for every (layer, expert) so the CPU reads cold RAM (not L3 cache),
    // matching the real model where 22 layers' experts total ~2GB and don't fit in cache.
    const size_t n_slots = (size_t)n_layers * N_USED;
    float* host; CK(cudaHostAlloc((void**)&host, n_slots*EXPERT_BYTES, cudaHostAllocDefault));
    #pragma omp parallel for
    for (size_t i=0;i<n_slots*fpe;i++) host[i] = (float)(i & 1023) * 0.001f;
    printf("host pinned = %.1f GB (distinct per layer)\n", n_slots*EXPERT_BYTES/1e9);
    float* gscratch; CK(cudaMalloc((void**)&gscratch, (size_t)N_USED*EXPERT_BYTES));
    float* gout; CK(cudaMalloc((void**)&gout, sizeof(float)));
    cudaStream_t sG; CK(cudaStreamCreate(&sG));

    auto layer_ptr = [&](int L){ return host + (size_t)L * N_USED * fpe; };
    auto gpu_branch = [&](int L, int n){ // copy layer L's n experts H2D + touch (async on sG)
        if (n<=0) return;
        CK(cudaMemcpyAsync(gscratch, layer_ptr(L), (size_t)n*EXPERT_BYTES, cudaMemcpyHostToDevice, sG));
        gpu_touch<<<512,256,0,sG>>>(gscratch, (size_t)n*fpe, gout);
    };

    auto time_it = [&](auto fn, const char* tag){
        for(int w=0;w<3;w++) fn();
        CK(cudaStreamSynchronize(sG));
        auto t0=clk::now();
        for(int it=0;it<iters;it++) fn();
        CK(cudaStreamSynchronize(sG));
        auto t1=clk::now();
        double ms=std::chrono::duration<double,std::milli>(t1-t0).count()/iters;
        printf("%-26s : %8.3f ms/token  (=%6.2f tok/s)\n", tag, ms, 1000.0/ms);
        return ms;
    };

    volatile double sink=0;
    auto cpu_slice = [&](int L, int n){ return cpu_experts(layer_ptr(L) + (size_t)n_gpu*fpe, n, fpe); }; // CPU reads experts [n_gpu..8)
    // baseline: all 8 experts on CPU, every layer (mirrors current 26 t/s offload)
    double cpu_only = time_it([&]{ for(int L=0;L<n_layers;L++) sink+=cpu_experts(layer_ptr(L),N_USED,fpe); }, "CPU-only (8 experts/layer)");
    // GPU branch alone (n_gpu experts copied+computed each layer, 1 sync/layer)
    double gpu_b = time_it([&]{ for(int L=0;L<n_layers;L++){ gpu_branch(L,n_gpu); CK(cudaStreamSynchronize(sG)); } }, "GPU-branch alone");
    // CPU branch alone (k experts)
    double cpu_b = time_it([&]{ for(int L=0;L<n_layers;L++) sink+=cpu_slice(L,k_cpu); }, "CPU-branch alone");
    // MANUAL SPLIT: per layer issue GPU async, run CPU (blocks), 1 sync/layer  <-- the thesis
    double split = time_it([&]{ for(int L=0;L<n_layers;L++){ gpu_branch(L,n_gpu); sink+=cpu_slice(L,k_cpu); CK(cudaStreamSynchronize(sG)); } }, "MANUAL SPLIT (overlap)");

    printf("\n--- analysis (n_layers=%d, k_cpu=%d, n_gpu=%d) ---\n", n_layers, k_cpu, n_gpu);
    printf("per-layer: CPU-branch=%.3f  GPU-branch=%.3f  (ideal split=max=%.3f, serial=sum=%.3f) ms\n",
           cpu_b/n_layers, gpu_b/n_layers, (cpu_b>gpu_b?cpu_b:gpu_b)/n_layers, (cpu_b+gpu_b)/n_layers);
    double ideal = (cpu_b>gpu_b?cpu_b:gpu_b);
    printf("split overlap eff = %.0f%% (100=%.3fms ideal, 0=%.3fms serial; got %.3f)\n",
           100.0*((cpu_b+gpu_b)-split)/((cpu_b+gpu_b)-ideal), ideal, cpu_b+gpu_b, split);
    printf(">>> SPEEDUP vs CPU-only = %.2fx  (%.1f -> %.1f tok/s)\n", cpu_only/split, 1000.0/cpu_only, 1000.0/split);
    (void)sink;
    return 0;
}
