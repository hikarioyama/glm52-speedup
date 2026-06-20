// vec_dot_bench.c — measure the REAL ggml q2_K / q3_K CPU vec_dot kernel throughput in a gemv
// (matrix x 1-vector = the decode case), isolated from llama.cpp's dispatch/threading/attention.
// Streams an >LLC weight matrix from RAM (2MiB-THP, == A1) with 24 threads pinned to cores 0-23.
// Verdict: if ~107 GB/s the kernel itself is the decode wall (optimize it); if ~200+ the model's
// 107 is overhead elsewhere (mul_mat_id dispatch / sync), not the kernel.
//
// build (from anywhere):
//   L=$HOME/src/llama.cpp
//   gcc -O3 -march=native -fopenmp -I $L/ggml/include vec_dot_bench.c -o vec_dot_bench \
//       -L $L/build/bin -lggml-cpu -lggml-base -Wl,-rpath,$L/build/bin
//   ./vec_dot_bench [K=6144] [matrixGB=8] [iters=5]
#define _GNU_SOURCE
#include "ggml.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sched.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <omp.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }

static void bench(enum ggml_type wt, const char* name, int K, double GB, int iters, int nth){
    const struct ggml_type_traits_cpu * tr = ggml_get_type_traits_cpu(wt);
    if (!tr || !tr->vec_dot){ printf("%s: no vec_dot\n", name); return; }
    const size_t row = ggml_row_size(wt, K);          // bytes per weight row (K elems, quantized)
    size_t N = (size_t)(GB*(1ull<<30)) / row;          // rows so the matrix > LLC -> RAM-streamed
    const size_t total = N * row;

    prctl(PR_SET_THP_DISABLE,0,0,0,0);                 // == A1
    void* W=NULL; if (posix_memalign(&W, 1<<21, total)){ perror("alloc W"); return; }
    madvise(W, total, MADV_HUGEPAGE); memset(W, 7, total);   // garbage bytes are fine for timing

    // activation: 1 row in the kernel's vec_dot_type (q8_K), quantized from a real f32 vector
    enum ggml_type at = tr->vec_dot_type;
    const struct ggml_type_traits_cpu * atr = ggml_get_type_traits_cpu(at);
    float* af = malloc((size_t)K*sizeof(float));
    for (int i=0;i<K;i++) af[i] = (float)((i%17)-8)*0.1f;
    void* A = malloc(ggml_row_size(at, K));
    atr->from_float(af, A, K);

    volatile double sink=0; double best=0;
    for (int it=0; it<iters; ++it){
        double t0=now();
        #pragma omp parallel num_threads(nth)
        {
            int id=omp_get_thread_num(); cpu_set_t s; CPU_ZERO(&s); CPU_SET(id,&s);
            pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
            size_t per=N/nth, r0=(size_t)id*per, r1=(id==nth-1)?N:r0+per;
            double acc=0;
            for (size_t r=r0;r<r1;++r){ float o; tr->vec_dot(K,&o,0,(const char*)W+r*row,0,A,0,1); acc+=o; }
            #pragma omp atomic
            sink += acc;
        }
        double dt=now()-t0, g=(double)total/dt/1e9; if (g>best) best=g;
    }
    printf("%-5s gemv vec_dot : %6.1f GB/s   (rows=%zu, row=%zuB, matrix=%.1fGB, K=%d)\n",
           name, best, N, row, total/1e9, K);
    free(W); free(af); free(A);
    if (sink==123456.789) printf("");
}

int main(int argc, char** argv){
    int    K     = argc>1 ? atoi(argv[1]) : 6144;
    double GB    = argc>2 ? atof(argv[2]) : 8.0;
    int    iters = argc>3 ? atoi(argv[3]) : 5;
    struct ggml_init_params ip = { 16*1024*1024, NULL, true }; ggml_init(ip); // init f16 tables etc.
    printf("=== vec_dot gemv microbench (24t cores0-23, 2MiB-THP) ===\n");
    bench(GGML_TYPE_Q2_K, "q2_K", K, GB, iters, 24);
    bench(GGML_TYPE_Q3_K, "q3_K", K, GB, iters, 24);
    printf("ref: model decode effective ~107 GB/s ; pure read 248 GB/s\n");
    printf("  ~107 -> kernel IS the wall (optimize vec_dot).  ~200+ -> 107 is dispatch/sync overhead.\n");
    return 0;
}
