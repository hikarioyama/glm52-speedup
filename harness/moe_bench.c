// moe_bench.c — faithful mimic of ggml_compute_forward_mul_mat_id for DECODE (batch 1), to see
// whether the model's ~107 GB/s effective is the realistic fragmented-MoE structure limit or there
// is wrapper headroom vs the idealized single-contiguous-matrix vec_dot ceiling (242 GB/s).
// Structure (from ggml-cpu.c:1525): per op, the top-8 experts are processed SERIALLY; each expert's
// 2048 rows are split across 24 threads (~86 rows/thread); one barrier/op. 256-expert pool (1GB,
// >LLC, 2MiB-THP) streamed by rotating the selected experts -> RAM-bound like the model.
//
//   build: L=$HOME/src/llama.cpp; gcc -O3 -march=native -fopenmp -I $L/ggml/include moe_bench.c \
//          -o moe_bench -L $L/build/bin -lggml-cpu -lggml-base -Wl,-rpath,$L/build/bin
//   run  : OMP_NUM_THREADS=24 OMP_PROC_BIND=close OMP_PLACES=cores ./moe_bench
#define _GNU_SOURCE
#include "ggml.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <omp.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

int main(int argc, char** argv){
    const int K        = argc>1 ? atoi(argv[1]) : 6144;   // n_embd (contract dim)
    const int N        = argc>2 ? atoi(argv[2]) : 2048;   // n_ff_exp (rows per expert)
    const int n_expert = argc>3 ? atoi(argv[3]) : 256;
    const int n_used   = argc>4 ? atoi(argv[4]) : 8;
    const int n_ops    = argc>5 ? atoi(argv[5]) : 4000;   // mul_mat_id ops to time
    const int nth      = 24;
    struct ggml_init_params ip={16<<20,NULL,true}; ggml_init(ip);
    const struct ggml_type_traits_cpu* TR = ggml_get_type_traits_cpu(GGML_TYPE_Q2_K);
    const size_t rowB = ggml_row_size(GGML_TYPE_Q2_K, K);
    const size_t expB = (size_t)N * rowB;                 // one expert's weight bytes (~4MB)
    const size_t pool = (size_t)n_expert * expB;          // 256 experts (~1GB)

    prctl(PR_SET_THP_DISABLE,0,0,0,0);
    char* W=NULL; if (posix_memalign((void**)&W,1<<21,pool)){perror("W");return 1;}
    madvise(W,pool,MADV_HUGEPAGE); memset(W,7,pool);
    // activation: 1 token, q8_K
    const struct ggml_type_traits_cpu* atr=ggml_get_type_traits_cpu(TR->vec_dot_type);
    float* af=malloc((size_t)K*4); for(int i=0;i<K;i++) af[i]=(float)((i%17)-8)*0.1f;
    char* A=malloc(ggml_row_size(TR->vec_dot_type,K)); atr->from_float(af,A,K);

    volatile double sink=0;
    // warm + time
    double best=0;
    for (int rep=0; rep<4; ++rep){
        double t0=now(); double s=0;
        for (int op=0; op<n_ops; ++op){
            // ggml-faithful: ONE parallel region per op (1 barrier); all n_used experts' rows
            // (n_used*N total) distributed across the 24 threads in contiguous chunks.
            const long total = (long)n_used * N;
            double racc=0;
            #pragma omp parallel for schedule(static) reduction(+:racc) num_threads(nth)
            for (long t=0; t<total; ++t){
                int u = (int)(t / N);
                int r = (int)(t % N);
                int e = (op*n_used + u) % n_expert;
                const char* ew = W + (size_t)e*expB + (size_t)r*rowB;
                float o; TR->vec_dot(K,&o,0,ew,0,A,0,1); racc+=o;
            }
            s+=racc;
        }
        sink+=s;
        double bytes=(double)n_ops*n_used*expB;
        double g=bytes/(now()-t0)/1e9; if(g>best)best=g;
    }
    printf("=== MoE mul_mat_id mimic (batch1, %d experts top-%d, K=%d N=%d, 24t) ===\n", n_expert,n_used,K,N);
    printf("expert=%.1fMB pool=%.2fGB  ->  %.1f GB/s\n", expB/1e6, pool/1e9, best);
    printf("ref: idealized vec_dot 242 ; model decode effective 107\n");
    printf("  ~107 -> fragmented-MoE structure IS the limit (wrapper not the lever).\n");
    printf("  ~200 -> structure is fine; the model's 107 is dispatch/sync overhead (C lever real).\n");
    if (sink==123.0) printf("");
    return 0;
}
