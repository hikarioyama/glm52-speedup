// frag_bench.c — does the SAME q2_K gemv work, but split into many small omp-parallel regions
// (a barrier between each), to isolate the per-op thread-dispatch/barrier overhead that the model
// pays (~66 CPU MoE ops/token) vs the single-region kernel ceiling (~242 GB/s).
// Pinning is left to OMP_PROC_BIND/OMP_PLACES (no per-region syscall confound). Sweeps region size;
// the model's per-op work is ~26 MB, so watch the ~13000-rows row.
//
//   build: L=$HOME/src/llama.cpp; gcc -O3 -march=native -fopenmp -I $L/ggml/include frag_bench.c \
//          -o frag_bench -L $L/build/bin -lggml-cpu -lggml-base -Wl,-rpath,$L/build/bin
//   run  : OMP_NUM_THREADS=24 OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=active ./frag_bench
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

static void* W; static size_t Nrows, rowB; static void* A; static int Kg;
static const struct ggml_type_traits_cpu* TR;
static volatile double sink=0;

static double run_frag(size_t rows_per_region, int iters){
    double best=0;
    for (int it=0; it<iters; ++it){
        double t0=now(); double s=0;
        for (size_t base=0; base<Nrows; base+=rows_per_region){
            size_t hi=base+rows_per_region; if (hi>Nrows) hi=Nrows;
            double racc=0;
            #pragma omp parallel for schedule(static) reduction(+:racc)
            for (size_t r=base; r<hi; ++r){ float o; TR->vec_dot(Kg,&o,0,(const char*)W+r*rowB,0,A,0,1); racc+=o; }
            s+=racc;
        }
        sink+=s;
        double g=(double)(Nrows*rowB)/(now()-t0)/1e9; if(g>best) best=g;
    }
    return best;
}

int main(int argc, char**argv){
    int K = argc>1?atoi(argv[1]):6144;
    double GB = argc>2?atof(argv[2]):8.0;
    int iters = argc>3?atoi(argv[3]):4;
    Kg=K;
    struct ggml_init_params ip={16<<20,NULL,true}; ggml_init(ip);
    TR = ggml_get_type_traits_cpu(GGML_TYPE_Q2_K);
    rowB = ggml_row_size(GGML_TYPE_Q2_K, K);
    Nrows = (size_t)(GB*(1ull<<30))/rowB;
    prctl(PR_SET_THP_DISABLE,0,0,0,0);
    if (posix_memalign(&W,1<<21,Nrows*rowB)){perror("W");return 1;}
    madvise(W,Nrows*rowB,MADV_HUGEPAGE); memset(W,7,Nrows*rowB);
    const struct ggml_type_traits_cpu* atr=ggml_get_type_traits_cpu(TR->vec_dot_type);
    float* af=malloc((size_t)K*4); for(int i=0;i<K;i++) af[i]=(float)((i%17)-8)*0.1f;
    A=malloc(ggml_row_size(TR->vec_dot_type,K)); atr->from_float(af,A,K);

    printf("=== q2_K gemv fragmented (24t, threads=%d, wait=%s) ===\n",
           omp_get_max_threads(), getenv("OMP_WAIT_POLICY")?getenv("OMP_WAIT_POLICY"):"(default)");
    printf("matrix=%.1fGB rows=%zu row=%zuB\n", Nrows*rowB/1e9, Nrows, rowB);
    size_t sweep[] = { Nrows, 200000, 50000, 13000, 6000, 2000, 800 };  // region sizes (rows)
    for (unsigned i=0;i<sizeof(sweep)/sizeof(sweep[0]);++i){
        size_t rpr = sweep[i]>Nrows?Nrows:sweep[i];
        double g = run_frag(rpr, iters);
        printf("region=%7zu rows (~%5.0f MB, %5zu regions): %6.1f GB/s\n",
               rpr, (double)rpr*rowB/1e6, (Nrows+rpr-1)/rpr, g);
    }
    printf("ref: single-region kernel 242 ; model decode 107 ; model per-op work ~26MB (~13000 rows)\n");
    if (sink==123.456) printf("");
    return 0;
}
