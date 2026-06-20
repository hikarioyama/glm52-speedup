// read_bw.c — pure-READ memory bandwidth microbench, matched to the GLM-5.2 expert-read config:
//   anon 2 MiB-THP buffer (== A1: GGML_CUDA_NO_PINNED + GGML_CPU_HUGEPAGE), 24 threads pinned to
//   physical cores 0-23 (== A3), AVX-512 loads. Reports the platform's true read ceiling so we can
//   tell whether the model's measured ~107 GB/s effective expert-read is the wall (-> no kernel
//   headroom, settle for Tier-2) or leaking (-> attack the expert-read kernel).
//
//   build: gcc -O3 -march=native -fopenmp read_bw.c -o read_bw
//   run  : ./read_bw [bufGB=24] [iters=6]      (run with the chat CLOSED so 24 cores are free)
#define _GNU_SOURCE
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
#include <immintrin.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }

int main(int argc, char** argv){
    const size_t GB    = argc>1 ? (size_t)atoll(argv[1]) : 24;
    const int    iters = argc>2 ? atoi(argv[2]) : 6;
    const size_t N     = GB*(1ull<<30);
    const int    nth   = 24;

    prctl(PR_SET_THP_DISABLE, 0, 0, 0, 0);          // == A1: clear inherited THP-disable
    void* buf=NULL;
    if (posix_memalign(&buf, 1<<21, N)){ perror("alloc"); return 1; }
    madvise(buf, N, MADV_HUGEPAGE);                  // == A1: request 2 MiB pages
    memset(buf, 1, N);                               // fault in (2 MiB-backed)

    omp_set_num_threads(nth);
    #pragma omp parallel                             // == A3: pin thread i -> physical core i
    { int id=omp_get_thread_num(); cpu_set_t s; CPU_ZERO(&s); CPU_SET(id,&s);
      pthread_setaffinity_np(pthread_self(), sizeof(s), &s); }

    volatile uint64_t sink=0;

    // ---- (1) pure sequential read: each thread streams a contiguous slice ----
    double best_seq=0;
    for (int it=0; it<iters; ++it){
        double t0=now(); uint64_t total=0;
        #pragma omp parallel reduction(+:total)
        { int id=omp_get_thread_num(); size_t per=N/nth; const char* p=(const char*)buf+id*per;
          __m512i a=_mm512_setzero_si512();
          for (size_t i=0;i+64<=per;i+=64) a=_mm512_add_epi64(a,_mm512_load_si512((const void*)(p+i)));
          uint64_t t[8]; _mm512_storeu_si512(t,a); for(int k=0;k<8;k++) total+=t[k]; }
        sink+=total; double g=(double)N/(now()-t0)/1e9; if(g>best_seq) best_seq=g;
    }

    // ---- (2) "expert gather": hop across 2 MiB blocks in a non-sequential order (mimics reading
    //          top-k experts scattered in the layer's weight region; defeats simple stream prefetch) ----
    double best_g=0; const size_t blk=2ull<<20; const size_t nblk=N/blk;
    for (int it=0; it<iters; ++it){
        double t0=now(); uint64_t total=0;
        #pragma omp parallel reduction(+:total)
        { int id=omp_get_thread_num(); __m512i a=_mm512_setzero_si512();
          for (size_t b=id; b<nblk; b+=nth){ size_t off=((b*16)%nblk)*blk; const char* p=(const char*)buf+off;
              for (size_t i=0;i+64<=blk;i+=64) a=_mm512_add_epi64(a,_mm512_load_si512((const void*)(p+i))); }
          uint64_t t[8]; _mm512_storeu_si512(t,a); for(int k=0;k<8;k++) total+=t[k]; }
        sink+=total; double g=(double)(nblk*blk)/(now()-t0)/1e9; if(g>best_g) best_g=g;
    }

    printf("\n==== read-BW microbench (24t, cores0-23, 2MiB-THP, %zuGB) ====\n", GB);
    printf("SEQUENTIAL read : %.1f GB/s   (platform pure-read ceiling)\n", best_seq);
    printf("GATHER read     : %.1f GB/s   (scattered 2MiB blocks ~ expert pattern)\n", best_g);
    printf("reference       : model effective expert-read ~107 GB/s ; STREAM TRIAD 152 GB/s\n");
    printf("verdict hint    : if SEQUENTIAL ~107-130 -> model near platform wall (no kernel headroom).\n");
    printf("                  if SEQUENTIAL >=180     -> 107 is access-pattern limited (kernel headroom).\n");
    if (sink==42ull) printf("");   // keep the compiler honest
    return 0;
}
