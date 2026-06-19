// Probe: is Zen5 maddubs/madd 512 double-pumped vs 256? Pure-throughput test.
#include <immintrin.h>
#include <cstdio>
#include <ctime>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(){
  const long N=200000000L;
  __m256i a256=_mm256_set1_epi8(3),b256=_mm256_set1_epi8(7),acc256=_mm256_setzero_si256();
  __m512i a512=_mm512_set1_epi8(3),b512=_mm512_set1_epi8(7),acc512=_mm512_setzero_si512();
  // 256: 2 ops to cover same 512 lanes
  double t0=now();
  for(long i=0;i<N;i++){__m256i p=_mm256_maddubs_epi16(a256,b256);acc256=_mm256_add_epi16(acc256,p);a256=_mm256_add_epi8(a256,b256);}
  double t1=now();
  for(long i=0;i<N;i++){__m512i p=_mm512_maddubs_epi16(a512,b512);acc512=_mm512_add_epi16(acc512,p);a512=_mm512_add_epi8(a512,b512);}
  double t2=now();
  volatile int sink=_mm256_extract_epi16(acc256,0)+_mm_extract_epi16(_mm512_castsi512_si128(acc512),0);(void)sink;
  printf("256-bit maddubs: %.3f ns/op (%.0f Mops/s)\n",(t1-t0)*1e9/N,N/(t1-t0)/1e6);
  printf("512-bit maddubs: %.3f ns/op (%.0f Mops/s)\n",(t2-t1)*1e9/N,N/(t2-t1)/1e6);
  printf("512 throughput per-lane-equiv vs 256: %.2fx (1.0=double-pumped, 2.0=native512)\n",(t1-t0)/(t2-t1));
  return 0;
}
