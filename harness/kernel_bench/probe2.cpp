// Is vpdpbusd (VNNI) double-pumped too on Zen5?
#include <immintrin.h>
#include <cstdio>
#include <ctime>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(){
  const long N=200000000L;
  __m256i a256=_mm256_set1_epi8(3),b256=_mm256_set1_epi8(7),acc256=_mm256_setzero_si256();
  __m512i a512=_mm512_set1_epi8(3),b512=_mm512_set1_epi8(7),acc512=_mm512_setzero_si512();
  double t0=now();
  for(long i=0;i<N;i++){acc256=_mm256_dpbusd_epi32(acc256,a256,b256);a256=_mm256_add_epi8(a256,b256);}
  double t1=now();
  for(long i=0;i<N;i++){acc512=_mm512_dpbusd_epi32(acc512,a512,b512);a512=_mm512_add_epi8(a512,b512);}
  double t2=now();
  volatile int s=_mm256_extract_epi32(acc256,0)+_mm_extract_epi32(_mm512_castsi512_si128(acc512),0);(void)s;
  printf("256 vpdpbusd: %.3f ns/op | 512 vpdpbusd: %.3f ns/op | 512-vs-256 lane-equiv: %.2fx\n",
         (t1-t0)*1e9/N,(t2-t1)*1e9/N,(t1-t0)/(t2-t1));
  // also: maddubs+madd (2 ops) vs single dpbusd for the SAME 32-int8 reduction
  return 0;
}
