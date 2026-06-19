#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/prctl.h>
static long vmstat(const char*k){FILE*f=fopen("/proc/vmstat","r");if(!f)return -1;char kk[64];long v=-1,val;while(fscanf(f,"%63s %ld",kk,&val)==2)if(!strcmp(kk,k)){v=val;break;}fclose(f);return v;}
int main(void){
    printf("before: THP_DISABLE=%d\n", prctl(PR_GET_THP_DISABLE,0,0,0,0));
    int cr = prctl(PR_SET_THP_DISABLE, 0, 0, 0, 0);
    printf("prctl(SET_THP_DISABLE,0) rc=%d (%s)\n", cr, cr?strerror(errno):"ok");
    printf("after:  THP_DISABLE=%d\n", prctl(PR_GET_THP_DISABLE,0,0,0,0));
    size_t size=(size_t)4*1073741824ULL, maplen=size+(2u<<20);
    long fa0=vmstat("thp_fault_alloc"), fb0=vmstat("thp_fault_fallback");
    void*raw=mmap(NULL,maplen,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    uintptr_t a=(uintptr_t)raw, al=(a+(2u<<20)-1)&~(uintptr_t)((2u<<20)-1); void*p=(void*)al;
    madvise(p,size,MADV_HUGEPAGE);
    memset(p,1,size);
    long fa1=vmstat("thp_fault_alloc"), fb1=vmstat("thp_fault_fallback");
    long hp=-1; { FILE*f=fopen("/proc/self/smaps_rollup","r");char l[256];while(f&&fgets(l,sizeof l,f))if(sscanf(l,"AnonHugePages: %ld kB",&hp)==1)break; if(f)fclose(f);}
    printf("thp_fault_alloc +%ld  fallback +%ld  AnonHugePages=%ld kB (%.1f%% of 4GB)\n",
        fa1-fa0, fb1-fb0, hp, 100.0*hp*1024.0/size);
    munmap(raw,maplen);
    return 0;
}
