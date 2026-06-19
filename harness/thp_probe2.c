// A1 diagnostic v2: is THP actually allocated for our anonymous region?
// Decisive signal = delta of /proc/vmstat thp_fault_alloc vs thp_fault_fallback
// across our own faults. Uses mmap (page-aligned) and a 2MiB-aligned variant.
// build: gcc -D_GNU_SOURCE -O2 thp_probe2.c -o thp_probe2 ; run: ./thp_probe2 [GB]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>

#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif

static long vmstat(const char * key) {
    FILE * f = fopen("/proc/vmstat","r"); if(!f) return -1;
    char k[64]; long v=-1, val;
    while (fscanf(f, "%63s %ld", k, &val) == 2) { if (!strcmp(k,key)) { v=val; break; } }
    fclose(f); return v;
}
static long self_anonhuge(void){
    FILE*f=fopen("/proc/self/smaps_rollup","r"); if(!f) return -1;
    char l[256]; long v=-1; while(fgets(l,sizeof l,f)) if(sscanf(l,"AnonHugePages: %ld kB",&v)==1) break;
    fclose(f); return v;
}

static void trial(const char * tag, size_t size, int align2m, int use_populate) {
    long fa0 = vmstat("thp_fault_alloc"), fb0 = vmstat("thp_fault_fallback");
    size_t maplen = size + (align2m ? (2u<<20) : 0);
    void * raw = mmap(NULL, maplen, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) { printf("[%s] mmap failed: %s\n", tag, strerror(errno)); return; }
    void * p = raw;
    if (align2m) { uintptr_t a=(uintptr_t)raw; uintptr_t al=(a+(2u<<20)-1)&~(uintptr_t)((2u<<20)-1); p=(void*)al; }
    printf("[%s] base=%p %%2MiB=%lu\n", tag, p, (unsigned long)((uintptr_t)p % (2u<<20)));
    errno=0; int mr = madvise(p, size, MADV_HUGEPAGE);
    printf("[%s] madvise(MADV_HUGEPAGE) rc=%d (%s)\n", tag, mr, mr?strerror(errno):"ok");
    if (use_populate) {
        errno=0; int pr = madvise(p, size, MADV_POPULATE_WRITE);
        printf("[%s] madvise(MADV_POPULATE_WRITE) rc=%d (%s)\n", tag, pr, pr?strerror(errno):"ok");
    } else {
        memset(p, 1, size);
    }
    long fa1 = vmstat("thp_fault_alloc"), fb1 = vmstat("thp_fault_fallback");
    long hp = self_anonhuge();
    printf("[%s] thp_fault_alloc +%ld  thp_fault_fallback +%ld  | AnonHugePages(rollup)=%ld kB (%.1f%% of %.1fGB)\n",
        tag, fa1-fa0, fb1-fb0, hp, hp<0?-1.0:100.0*hp*1024.0/size, size/1073741824.0);
    munmap(raw, maplen);
}

int main(int argc, char**argv){
    double gb = argc>1?atof(argv[1]):8.0; size_t size=(size_t)(gb*1073741824.0);
    char b[256]; FILE*f;
    f=fopen("/sys/kernel/mm/transparent_hugepage/enabled","r"); if(f){if(fgets(b,sizeof b,f))printf("enabled: %s",b);fclose(f);}
    f=fopen("/sys/kernel/mm/transparent_hugepage/defrag","r"); if(f){if(fgets(b,sizeof b,f))printf("defrag : %s",b);fclose(f);}
    printf("---- %.1f GB per trial ----\n", gb);
    trial("mmap memset       ", size, 0, 0);
    trial("mmap 2Maln memset ", size, 1, 0);
    trial("mmap 2Maln POPULATE", size, 1, 1);
    return 0;
}
