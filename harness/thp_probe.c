// A1 diagnostic: does posix_memalign(64,...) + madvise(MADV_HUGEPAGE) actually get
// THP-backed under the current system THP knobs? Isolates the llama.cpp result.
// build: gcc -D_GNU_SOURCE -O2 thp_probe.c -o thp_probe ; run: ./thp_probe [GB]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>

static long anon_hugepages_kb(void) {
    FILE * f = fopen("/proc/self/smaps_rollup", "r");
    if (!f) return -1;
    char line[256]; long v = -1;
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "AnonHugePages: %ld kB", &v) == 1) break;
    }
    fclose(f);
    return v;
}

static void run(const char * tag, size_t size, int do_madv, int page_align) {
    void * p = NULL;
    int rc = posix_memalign(&p, 64, size);
    if (rc != 0 || !p) { printf("[%s] posix_memalign failed rc=%d\n", tag, rc); return; }
    uintptr_t a = (uintptr_t)p;
    long ps = sysconf(_SC_PAGESIZE);
    printf("[%s] ptr=%p  %%4096=%lu  %%2MiB=%lu  pagesize=%ld\n",
           tag, p, (unsigned long)(a % 4096), (unsigned long)(a % (2u<<20)), ps);
    if (do_madv) {
        void * mp = p; size_t ms = size;
        if (page_align) {
            uintptr_t aligned = (a + ps - 1) & ~((uintptr_t)ps - 1);
            mp = (void*)aligned;
            ms = (size - (aligned - a)) & ~((size_t)ps - 1);
        }
        errno = 0;
        int mr = madvise(mp, ms, MADV_HUGEPAGE);
        printf("[%s] madvise(%p, %zu) rc=%d errno=%d (%s)\n",
               tag, mp, ms, mr, errno, mr ? strerror(errno) : "ok");
    }
    // fault every page
    memset(p, 1, size);
    // touch again to give khugepaged nothing to do differently; read back
    volatile char sink = 0; for (size_t i = 0; i < size; i += 4096) sink ^= ((char*)p)[i]; (void)sink;
    long hp = anon_hugepages_kb();
    printf("[%s] AnonHugePages(rollup)=%ld kB  of %.1f GB  => %.1f%% huge\n",
           tag, hp, size/1073741824.0, hp < 0 ? -1.0 : 100.0*hp*1024.0/size);
    free(p);
}

int main(int argc, char ** argv) {
    double gb = argc > 1 ? atof(argv[1]) : 8.0;
    size_t size = (size_t)(gb * 1073741824.0);
    char buf[4096]; FILE * f;
    f = fopen("/sys/kernel/mm/transparent_hugepage/enabled","r");
    if (f){ if(fgets(buf,sizeof buf,f)) printf("THP enabled: %s", buf); fclose(f);}
    f = fopen("/sys/kernel/mm/transparent_hugepage/defrag","r");
    if (f){ if(fgets(buf,sizeof buf,f)) printf("THP defrag : %s", buf); fclose(f);}
    printf("---- size = %.1f GB ----\n", gb);
    run("no-madv      ", size, 0, 0);
    run("madv-raw     ", size, 1, 0);   // madvise on the raw 64-aligned ptr (what my patch did)
    run("madv-pagealn ", size, 1, 1);   // madvise on a page-aligned subrange (the fix)
    return 0;
}
