#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/prctl.h>
int main(void){
    printf("PR_GET_THP_DISABLE = %d\n", prctl(PR_GET_THP_DISABLE,0,0,0,0));
    { FILE*f=fopen("/proc/self/status","r"); char l[256];
      while(f&&fgets(l,sizeof l,f)) if(strstr(l,"THP")||strstr(l,"Huge")) printf("status: %s",l);
      if(f)fclose(f); }
    size_t size=(size_t)4*1073741824ULL;
    size_t maplen=size+(2u<<20);
    void*raw=mmap(NULL,maplen,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(raw==MAP_FAILED){perror("mmap");return 1;}
    uintptr_t a=(uintptr_t)raw, al=(a+(2u<<20)-1)&~(uintptr_t)((2u<<20)-1);
    void*p=(void*)al;
    int mr=madvise(p,size,MADV_HUGEPAGE);
    printf("madvise rc=%d (%s)  base=%p %%2MiB=%lu\n",mr,mr?strerror(errno):"ok",p,(unsigned long)(al%(2u<<20)));
    memset(p,1,size);
    // dump the smaps entry covering p
    char key[32]; snprintf(key,sizeof key,"%lx-",(unsigned long)al);
    FILE*f=fopen("/proc/self/smaps","r"); char l[512]; int hit=0,printed=0;
    while(f&&fgets(l,sizeof l,f)){
        if(l[0]>='0'&&((l[0]<='9')||(l[0]>='a'&&l[0]<='f'))){ // header line
            unsigned long s=strtoul(l,NULL,16);
            hit = (s==al || (s<=al && al < s+ (2UL<<40))) && (strstr(l,"-")!=NULL) && s==al;
            if(hit){printf("MAP: %s",l);printed=0;}
        } else if(hit){
            if(strstr(l,"AnonHugePages")||strstr(l,"THPeligible")||strstr(l,"VmFlags")||strstr(l,"Rss")){printf("  %s",l);printed++;}
            if(strstr(l,"VmFlags")) break;
        }
    }
    if(f)fclose(f);
    munmap(raw,maplen);
    return 0;
}
