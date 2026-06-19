// B2 de-risk PoC: pinned H2D bandwidth + compute/copy overlap on RTX PRO 6000 (SM120)
// 問い: cold expert 転送(~1.4GB/token)を GPU 計算の裏に隠せるか。
//   1) 各GPUの実 pinned H2D 帯域 (Gen5 が本当に出るか / Max-Q が降速しないか)
//   2) compute stream と copy stream が重なるか (壁時計 ~= max(compute,copy) か sum か)
//   3) 両GPU同時 H2D の合計帯域 (PCIe root の天井)
#include <cstdio>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e){printf("ERR %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

// 適度に重いダミー計算カーネル (FMA ループ) — iters で時間を調整
__global__ void busy(float* d, size_t n, int iters){
  size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
  if(i>=n) return;
  float x = d[i];
  for(int k=0;k<iters;k++) x = fmaf(x, 1.0000001f, 0.0000001f);
  d[i]=x;
}

static float h2d_bw(int dev, size_t bytes, int reps){
  cudaSetDevice(dev);
  void *h,*dptr;
  cudaHostAlloc(&h, bytes, cudaHostAllocDefault);     // pinned
  cudaMalloc(&dptr, bytes);
  cudaStream_t s; cudaStreamCreate(&s);
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  // warmup
  cudaMemcpyAsync(dptr,h,bytes,cudaMemcpyHostToDevice,s); cudaStreamSynchronize(s);
  cudaEventRecord(a,s);
  for(int r=0;r<reps;r++) cudaMemcpyAsync(dptr,h,bytes,cudaMemcpyHostToDevice,s);
  cudaEventRecord(b,s); cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms,a,b);
  cudaFreeHost(h); cudaFree(dptr); cudaStreamDestroy(s);
  cudaEventDestroy(a); cudaEventDestroy(b);
  return (bytes*(double)reps/1e9)/(ms/1e3); // GB/s
}

int main(){
  int ndev; CK(cudaGetDeviceCount(&ndev));
  printf("GPUs: %d\n", ndev);
  const size_t BYTES = 1400ull*1024*1024; // ~1.4GB = cold/token 見積
  const int REPS = 10;

  // (1) 各GPU 単独 H2D 帯域
  for(int d=0; d<ndev; d++){
    float bw = h2d_bw(d, BYTES, REPS);
    int gen=0,w=0; cudaDeviceGetAttribute(&gen, cudaDevAttrPciDeviceId, d); // placeholder
    printf("[GPU%d] pinned H2D %.1f GB/s  (1.4GB -> %.2f ms)\n", d, bw, BYTES/1e9/bw*1e3);
  }

  // (3) 両GPU 同時 H2D (PCIe root 合計天井)
  if(ndev>=2){
    void *h0,*h1,*d0,*d1; cudaStream_t s0,s1; cudaEvent_t a,b;
    cudaSetDevice(0); cudaHostAlloc(&h0,BYTES,0); cudaMalloc(&d0,BYTES); cudaStreamCreate(&s0);
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaSetDevice(1); cudaHostAlloc(&h1,BYTES,0); cudaMalloc(&d1,BYTES); cudaStreamCreate(&s1);
    cudaSetDevice(0); cudaEventRecord(a,0);
    for(int r=0;r<REPS;r++){
      cudaSetDevice(0); cudaMemcpyAsync(d0,h0,BYTES,cudaMemcpyHostToDevice,s0);
      cudaSetDevice(1); cudaMemcpyAsync(d1,h1,BYTES,cudaMemcpyHostToDevice,s1);
    }
    cudaSetDevice(0); cudaEventRecord(b,0); cudaEventSynchronize(b);
    cudaSetDevice(1); cudaStreamSynchronize(s1);
    float ms; cudaSetDevice(0); cudaEventElapsedTime(&ms,a,b);
    double gb = BYTES*2.0*REPS/1e9;
    printf("[BOTH] simultaneous H2D total %.1f GB/s\n", gb/(ms/1e3));
  }

  // (2) overlap: compute stream A + copy stream B (GPU0)
  {
    cudaSetDevice(0);
    size_t N = 64ull*1024*1024; // 256MB working set for busy kernel
    float* dd; cudaMalloc(&dd, N*sizeof(float));
    void *h,*dcopy; cudaHostAlloc(&h,BYTES,0); cudaMalloc(&dcopy,BYTES);
    cudaStream_t sc, scopy; cudaStreamCreate(&sc); cudaStreamCreate(&scopy);
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    int iters = 4000; // ~busy time を copy と同程度に調整
    dim3 bl(256), gr((N+255)/256);

    auto timeit=[&](bool overlap)->float{
      cudaEventRecord(a,0);
      if(overlap){
        busy<<<gr,bl,0,sc>>>(dd,N,iters);
        cudaMemcpyAsync(dcopy,h,BYTES,cudaMemcpyHostToDevice,scopy);
      } else {
        busy<<<gr,bl,0,sc>>>(dd,N,iters); cudaStreamSynchronize(sc);
        cudaMemcpyAsync(dcopy,h,BYTES,cudaMemcpyHostToDevice,scopy); cudaStreamSynchronize(scopy);
      }
      cudaDeviceSynchronize();
      cudaEventRecord(b,0); cudaEventSynchronize(b);
      float ms; cudaEventElapsedTime(&ms,a,b); return ms;
    };
    // measure compute-only and copy-only
    cudaEventRecord(a,0); busy<<<gr,bl,0,sc>>>(dd,N,iters); cudaDeviceSynchronize();
    cudaEventRecord(b,0); cudaEventSynchronize(b); float tc; cudaEventElapsedTime(&tc,a,b);
    cudaEventRecord(a,0); cudaMemcpyAsync(dcopy,h,BYTES,cudaMemcpyHostToDevice,scopy); cudaDeviceSynchronize();
    cudaEventRecord(b,0); cudaEventSynchronize(b); float tcopy; cudaEventElapsedTime(&tcopy,a,b);
    float tser = timeit(false);
    float tovl = timeit(true);
    printf("[OVERLAP GPU0] compute=%.2fms copy=%.2fms  serial=%.2fms overlap=%.2fms\n",
           tc, tcopy, tser, tovl);
    printf("  -> overlap %s (overlap %.2fms vs max(c,copy) %.2fms; sum %.2fms)\n",
           tovl < (tc+tcopy)*0.7 ? "WORKS" : "FAILED", tovl, tc>tcopy?tc:tcopy, tc+tcopy);
  }
  return 0;
}
