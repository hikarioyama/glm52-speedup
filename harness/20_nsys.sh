#!/bin/bash
# Nsight Systems: 短い decode の timeline で「真の律速カーネル」と launch/sync overhead を晒す。
source ~/projects/glm52-speedup/harness/env.sh
NCPUMOE=${1:-0}
REP=$OUT/20_nsys_ncpumoe$NCPUMOE
PROMPT="Explain in detail how a Mixture-of-Experts transformer routes tokens."

$NSYS profile \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --sample=none \
  -o $REP --force-overwrite=true \
  $LCPP/llama-cli -m "$MODEL" -ngl 999 --n-cpu-moe $NCPUMOE -fa 1 \
    -no-cnv -p "$PROMPT" -n 64 2>&1 | tail -20

echo "=== top kernels by total GPU time ==="
$NSYS stats --report cuda_gpu_kern_sum --format table $REP.nsys-rep 2>/dev/null | head -30 | tee $OUT/20_nsys_kernsum.txt
echo "=== memops (HtoD = expert offload転送が見える) ==="
$NSYS stats --report cuda_gpu_mem_time_sum --format table $REP.nsys-rep 2>/dev/null | head -20 | tee $OUT/20_nsys_memops.txt
