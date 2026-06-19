#!/bin/bash
cd ~/projects/glm52-speedup/harness
for r in 1 2 3; do
  unset GGML_CUDA_HOST_THP
  export GGML_CUDA_NO_PINNED=1 GGML_CPU_HUGEPAGE=1
  ./a1_thp_measure.sh "a2pf_r$r" 480 > "out/a1_thp_a2pf_r$r.console" 2>&1
  ts=$(grep -aE "eval time" "out/a1_thp_a2pf_r$r.log" | tail -1 | grep -oE "[0-9.]+ tokens per second" | grep -oE "^[0-9.]+")
  cc=$(grep -a "SYNC_TIME" "out/a1_thp_a2pf_r$r.log" | tail -1 | grep -oE "CUMUL.*" | grep -oE "cmp_cpu=[0-9]+" | grep -oE "[0-9]+")
  echo "RESULT a2pf_r$r: t/s=$ts cmp_cpu=${cc}us"
done
echo "A2 BATCH DONE"
