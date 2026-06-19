#!/bin/bash
cd ~/projects/glm52-speedup/harness
for r in 1 2 3; do
  unset GGML_CUDA_NO_PINNED GGML_CUDA_HOST_THP GGML_CPU_HUGEPAGE
  export EXTRA_ARGS="--cpu-mask FFFFFF --cpu-strict 1"
  ./a1_thp_measure.sh "basepin_r$r" 480 > "out/a1_thp_basepin_r$r.console" 2>&1
  ts=$(grep -aE "eval time" "out/a1_thp_basepin_r$r.log"|tail -1|grep -oE "[0-9.]+ tokens per second"|grep -oE "^[0-9.]+")
  cc=$(grep -a "SYNC_TIME" "out/a1_thp_basepin_r$r.log"|tail -1|grep -oE "CUMUL.*"|grep -oE "cmp_cpu=[0-9]+"|grep -oE "[0-9]+")
  sh=$(grep -oE "ShmemPmdMapped:[[:space:]]+[0-9]+" "out/a1_thp_basepin_r$r.smaps"|grep -oE "[0-9]+"|sort -n|tail -1)
  echo "RESULT basepin_r$r: t/s=$ts cmp_cpu=${cc}us ShmemPmd=${sh}kB"
done
echo "BASEPIN BATCH DONE"
