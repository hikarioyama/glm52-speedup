#!/bin/bash
# A1 confirmation: alternate baseline vs nopinhuge to separate signal from noise.
cd ~/projects/glm52-speedup/harness
run() { # $1=label  $2=mode(base|nopin)
  unset GGML_CUDA_NO_PINNED GGML_CUDA_HOST_THP GGML_CPU_HUGEPAGE
  if [ "$2" = "nopin" ]; then export GGML_CUDA_NO_PINNED=1 GGML_CPU_HUGEPAGE=1; fi
  ./a1_thp_measure.sh "$1" 480 > "out/a1_thp_$1.console" 2>&1
  ts=$(grep -aE "eval time" "out/a1_thp_$1.log" | tail -1 | grep -oE "[0-9.]+ tokens per second" | grep -oE "^[0-9.]+")
  cc=$(grep -a "SYNC_TIME" "out/a1_thp_$1.log" | tail -1 | grep -oE "CUMUL.*" | grep -oE "cmp_cpu=[0-9]+" | grep -oE "[0-9]+")
  echo "RESULT $1 ($2): t/s=$ts cmp_cpu=${cc}us"
}
run nopin_r2 nopin
run base_r3  base
run nopin_r3 nopin
echo "BATCH DONE"
