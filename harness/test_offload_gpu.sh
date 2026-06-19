#!/bin/bash
# Q5-3 実験: env+flagだけで cold expert を GPU計算(used-expert copy経路)に切替えられるか
# 期待: GGML_OP_OFFLOAD_MIN_BATCH=1 で decode MUL_MAT_ID が CUDA割当 → host(CUDA_Host)の
# expert を used分だけ copy して GPU計算。26 t/s から動くか?
source ~/projects/glm52-speedup/harness/env.sh
TAG=${1:-min_batch1}
LOG=$OUT/test_offload_$TAG.log
# cold=CPU(--no-mmapでCUDA_Host化) 19層 / hot CUDA0,1。comma単一-ot(正式形)
OT="blk\.([0-9]|1[0-9]|2[01])\.ffn_(gate|up|down)_exps\.weight=CPU,blk\.(2[2-9]|3[0-9]|4[0-9])\.ffn_(gate|up|down)_exps\.weight=CUDA0,blk\.(5[0-9]|6[0-9]|7[0-9])\.ffn_(gate|up|down)_exps\.weight=CUDA1"

GGML_OP_OFFLOAD_MIN_BATCH=${MINB:-1} \
$LCPP/llama-completion -m "$MODEL" -ngl 999 -fa 1 --fit off --no-mmap \
  -ot "$OT" \
  -c 4096 -n 80 --no-warmup \
  -p "The theory of computation studies how problems can be solved using algorithms, and it" \
  2>&1 | tee $LOG | grep -iE 'eval time|per second|cudaMalloc|out of memory|error|CUDA0 model|offload' | tail
