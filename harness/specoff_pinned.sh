#!/bin/bash
set -u
source ~/projects/glm52-speedup/harness/env.sh
TOOL="$LCPP/llama-mtp-alpha"
OT="blk\.([0-9]|1[0-9]|2[01])\.ffn_(gate|up|down)_exps\.weight=CPU,blk\.(2[2-9]|3[0-9]|4[0-9])\.ffn_(gate|up|down)_exps\.weight=CUDA0,blk\.(5[0-9]|6[0-9]|7[0-7])\.ffn_(gate|up|down)_exps\.weight=CUDA1,blk\.78\.ffn_(gate|up|down)_exps\.weight=CPU"
export LLAMA_MTP_ALPHA_SEQ_FILE=/tmp/mtp_alpha_seq.txt
echo "===== spec-OFF baseline A3-PINNED (apples-to-apples) $(date +%T) ====="
env GGML_CUDA_NO_PINNED=1 GGML_CPU_HUGEPAGE=1 \
  "$TOOL" -m "$MODEL" -ngl 999 -fa 1 --fit off --no-mmap \
    -ctk q8_0 -ctv q8_0 --cpu-mask FFFFFF --cpu-strict 1 \
    -ot "$OT" -c 4096 -b 512 -ub 512 -t 24 \
    --enable-mtp -n 256 -p "The history of the Roman Empire is a long and complex story that begins with"
echo "rc=$? $(date +%T)"
