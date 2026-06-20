#!/bin/bash
set -u
source ~/projects/glm52-speedup/harness/env.sh
TOOL="$LCPP/llama-mtp-alpha"
# layers 0-21 FULLY on CPU (attn+experts+norms) -> contiguous CPU work, no per-layer ping-pong.
# 22-77 experts split CUDA0/CUDA1 (attn on GPU, as now). 78 experts CPU (nextn).
OT="blk\.([0-9]|1[0-9]|2[01])\.=CPU,blk\.(2[2-9]|3[0-9]|4[0-9])\.ffn_(gate|up|down)_exps\.weight=CUDA0,blk\.(5[0-9]|6[0-9]|7[0-7])\.ffn_(gate|up|down)_exps\.weight=CUDA1,blk\.78\.ffn_(gate|up|down)_exps\.weight=CPU"
export LLAMA_MTP_ALPHA_SEQ_FILE=/tmp/mtp_alpha_seq.txt LLAMA_MTP_SPEC_OUT=/tmp/mtp_spec_on.txt
echo "===== spec-ON (MTP) + CONTIGUOUS-CPU layers 0-21 $(date +%T) ====="
env GGML_CUDA_NO_PINNED=1 GGML_CPU_HUGEPAGE=1 LLAMA_MTP_SPEC=1 GGML_SCHED_SYNC_COUNT=3 \
  "$TOOL" -m "$MODEL" -ngl 999 -fa 1 --fit off --no-mmap \
    -ctk q8_0 -ctv q8_0 -ot "$OT" -c 4096 -b 512 -ub 512 -t 24 \
    --enable-mtp -n 256 -p "The history of the Roman Empire is a long and complex story that begins with"
echo "rc=$? $(date +%T)"
