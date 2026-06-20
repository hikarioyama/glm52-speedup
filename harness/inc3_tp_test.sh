#!/bin/bash
set -u
source ~/projects/glm52-speedup/harness/env.sh
TOOL="$LCPP/llama-mtp-alpha"
# TP: -sm row splits GPU-resident tensors row-wise across both GPUs (concurrent). Keep ONLY the
# offloaded experts (blk 0-21, 78) on CPU; let -sm row auto-split the rest across CUDA0+CUDA1.
OT="blk\.([0-9]|1[0-9]|2[01])\.ffn_(gate|up|down)_exps\.weight=CPU,blk\.78\.ffn_(gate|up|down)_exps\.weight=CPU"
export LLAMA_MTP_ALPHA_SEQ_FILE=/tmp/mtp_alpha_seq.txt LLAMA_MTP_SPEC_OUT=/tmp/mtp_spec_on.txt
echo "===== spec-ON (MTP in-graph) + TENSOR-PARALLEL (-sm row) $(date +%T) ====="
env GGML_CUDA_NO_PINNED=1 GGML_CPU_HUGEPAGE=1 LLAMA_MTP_SPEC=1 GGML_SCHED_SYNC_COUNT=3 \
  "$TOOL" -m "$MODEL" -ngl 999 -fa 1 --fit off --no-mmap -sm row \
    -ctk q8_0 -ctv q8_0 -ot "$OT" -c 4096 -b 512 -ub 512 -t 24 \
    --enable-mtp -n 256 -p "The history of the Roman Empire is a long and complex story that begins with"
echo "rc=$? $(date +%T)"
