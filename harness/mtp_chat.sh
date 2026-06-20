#!/bin/bash
# Interactive terminal chat with GLM-5.2 accelerated by the MTP self-spec loop (~30 t/s, greedy).
# Run this in YOUR terminal. Model load ~3min, then type. Thinking OFF by default (THINK=1 to enable).
# Commands: /think /nothink /reset /quit
source ~/projects/glm52-speedup/harness/env.sh
OT="blk\.([0-9]|1[0-9]|2[01])\.ffn_(gate|up|down)_exps\.weight=CPU,blk\.(2[2-9]|3[0-9]|4[0-9])\.ffn_(gate|up|down)_exps\.weight=CUDA0,blk\.(5[0-9]|6[0-9]|7[0-7])\.ffn_(gate|up|down)_exps\.weight=CUDA1,blk\.78\.ffn_(gate|up|down)_exps\.weight=CPU"
exec env GGML_CUDA_NO_PINNED=1 GGML_CPU_HUGEPAGE=1 LLAMA_MTP_SPEC=1 LLAMA_MTP_CHAT=1 \
  "$LCPP/llama-mtp-alpha" -m "$MODEL" -ngl 999 -fa 1 --fit off --no-mmap \
    -ctk q8_0 -ctv q8_0 -ot "$OT" -c 8192 -b 512 -ub 512 -t 24 --enable-mtp -n 2048
