#!/bin/bash
set -u
source ~/projects/glm52-speedup/harness/env.sh
TOOL="$LCPP/llama-mtp-alpha"
OT="blk\.([0-9]|1[0-9]|2[01])\.ffn_(gate|up|down)_exps\.weight=CPU,blk\.(2[2-9]|3[0-9]|4[0-9])\.ffn_(gate|up|down)_exps\.weight=CUDA0,blk\.(5[0-9]|6[0-9]|7[0-7])\.ffn_(gate|up|down)_exps\.weight=CUDA1,blk\.78\.ffn_(gate|up|down)_exps\.weight=CPU"
export LLAMA_MTP_ALPHA_SEQ_FILE=/tmp/mtp_alpha_seq.txt
export LLAMA_MTP_SPEC_OUT=/tmp/mtp_spec_on.txt
echo "===== spec-ON (cheap draft) + TRACE $(date +%T) ====="
env GGML_CUDA_NO_PINNED=1 GGML_CPU_HUGEPAGE=1 LLAMA_MTP_SPEC=1 LLAMA_MTP_TRACE=1 \
  "$TOOL" -m "$MODEL" -ngl 999 -fa 1 --fit off --no-mmap \
    -ctk q8_0 -ctv q8_0 -ot "$OT" -c 4096 -b 512 -ub 512 -t 24 \
    --enable-mtp -n 256 -p "The history of the Roman Empire is a long and complex story that begins with"
echo "rc=$? $(date +%T)"
echo "===== LOSSLESS GATE ====="
NB=$(wc -l < /tmp/mtp_spec_on.txt); NA=$(wc -l < /tmp/mtp_specoff_gen.txt)
CMP=$(( NA<NB ? NA : NB ))
head -n "$CMP" /tmp/mtp_specoff_gen.txt > /tmp/_a.txt
head -n "$CMP" /tmp/mtp_spec_on.txt   > /tmp/_b.txt
if diff -q /tmp/_a.txt /tmp/_b.txt >/dev/null; then echo "GATE PASS: $CMP tokens byte-identical (LOSSLESS)"
else echo "GATE FAIL"; diff /tmp/_a.txt /tmp/_b.txt | head -8; fi
