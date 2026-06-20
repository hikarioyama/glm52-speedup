#!/bin/bash
# INC-1 retest after the nextn LLM_TENSOR_LAYER_REPEATING classification fix.
# Rebuild, then run ONLY the MTP-on arm (off baseline already captured in inc1_off.out, unaffected
# by the arch change since MTP-off takes the SKIP early-return). Compare + report.
set -u
source ~/projects/glm52-speedup/harness/env.sh
OUTD=$OUT
P="Once upon a time, in a distant kingdom beyond the mountains, there lived a young"
N=48

echo "===== BUILD $(date +%T) ====="
source ~/miniforge3/etc/profile.d/conda.sh && conda activate llamacpp-cu131
cmake --build ~/src/llama.cpp/build -j "$(nproc)" --target llama-completion 2>&1 | tail -5
BRC=${PIPESTATUS[0]}
echo "build rc=$BRC"
[ "$BRC" -ne 0 ] && { echo "BUILD FAILED, abort"; exit 1; }

if pgrep -af "$LCPP/llama-" | grep -qv pgrep; then
  echo "!!! llama proc alive, abort" >&2; exit 1
fi

OT="blk\.([0-9]|1[0-9]|2[01])\.ffn_(gate|up|down)_exps\.weight=CPU,blk\.(2[2-9]|3[0-9]|4[0-9])\.ffn_(gate|up|down)_exps\.weight=CUDA0,blk\.(5[0-9]|6[0-9]|7[0-7])\.ffn_(gate|up|down)_exps\.weight=CUDA1,blk\.78\.ffn_(gate|up|down)_exps\.weight=CPU"

echo "===== ARM on (retest) start $(date +%T) ====="
GGML_CUDA_NO_PINNED=1 GGML_CPU_HUGEPAGE=1 \
"$LCPP/llama-completion" -m "$MODEL" -ngl 999 -fa 1 --fit off --no-mmap \
  -ctk q8_0 -ctv q8_0 -ot "$OT" -c 4096 -t 24 \
  --no-warmup --ignore-eos --temp 0 --seed 1234 -n "$N" -p "$P" --enable-mtp \
  > "$OUTD/inc1_on2.out" 2> "$OUTD/inc1_on2.log"
echo "ARM on rc=$? $(date +%T)"

echo
echo "############ INC-1 RETEST RESULTS ############"
echo "== crash/error markers in on2.log =="
grep -aiE "abort|GGML_ABORT|assert|used with a layer|error|oom|out of memory|terminate|nan" "$OUTD/inc1_on2.log" | grep -aivE "n_threads|no error" | head -15 || true
echo "(empty above = no crash)"
echo
echo "== nextn 'unused' WARN (should be 0 on) =="; grep -ac "unused tensor blk.78.nextn" "$OUTD/inc1_on2.log"
echo "== blk78 'unused' WARN (should be 0 on) =="; grep -ac "unused tensor blk.78\." "$OUTD/inc1_on2.log"
echo
echo "== reached generation? (look for prompt echo / eval) =="
grep -aiE "eval time|sampling time|llama_perf|n_eval" "$OUTD/inc1_on2.log" | tail -3
echo
echo "== VRAM after load (from log if present) / model buffer sizes =="
grep -aiE "CUDA0 model buffer|CUDA1 model buffer|CPU model buffer|buffer size|KV self" "$OUTD/inc1_on2.log" | tail -8
echo
echo "== BYTE-IDENTICAL gate: diff inc1_off.out (saved baseline) vs inc1_on2.out =="
if diff -q "$OUTD/inc1_off.out" "$OUTD/inc1_on2.out" >/dev/null 2>&1; then
  echo "   PASS: MTP-on generation byte-identical to MTP-off baseline"
else
  echo "   diff (first 30 lines):"; diff "$OUTD/inc1_off.out" "$OUTD/inc1_on2.out" | head -30
fi
echo
echo "== on2 generation tail =="; tail -c 500 "$OUTD/inc1_on2.out"
echo
echo "############ DONE $(date +%T) ############"
