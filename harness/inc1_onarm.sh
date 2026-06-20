#!/bin/bash
# INC-1 MTP-on load test (post arch fix). No set -u (conda activate trips it). Build already done.
source ~/projects/glm52-speedup/harness/env.sh
OUTD=$OUT
P="Once upon a time, in a distant kingdom beyond the mountains, there lived a young"
N=48

if pgrep -af "$LCPP/llama-" | grep -qv pgrep; then echo "!!! llama proc alive, abort"; exit 1; fi

OT="blk\.([0-9]|1[0-9]|2[01])\.ffn_(gate|up|down)_exps\.weight=CPU,blk\.(2[2-9]|3[0-9]|4[0-9])\.ffn_(gate|up|down)_exps\.weight=CUDA0,blk\.(5[0-9]|6[0-9]|7[0-7])\.ffn_(gate|up|down)_exps\.weight=CUDA1,blk\.78\.ffn_(gate|up|down)_exps\.weight=CPU"

echo "===== ARM on (post-fix) start $(date +%T) ====="
GGML_CUDA_NO_PINNED=1 GGML_CPU_HUGEPAGE=1 \
"$LCPP/llama-completion" -m "$MODEL" -ngl 999 -fa 1 --fit off --no-mmap \
  -ctk q8_0 -ctv q8_0 -ot "$OT" -c 4096 -t 24 \
  --no-warmup --ignore-eos --temp 0 --seed 1234 -n "$N" -p "$P" --enable-mtp \
  > "$OUTD/inc1_on2.out" 2> "$OUTD/inc1_on2.log"
echo "ARM on rc=$? $(date +%T)"

echo
echo "############ INC-1 RETEST RESULTS ############"
echo "== crash/error markers =="
grep -aiE "abort|GGML_ABORT|assert|used with a layer|error|oom|out of memory|terminate|: nan" "$OUTD/inc1_on2.log" | grep -aivE "n_threads|no error|0 error" | head -15
echo "(empty above = no crash)"
echo "== nextn 'unused' WARN (want 0) =="; grep -ac "unused tensor blk.78.nextn" "$OUTD/inc1_on2.log"
echo "== blk78 'unused' WARN (want 0) =="; grep -ac "unused tensor blk.78\." "$OUTD/inc1_on2.log"
echo "== reached eval? =="; grep -aiE "eval time|llama_perf|tokens per second" "$OUTD/inc1_on2.log" | tail -3
echo "== model buffers / KV =="; grep -aiE "model buffer|CUDA0 buffer|CUDA1 buffer|CPU_Mapped|KV self|llama_kv" "$OUTD/inc1_on2.log" | tail -10
echo "== BYTE-IDENTICAL vs saved off baseline =="
if diff -q "$OUTD/inc1_off.out" "$OUTD/inc1_on2.out" >/dev/null 2>&1; then
  echo "   PASS byte-identical"
else
  echo "   diff:"; diff "$OUTD/inc1_off.out" "$OUTD/inc1_on2.out" | head -25
fi
echo "== on2 gen tail =="; tail -c 500 "$OUTD/inc1_on2.out"
echo "############ DONE $(date +%T) ############"
