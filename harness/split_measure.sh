#!/bin/bash
# Split (CPU||GPU expert split) measurement on the new canonical base (q8_0 KV + A3 pinning).
# Split keeps experts in CUDA_Host (pinned) so NO_PINNED must be OFF; LLAMA_MOE_CPU_SPLIT=k.
#   usage: split_measure.sh <k> [N]
set -u
source ~/projects/glm52-speedup/harness/env.sh
K=${1:-6}; N=${2:-480}
LOG=$OUT/split_k${K}.log; SMAPS=$OUT/split_k${K}.smaps; : > "$SMAPS"
GUARD="$LCPP/llama-(server|bench|completion|cli)"
if pgrep -af "$GUARD" >/dev/null 2>&1; then echo "!!! llama running, abort" >&2; pgrep -af "$GUARD" >&2; exit 1; fi
# split -ot: experts blk3-24 -> CUDA_Host (22 layers, CPU||GPU split), 25-49 -> CUDA0, 50-78 -> CUDA1
OT="blk\.([3-9]|1[0-9]|2[0-4])\.ffn_(gate|up|down)_exps\.weight=CUDA_Host,blk\.(2[5-9]|3[0-9]|4[0-9])\.ffn_(gate|up|down)_exps\.weight=CUDA0,blk\.(5[0-9]|6[0-9]|7[0-8])\.ffn_(gate|up|down)_exps\.weight=CUDA1"
P="The capital of France is Paris. Explain in three sentences why rivers are important to civilization."
echo "=== split k=$K  (q8_0 KV, --cpu-mask pin) N=$N ==="
GGML_SCHED_SYNC_COUNT=3 LLAMA_MOE_CPU_SPLIT=$K $LCPP/llama-completion -m "$MODEL" -ngl 999 -fa 1 --fit off --no-mmap \
  -ctk q8_0 -ctv q8_0 --cpu-mask FFFFFF --cpu-strict 1 \
  -ot "$OT" -c 4096 --no-warmup --ignore-eos --seed 1234 --temp 0 -n "$N" -t 24 -p "$P" >"$LOG" 2>&1 &
PID=$!
while kill -0 "$PID" 2>/dev/null; do
  [ -r "/proc/$PID/smaps_rollup" ] && echo "$(grep -m1 '^Rss:' /proc/$PID/smaps_rollup 2>/dev/null)" >> "$SMAPS"
  sleep 3
done
wait "$PID"; RC=$?
echo "--- exit $RC ---"
ts=$(grep -aE "eval time" "$LOG"|tail -1|grep -oE "[0-9.]+ tokens per second"|grep -oE "^[0-9.]+")
cc=$(grep -a SYNC_TIME "$LOG"|tail -1|grep -oE "CUMUL.*"|grep -oE "cmp_cpu=[0-9]+"|grep -oE "[0-9]+")
err=$(grep -aiE "out of memory|CUDA error|GGML_ASSERT|error loading|failed to load" "$LOG"|head -1)
echo "RESULT split k=$K: t/s=${ts:-FAIL} cmp_cpu=${cc:-?}us ${err:+[ERR: $err]}"
