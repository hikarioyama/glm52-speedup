#!/bin/bash
# A1-validate: measure cmp_cpu + AnonHugePages of the offloaded-expert CPU buffers
# under whatever THP mode is currently active. Run once per THP arm (always / never).
# Canonical config = 19-offload, split OFF (matches the ~25 t/s baseline profile).
#   usage: a1_thp_measure.sh <label>   e.g. a1_thp_measure.sh always
set -u
source ~/projects/glm52-speedup/harness/env.sh
LABEL=${1:-run}
N=${2:-480}
LOG=$OUT/a1_thp_${LABEL}.log
SMAPS=$OUT/a1_thp_${LABEL}.smaps
: > "$SMAPS"

# match the actual llama BINARY path (not wrapper bash cmdlines that merely contain the string)
GUARD="$LCPP/llama-(server|bench|completion|cli)"
if pgrep -af "$GUARD" >/dev/null 2>&1; then
  echo "!!! llama process already running, abort" >&2
  pgrep -af "$GUARD" >&2
  exit 1
fi

# 19-offload, split OFF: blk0-21 -> CPU (MoE layers 3-21 = 19), 22-49 -> CUDA0, 50-78 -> CUDA1
OT="blk\.([0-9]|1[0-9]|2[01])\.ffn_(gate|up|down)_exps\.weight=CPU,blk\.(2[2-9]|3[0-9]|4[0-9])\.ffn_(gate|up|down)_exps\.weight=CUDA0,blk\.(5[0-9]|6[0-9]|7[0-9])\.ffn_(gate|up|down)_exps\.weight=CUDA1"
P="The capital of France is Paris. Explain in three sentences why rivers are important to civilization."

echo "=== A1 THP measure: label=$LABEL  N=$N ==="
echo "--- THP state at start ---"
cat /sys/kernel/mm/transparent_hugepage/enabled
grep -iE "AnonHugePages|HugePages_Total" /proc/meminfo

# canonical config now uses q8_0 KV cache (user decision 2026-06-20)
KVTYPE=${KVTYPE:-q8_0}
GGML_SCHED_SYNC_COUNT=3 $LCPP/llama-completion -m "$MODEL" -ngl 999 -fa 1 --fit off --no-mmap \
  -ctk "$KVTYPE" -ctv "$KVTYPE" \
  -ot "$OT" -c 4096 --no-warmup --ignore-eos --seed 1234 --temp 0 -n "$N" -t 24 -p "$P" \
  ${EXTRA_ARGS:-} \
  >"$LOG" 2>&1 &
PID=$!
echo "llama-completion PID=$PID, sampling smaps until exit..."

# sample the process AnonHugePages every 3s while it lives (load ~2.5min then decode)
while kill -0 "$PID" 2>/dev/null; do
  if [ -r "/proc/$PID/smaps_rollup" ]; then
    ts=$(awk 'BEGIN{print systime()}' 2>/dev/null)
    rss=$(grep -m1 '^Rss:' /proc/$PID/smaps_rollup 2>/dev/null)
    anonhp=$(grep -m1 '^AnonHugePages:' /proc/$PID/smaps_rollup 2>/dev/null)
    shmemhp=$(grep -m1 '^ShmemPmdMapped:' /proc/$PID/smaps_rollup 2>/dev/null)
    echo "t=$ts $rss | $anonhp | $shmemhp" >> "$SMAPS"
  fi
  sleep 3
done
wait "$PID"
RC=$?

echo "--- exit code $RC ---"
echo "=== RESULT ($LABEL) ==="
echo "t/s:"; grep -aE "tokens per second|eval time" "$LOG" | tail -4
echo "last [SYNC_TIME] (cmp_cpu = CUMUL/call us):"; grep -a "SYNC_TIME" "$LOG" | tail -2
echo "peak AnonHugePages of process (kB):"; grep -oE "AnonHugePages:[[:space:]]+[0-9]+" "$SMAPS" | grep -oE "[0-9]+" | sort -n | tail -1
echo "peak ShmemPmdMapped of process (kB):"; grep -oE "ShmemPmdMapped:[[:space:]]+[0-9]+" "$SMAPS" | grep -oE "[0-9]+" | sort -n | tail -1
echo "peak Rss of process (kB):"; grep -oE "Rss:[[:space:]]+[0-9]+" "$SMAPS" | grep -oE "[0-9]+" | sort -n | tail -1
echo "=== done ($LABEL) ==="
