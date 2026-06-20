#!/bin/bash
# INC-1 load gate: verify --enable-mtp loads layer-78 + nextn tensors non-null,
# no crash/OOM, VRAM fits, AND base greedy decode is byte-identical MTP-off vs MTP-on.
# Sequential (1 model load at a time). Run in background; check out/inc1_*.
set -u
source ~/projects/glm52-speedup/harness/env.sh
OUTD=$OUT
P="Once upon a time, in a distant kingdom beyond the mountains, there lived a young"
N=48

# guard: refuse if a llama proc is alive
if pgrep -af "$LCPP/llama-" | grep -qv pgrep; then
  echo "!!! llama proc already running, abort" >&2; pgrep -af "$LCPP/llama-" | grep -v pgrep >&2; exit 1
fi

# layer-78 experts -> CPU (CUDA1 tight); 78 excluded from CUDA1 range so order-independent (no overlap).
OT="blk\.([0-9]|1[0-9]|2[01])\.ffn_(gate|up|down)_exps\.weight=CPU,blk\.(2[2-9]|3[0-9]|4[0-9])\.ffn_(gate|up|down)_exps\.weight=CUDA0,blk\.(5[0-9]|6[0-9]|7[0-7])\.ffn_(gate|up|down)_exps\.weight=CUDA1,blk\.78\.ffn_(gate|up|down)_exps\.weight=CPU"

run_arm () {
  local tag="$1"; shift
  echo "===== ARM $tag start $(date +%T) ====="
  GGML_CUDA_NO_PINNED=1 GGML_CPU_HUGEPAGE=1 \
  "$LCPP/llama-completion" -m "$MODEL" -ngl 999 -fa 1 --fit off --no-mmap \
    -ctk q8_0 -ctv q8_0 -ot "$OT" -c 4096 -t 24 \
    --no-warmup --ignore-eos --temp 0 --seed 1234 -n "$N" -p "$P" \
    "$@" \
    > "$OUTD/inc1_${tag}.out" 2> "$OUTD/inc1_${tag}.log"
  local rc=$?
  echo "ARM $tag rc=$rc $(date +%T)"
  echo "--- VRAM right after (process already exited; informational baseline) ---"
  return $rc
}

# Arm A: MTP off (rebuilt binary, base behavior)
run_arm off
# Arm B: MTP on
run_arm on --enable-mtp

echo
echo "############ INC-1 GATE RESULTS ############"
echo "== rc/errors =="
grep -aiE "error|abort|oom|out of memory|failed|assert|terminate|cuda" "$OUTD/inc1_off.log" "$OUTD/inc1_on.log" | grep -aivE "no error|error_|0 error" | head -20 || echo "(no error lines)"
echo
echo "== nextn tensors: 'unused' WARN should be PRESENT in off, ABSENT in on =="
echo "-- off:"; grep -ac "unused tensor blk.78.nextn" "$OUTD/inc1_off.log" | sed 's/^/   nextn-unused-warn count = /'
echo "-- on: "; grep -ac "unused tensor blk.78.nextn" "$OUTD/inc1_on.log"  | sed 's/^/   nextn-unused-warn count = /'
echo "   (off should be 4, on should be 0)"
echo
echo "== layer-78 attn/ffn 'unused' WARN (off PRESENT, on ABSENT) =="
echo "-- off:"; grep -ac "unused tensor blk.78\." "$OUTD/inc1_off.log" | sed 's/^/   blk78-unused count = /'
echo "-- on: "; grep -ac "unused tensor blk.78\." "$OUTD/inc1_on.log"  | sed 's/^/   blk78-unused count = /'
echo
echo "== load success markers =="
grep -aiE "llama_init|model loaded|main: |sampling|generate:" "$OUTD/inc1_on.log" | tail -4
echo
echo "== BYTE-IDENTICAL gate: diff off.out vs on.out (empty = PASS) =="
if diff -q "$OUTD/inc1_off.out" "$OUTD/inc1_on.out" >/dev/null; then
  echo "   PASS: generations byte-identical"
else
  echo "   *** MISMATCH ***"; diff "$OUTD/inc1_off.out" "$OUTD/inc1_on.out" | head -30
fi
echo
echo "== sample generation (on) tail =="
tail -c 400 "$OUTD/inc1_on.out"
echo
echo "############ DONE $(date +%T) ############"
