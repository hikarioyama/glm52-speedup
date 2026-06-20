#!/bin/bash
# INC-2 α-probe gate runner. Sequential (1 model load at a time). Run in background.
#   Gate 3 : byte-identical probe-off (graph refactor is behavior-preserving)
#   Gate 1 : generate S (greedy, temp 0) with the new tool  + probe forward (no NaN/crash)
#   Gate 4 : report α
# Gate 2 (offline numpy math cross-check) is a separate script (inc2_mathcheck.*).
set -u
source ~/projects/glm52-speedup/harness/env.sh
OUTD=$OUT
TOOL="$LCPP/llama-mtp-alpha"

# guard: refuse if a llama proc is alive
if pgrep -af "$LCPP/llama-" | grep -qv pgrep; then
  echo "!!! llama proc already running, abort" >&2; pgrep -af "$LCPP/llama-" | grep -v pgrep >&2; exit 1
fi

# layer-78 experts -> CPU (CUDA1 tight); 78 excluded from CUDA1 range (no overlap). Same as INC-1.
OT="blk\.([0-9]|1[0-9]|2[01])\.ffn_(gate|up|down)_exps\.weight=CPU,blk\.(2[2-9]|3[0-9]|4[0-9])\.ffn_(gate|up|down)_exps\.weight=CUDA0,blk\.(5[0-9]|6[0-9]|7[0-7])\.ffn_(gate|up|down)_exps\.weight=CUDA1,blk\.78\.ffn_(gate|up|down)_exps\.weight=CPU"

COMMON_ENV=(GGML_CUDA_NO_PINNED=1 GGML_CPU_HUGEPAGE=1)
# big enough context for prompt + N=256 generation/prefill.
# probe/dump need a SINGLE ubatch over all N tokens (so the shift-by-one emb and the all-N
# t_logits readback are correct) -> pin -b/-ub >= N.
CTX=4096
UB=4096
N=256
SEQ=/tmp/mtp_alpha_seq.txt
export LLAMA_MTP_ALPHA_SEQ_FILE=$SEQ

# ----------------------------------------------------------------------------
# Gate 3: byte-identical probe-off. Re-run the EXACT INC-1 off arm config with
# the rebuilt (refactored) binary; diff against the saved byte-identical baseline.
# ----------------------------------------------------------------------------
echo "===== GATE 3: byte-identical probe-off (graph refactor) $(date +%T) ====="
P3="Once upon a time, in a distant kingdom beyond the mountains, there lived a young"
env "${COMMON_ENV[@]}" \
  "$LCPP/llama-completion" -m "$MODEL" -ngl 999 -fa 1 --fit off --no-mmap \
    -ctk q8_0 -ctv q8_0 -ot "$OT" -c 4096 -t 24 \
    --no-warmup --ignore-eos --temp 0 --seed 1234 -n 48 -p "$P3" --enable-mtp \
    > "$OUTD/inc2_g3_on.out" 2> "$OUTD/inc2_g3_on.log"
echo "g3 rc=$?"
echo "-- diff inc2_g3_on.out vs inc1_off.out (empty = PASS) --"
if diff -q "$OUTD/inc2_g3_on.out" "$OUTD/inc1_off.out" >/dev/null; then
  echo "   GATE3 PASS: probe-off generation byte-identical to INC-1 baseline"
else
  echo "   GATE3 *** MISMATCH ***"; diff "$OUTD/inc2_g3_on.out" "$OUTD/inc1_off.out" | head -40
fi
echo

# ----------------------------------------------------------------------------
# Gate 1a: GENERATE S (probe env UNSET). On-distribution greedy temp-0 sequence.
# ----------------------------------------------------------------------------
echo "===== GATE 1a: generate S (probe OFF) $(date +%T) ====="
P="The history of the Roman Empire is a long and complex story that begins with"
env "${COMMON_ENV[@]}" \
  "$TOOL" -m "$MODEL" -ngl 999 -fa 1 --fit off --no-mmap \
    -ctk q8_0 -ctv q8_0 -ot "$OT" -c "$CTX" -b "$UB" -ub "$UB" -t 24 \
    --enable-mtp -n "$N" -p "$P" \
    > "$OUTD/inc2_gen.log" 2>&1
echo "gen rc=$?  $(date +%T)"
grep -aE "wrote S|S text|EOG|error|abort|assert|CUDA error" "$OUTD/inc2_gen.log" | head -20
echo "   S file: $(wc -l < $SEQ 2>/dev/null) tokens"
echo

# ----------------------------------------------------------------------------
# Gate 1b + 4: PROBE forward (LLAMA_MTP_PROBE=1, dedicated process) → α
# ----------------------------------------------------------------------------
echo "===== GATE 1b/4: probe forward (LLAMA_MTP_PROBE=1) → alpha $(date +%T) ====="
env "${COMMON_ENV[@]}" LLAMA_MTP_PROBE=1 \
  "$TOOL" -m "$MODEL" -ngl 999 -fa 1 --fit off --no-mmap \
    -ctk q8_0 -ctv q8_0 -ot "$OT" -c "$CTX" -b "$UB" -ub "$UB" -t 24 \
    --enable-mtp -n "$N" -p "$P" \
    > "$OUTD/inc2_probe.log" 2>&1
echo "probe rc=$?  $(date +%T)"
echo "-- error scan --"
grep -aiE "error|abort|assert|terminate|cuda error|nan|map_layer_ids|out of range" "$OUTD/inc2_probe.log" | grep -aivE "no error" | head -20 || echo "(clean)"
echo "-- alpha result --"
sed -n '/MTP nextn .-probe/,/======/p' "$OUTD/inc2_probe.log"
echo

# ----------------------------------------------------------------------------
# Gate 2: offline math cross-check. Dump named nextn tensors for ONE column,
# then recompute in numpy from the GGUF weights and compare.
# ----------------------------------------------------------------------------
echo "===== GATE 2: dump + offline math cross-check $(date +%T) ====="
rm -f /tmp/mtp_dump_*.bin
env "${COMMON_ENV[@]}" LLAMA_MTP_PROBE=1 LLAMA_MTP_DUMP=1 LLAMA_MTP_DUMP_COL=32 \
  "$TOOL" -m "$MODEL" -ngl 999 -fa 1 --fit off --no-mmap \
    -ctk q8_0 -ctv q8_0 -ot "$OT" -c "$CTX" -b "$UB" -ub "$UB" -t 24 \
    --enable-mtp -n "$N" -p "$P" \
    > "$OUTD/inc2_dump.log" 2>&1
echo "dump rc=$?  $(date +%T)"
grep -aE "dump:" "$OUTD/inc2_dump.log" | head
echo "-- numpy cross-check --"
source ~/miniforge3/etc/profile.d/conda.sh && conda activate llamacpp-cu131
python ~/projects/glm52-speedup/harness/inc2_mathcheck.py 2>&1 | tee "$OUTD/inc2_mathcheck.log"
echo
echo "############ INC-2 DONE $(date +%T) ############"
