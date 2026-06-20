#!/bin/bash
# INC-3 self-spec runner. Sequential (1 model load at a time). Run in background.
#   Step A : spec-OFF baseline — generate S (greedy temp 0) + decode t/s + token stream (S)
#   Step B : spec-ON self-spec — fused K=1 loop + decode t/s + real-loop alpha + token stream
#   Gate   : lossless — diff spec-ON stream vs S[n_prompt:] (must be identical)
set -u
source ~/projects/glm52-speedup/harness/env.sh
OUTD=$OUT
TOOL="$LCPP/llama-mtp-alpha"

if pgrep -af "$LCPP/llama-" | grep -qv pgrep; then
  echo "!!! llama proc already running, abort" >&2; pgrep -af "$LCPP/llama-" | grep -v pgrep >&2; exit 1
fi

# layer-78 experts -> CPU (CUDA1 tight). Identical to INC-2.
OT="blk\.([0-9]|1[0-9]|2[01])\.ffn_(gate|up|down)_exps\.weight=CPU,blk\.(2[2-9]|3[0-9]|4[0-9])\.ffn_(gate|up|down)_exps\.weight=CUDA0,blk\.(5[0-9]|6[0-9]|7[0-7])\.ffn_(gate|up|down)_exps\.weight=CUDA1,blk\.78\.ffn_(gate|up|down)_exps\.weight=CPU"

COMMON_ENV=(GGML_CUDA_NO_PINNED=1 GGML_CPU_HUGEPAGE=1)
CTX=4096
# spec loop only ever submits 1-2 token batches; keep -b/-ub modest. (probe needed N; spec doesn't)
UB=512
N=${N:-256}
PROMPT=${PROMPT:-"The history of the Roman Empire is a long and complex story that begins with"}
SEQ=/tmp/mtp_alpha_seq.txt
SPEC_OUT=/tmp/mtp_spec_on.txt
export LLAMA_MTP_ALPHA_SEQ_FILE=$SEQ
export LLAMA_MTP_SPEC_OUT=$SPEC_OUT

run_one() {  # $1=label $2=extra_env(name=val or empty)
  local label=$1; local extra=$2
  env "${COMMON_ENV[@]}" $extra \
    "$TOOL" -m "$MODEL" -ngl 999 -fa 1 --fit off --no-mmap \
      -ctk q8_0 -ctv q8_0 -ot "$OT" -c "$CTX" -b "$UB" -ub "$UB" -t 24 \
      --enable-mtp -n "$N" -p "$PROMPT"
}

echo "===== STEP A: spec-OFF baseline (generate S + t/s) $(date +%T) ====="
run_one "specoff" "" > "$OUTD/inc3_specoff.log" 2>&1
echo "rc=$?  $(date +%T)"
grep -aE "spec-OFF|t/s|generated tokens|wrote S|EOG" "$OUTD/inc3_specoff.log" | head
NPROMPT=$(grep -aoE "wrote S \([0-9]+ tokens, [0-9]+ prompt" "$OUTD/inc3_specoff.log" | grep -aoE "[0-9]+ prompt" | grep -aoE "[0-9]+")
echo "n_prompt=$NPROMPT  S lines=$(wc -l < $SEQ 2>/dev/null)"
echo

echo "===== STEP B: spec-ON self-spec (fused K=1) $(date +%T) ====="
run_one "specon" "LLAMA_MTP_SPEC=1" > "$OUTD/inc3_specon.log" 2>&1
echo "rc=$?  $(date +%T)"
grep -aiE "error|abort|assert|terminate|cuda error|nan|not captured|placeholder" "$OUTD/inc3_specon.log" | head
sed -n '/MTP self-spec/,/======/p' "$OUTD/inc3_specon.log"
echo

echo "===== GATE: lossless (spec-ON stream vs S[n_prompt:]) $(date +%T) ====="
if [ -z "${NPROMPT:-}" ]; then echo "  cannot determine n_prompt — skip"; else
  tail -n +"$((NPROMPT+1))" "$SEQ" > /tmp/mtp_specoff_gen.txt
  NA=$(wc -l < /tmp/mtp_specoff_gen.txt); NB=$(wc -l < "$SPEC_OUT" 2>/dev/null || echo 0)
  echo "  spec-OFF gen tokens=$NA   spec-ON tokens=$NB"
  # compare the overlapping prefix (spec may stop a token early/late at the boundary)
  CMP=$(( NA < NB ? NA : NB ))
  if [ "$CMP" -lt 1 ]; then echo "  GATE: NO DATA"; else
    head -n "$CMP" /tmp/mtp_specoff_gen.txt > /tmp/_a.txt
    head -n "$CMP" "$SPEC_OUT"              > /tmp/_b.txt
    if diff -q /tmp/_a.txt /tmp/_b.txt >/dev/null; then
      echo "  GATE PASS: first $CMP tokens byte-identical (LOSSLESS)"
    else
      echo "  GATE *** FAIL ***: token streams differ"
      diff /tmp/_a.txt /tmp/_b.txt | head -20
      echo "  (first divergence above; spec loop is BROKEN if this fires)"
    fi
  fi
fi
echo
echo "############ INC-3 DONE $(date +%T) ############"
