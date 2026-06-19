#!/bin/bash
# GLM-5.2 A/B: baseline (no split) vs CPU∥GPU split WITH the events-at-n_copies=1 fix.
# One run captures: decode tok/s (A/B), real-model SYNC_COUNT structure (confirm war=0),
# and nvidia-smi dmon during the split run (overlap evidence). ~12 min (2 loads).
#   usage: test_glm52_events.sh [k]   (default k=4)
set -u
LCPP=/home/hikari/src/llama.cpp/build/bin
MODEL=/mnt/data/models/GLM-5.2-GGUF/UD-Q2_K_XL/GLM-5.2-UD-Q2_K_XL-00001-of-00007.gguf
OUT=/home/hikari/projects/glm52-speedup/harness/out
export CUDA_VISIBLE_DEVICES=0,1
K=${1:-4}
mkdir -p "$OUT"

# SAME offload config as test_glm52_split.sh:
# host(CUDA_Host pinned)=blk3-24(22) / CUDA0=blk25-49 / CUDA1=blk50-78
OT="blk\.([3-9]|1[0-9]|2[0-4])\.ffn_(gate|up|down)_exps\.weight=CUDA_Host,blk\.(2[5-9]|3[0-9]|4[0-9])\.ffn_(gate|up|down)_exps\.weight=CUDA0,blk\.(5[0-9]|6[0-9]|7[0-8])\.ffn_(gate|up|down)_exps\.weight=CUDA1"
P="The capital of France is Paris. Explain in three sentences why rivers are important to civilization."
NPRED=96

if pgrep -af 'llama-completion|llama-server|llama-cli|llama-bench' | grep -qv pgrep; then
  echo "!!! llama process running, abort"; pgrep -af 'llama-' | grep -v pgrep; exit 1
fi

run() {  # $1=tag  $2=extra_env  $3=dmon(1/0)
  local tag=$1 env_kv=$2 dmon=$3
  local log=$OUT/ev_${tag}.log
  local dlog=$OUT/ev_${tag}.dmon
  echo "######## $tag  (env: ${env_kv:-none}) ########"
  local dmpid=""
  if [ "$dmon" = "1" ]; then
    nvidia-smi dmon -s put -d 1 -o T >"$dlog" 2>&1 &
    dmpid=$!
  fi
  env ${env_kv:+$env_kv} GGML_SCHED_SYNC_COUNT=1 $LCPP/llama-completion -m "$MODEL" \
    -ngl 999 -fa 1 --fit off --no-mmap -ot "$OT" -c 4096 --no-warmup --ignore-eos \
    --seed 1234 --temp 0 -n $NPRED -t 24 -p "$P" >"$log" 2>&1
  [ -n "$dmpid" ] && kill "$dmpid" 2>/dev/null
  local tps=$(grep -oE "[0-9.]+ tokens per second" "$log" | tail -1 | grep -oE "^[0-9.]+")
  echo ">>> $tag decode tok/s = ${tps:-FAIL}"
  grep -m1 "MOE_CPU_SPLIT" "$log" || true
  grep -m1 "CUMUL" "$log" || true
  grep -iE "GGML_ASSERT|abort|out of memory|failed to load|CUDA error|nan " "$log" | head -3 || true
  awk '/why rivers/{f=1} f{print}' "$log" | sed '/eval time/d' | head -4
  echo ""
}

echo "=== load 1/2: BASELINE (no split) ==="
run baseline "" 0
echo "=== load 2/2: SPLIT k=$K + events ==="
run split_k${K} "LLAMA_MOE_CPU_SPLIT=$K" 1

echo "===================== SUMMARY ====================="
B=$(grep -oE '[0-9.]+ tokens per second' $OUT/ev_baseline.log | tail -1 | grep -oE '^[0-9.]+')
S=$(grep -oE '[0-9.]+ tokens per second' $OUT/ev_split_k${K}.log | tail -1 | grep -oE '^[0-9.]+')
echo "baseline   = ${B:-FAIL} tok/s"
echo "split k$K   = ${S:-FAIL} tok/s"
if [ -n "${B:-}" ] && [ -n "${S:-}" ]; then
  awk -v b="$B" -v s="$S" 'BEGIN{printf ">>> speedup = %.3fx  (%.2f -> %.2f t/s)\n", s/b, b, s}'
fi
echo "--- split decode-phase dmon tail (GPU sm%% + PCIe rx/tx, overlap evidence) ---"
tail -16 $OUT/ev_split_k${K}.dmon 2>/dev/null
echo "DONE"
