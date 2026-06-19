#!/bin/bash
# GLM-5.2 dual-GPU split test: GPU experts split across CUDA0+CUDA1 (both PCIe links).
# Needs staging headroom on BOTH GPUs => move 4 layers (blk75-78) off CUDA1 to host.
# Compares: baseline(no split) / single-GPU split / dual-GPU split, SAME -ot config.
#   usage: test_dual_gpu.sh [k]   (default k=4; n_gpu=8-k split across the 2 GPUs)
set -u
LCPP=/home/hikari/src/llama.cpp/build/bin
MODEL=/mnt/data/models/GLM-5.2-GGUF/UD-Q2_K_XL/GLM-5.2-UD-Q2_K_XL-00001-of-00007.gguf
OUT=/home/hikari/projects/glm52-speedup/harness/out
export CUDA_VISIBLE_DEVICES=0,1
K=${1:-4}
mkdir -p "$OUT"
# host=blk3-24 + blk75-78 (26 offloaded) / CUDA0=blk25-49 / CUDA1=blk50-74 (headroom for staging)
OT="blk\.([3-9]|1[0-9]|2[0-4]|7[5-8])\.ffn_(gate|up|down)_exps\.weight=CUDA_Host,blk\.(2[5-9]|3[0-9]|4[0-9])\.ffn_(gate|up|down)_exps\.weight=CUDA0,blk\.(5[0-9]|6[0-9]|7[0-4])\.ffn_(gate|up|down)_exps\.weight=CUDA1"
P="The capital of France is Paris. Explain in three sentences why rivers are important to civilization."
NPRED=96

if pgrep -af 'llama-completion|llama-server|llama-cli|llama-bench' | grep -qv pgrep; then
  echo "!!! llama process running, abort"; exit 1
fi

run() {  # $1=tag  $2=extra_env  $3=dmon
  local tag=$1 env_kv=$2 dmon=$3
  local log=$OUT/dual_${tag}.log dlog=$OUT/dual_${tag}.dmon
  echo "######## $tag (env: ${env_kv:-none}) ########"
  local dmpid=""
  [ "$dmon" = 1 ] && { nvidia-smi dmon -s put -d 1 -o T >"$dlog" 2>&1 & dmpid=$!; }
  env ${env_kv:+$env_kv} GGML_SCHED_SYNC_COUNT=3 $LCPP/llama-completion -m "$MODEL" \
    -ngl 999 -fa 1 --fit off --no-mmap -ot "$OT" -c 4096 --no-warmup --ignore-eos \
    --seed 1234 --temp 0 -n $NPRED -t 24 -p "$P" >"$log" 2>&1
  [ -n "$dmpid" ] && kill "$dmpid" 2>/dev/null
  local tps=$(grep -oE "[0-9.]+ tokens per second" "$log" | tail -1 | grep -oE "^[0-9.]+")
  echo ">>> $tag = ${tps:-FAIL} t/s"
  grep -m1 "MOE_CPU_SPLIT" "$log" || true
  grep -m1 "SYNC_TIME] call#64" "$log" || true
  grep -iE "out of memory|CUDA error|GGML_ASSERT|failed to|nan " "$log" | head -2 || true
  awk '/why rivers/{f=1} f{print}' "$log" | sed '/eval time/d' | head -3
  echo ""
}

echo "=== 1/3 baseline (no split) ==="; run baseline "" 0
echo "=== 2/3 single-GPU split k=$K ==="; run single_k${K} "LLAMA_MOE_CPU_SPLIT=$K" 0
echo "=== 3/3 DUAL-GPU split k=$K ==="; run dual_k${K} "LLAMA_MOE_CPU_SPLIT=$K LLAMA_MOE_DUAL_GPU=1" 1

echo "===================== SUMMARY ====================="
for t in baseline single_k${K} dual_k${K}; do
  echo "$t = $(grep -oE '[0-9.]+ tokens per second' $OUT/dual_${t}.log | tail -1)"
done
echo "--- dual decode dmon tail (both GPUs sm%% + PCIe rx, want BOTH links active) ---"
tail -14 $OUT/dual_dual_k${K}.dmon 2>/dev/null
echo "DONE"
