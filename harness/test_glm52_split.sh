#!/bin/bash
# GLM-5.2 real-model test: baseline (no split) vs CPU∥GPU split, SAME offload config.
# host(CUDA_Host pinned)=blk3-24(22) / CUDA0=blk25-49(25, ~11GB headroom for staging) / CUDA1=blk50-78(29)
# NOTE: llama-completion has rpath to the conda CUDA libs baked in; no conda activate needed.
LCPP=/home/hikari/src/llama.cpp/build/bin
MODEL=/mnt/data/models/GLM-5.2-GGUF/UD-Q2_K_XL/GLM-5.2-UD-Q2_K_XL-00001-of-00007.gguf
OUT=/home/hikari/projects/glm52-speedup/harness/out
export CUDA_VISIBLE_DEVICES=0,1
K=${1:-4}

OT="blk\.([3-9]|1[0-9]|2[0-4])\.ffn_(gate|up|down)_exps\.weight=CUDA_Host,blk\.(2[5-9]|3[0-9]|4[0-9])\.ffn_(gate|up|down)_exps\.weight=CUDA0,blk\.(5[0-9]|6[0-9]|7[0-8])\.ffn_(gate|up|down)_exps\.weight=CUDA1"
P="The capital of France is Paris. Explain in three sentences why rivers are important to civilization."

if pgrep -af 'llama-completion|llama-server|llama-cli|llama-bench' | grep -qv pgrep; then
  echo "!!! llama process running, abort"; pgrep -af 'llama-' | grep -v pgrep; exit 1
fi

run() {  # $1=tag  $2=extra_env(KEY=VAL or empty)
  local tag=$1 env_kv=$2
  local log=$OUT/glm52_${tag}.log
  echo "######## $tag  (env: ${env_kv:-none}) ########"
  env ${env_kv:+$env_kv} $LCPP/llama-completion -m "$MODEL" -ngl 999 -fa 1 --fit off --no-mmap \
    -ot "$OT" -c 4096 --no-warmup --seed 1234 --temp 0 -n 96 -p "$P" >"$log" 2>&1
  local tps=$(grep -oE "[0-9.]+ tokens per second" "$log" | tail -1 | grep -oE "^[0-9.]+")
  echo ">>> $tag decode tok/s = ${tps:-FAIL}"
  grep -m1 "MOE_CPU_SPLIT" "$log" || true
  grep -iE "GGML_ASSERT|abort|out of memory|failed to|CUDA error|nan" "$log" | head -3 || true
  # show generated text head
  awk '/why rivers/{f=1} f{print}' "$log" | sed '/llama_perf|eval time/d' | head -5
  echo ""
}

echo "=== load 1/2: BASELINE (no split) ==="
run baseline ""
echo "=== load 2/2: SPLIT k=$K ==="
run split_k${K} "LLAMA_MOE_CPU_SPLIT=$K"

echo "===================== SUMMARY ====================="
grep -h ">>> " /dev/stdin <<EOF
$(grep -oE "[0-9.]+ tokens per second" $OUT/glm52_baseline.log | tail -1 | sed 's/^/baseline: /')
$(grep -oE "[0-9.]+ tokens per second" $OUT/glm52_split_k${K}.log | tail -1 | sed 's/^/split:    /')
EOF
echo "baseline = $(grep -oE '[0-9.]+ tokens per second' $OUT/glm52_baseline.log | tail -1)"
echo "split k$K = $(grep -oE '[0-9.]+ tokens per second' $OUT/glm52_split_k${K}.log | tail -1)"
echo "DONE"
