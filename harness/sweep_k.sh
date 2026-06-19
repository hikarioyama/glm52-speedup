#!/bin/bash
# GLM-5.2 k-sweep: find the CPU/GPU expert-split balance that maximizes decode t/s.
# k = number of experts computed on CPU (n_gpu = 8-k copied+computed on GPU).
# One model load per k (k is graph-structural, read at build time). ~5 min/k.
#   usage: sweep_k.sh "2 3 4 5"   (default "3 4 5")
set -u
LCPP=/home/hikari/src/llama.cpp/build/bin
MODEL=/mnt/data/models/GLM-5.2-GGUF/UD-Q2_K_XL/GLM-5.2-UD-Q2_K_XL-00001-of-00007.gguf
OUT=/home/hikari/projects/glm52-speedup/harness/out
export CUDA_VISIBLE_DEVICES=0,1
KS=${1:-"3 4 5"}
mkdir -p "$OUT"
OT="blk\.([3-9]|1[0-9]|2[0-4])\.ffn_(gate|up|down)_exps\.weight=CUDA_Host,blk\.(2[5-9]|3[0-9]|4[0-9])\.ffn_(gate|up|down)_exps\.weight=CUDA0,blk\.(5[0-9]|6[0-9]|7[0-8])\.ffn_(gate|up|down)_exps\.weight=CUDA1"
P="The capital of France is Paris. Explain in three sentences why rivers are important to civilization."

if pgrep -af 'llama-completion|llama-server|llama-cli|llama-bench' | grep -qv pgrep; then
  echo "!!! llama process running, abort"; exit 1
fi

echo "===== GLM-5.2 k-sweep (k = experts on CPU) ====="
for K in $KS; do
  log=$OUT/sweep_k${K}.log
  LLAMA_MOE_CPU_SPLIT=$K $LCPP/llama-completion -m "$MODEL" -ngl 999 -fa 1 --fit off --no-mmap \
    -ot "$OT" -c 4096 --no-warmup --ignore-eos --seed 1234 --temp 0 -n 96 -t 24 -p "$P" >"$log" 2>&1
  tps=$(grep -oE "[0-9.]+ tokens per second" "$log" | tail -1 | grep -oE "^[0-9.]+")
  err=$(grep -iE "GGML_ASSERT|out of memory|CUDA error|nan " "$log" | head -1)
  printf "k=%s  (cpu=%s gpu=%s)  decode = %s t/s  %s\n" "$K" "$K" "$((8-K))" "${tps:-FAIL}" "${err:+[ERR: $err]}"
done
echo "===== done ====="
