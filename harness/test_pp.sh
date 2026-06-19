#!/bin/bash
# GLM-5.2 split k=4 WITH events (LLAMA_FORCE_PP) — does n_copies>1 unlock overlap?
# more host layers for VRAM headroom (n_copies=4 multiplies staging buffers).
# host=blk3-32(30) / CUDA0=33-54(22) / CUDA1=55-78(24) — extra headroom for n_copies=4 staging
LCPP=/home/hikari/src/llama.cpp/build/bin
MODEL=/mnt/data/models/GLM-5.2-GGUF/UD-Q2_K_XL/GLM-5.2-UD-Q2_K_XL-00001-of-00007.gguf
OUT=/home/hikari/projects/glm52-speedup/harness/out
export CUDA_VISIBLE_DEVICES=0,1
K=${1:-4}
OT="blk\.([3-9]|1[0-9]|2[0-9]|3[0-2])\.ffn_(gate|up|down)_exps\.weight=CUDA_Host,blk\.(3[3-9]|4[0-9]|5[0-4])\.ffn_(gate|up|down)_exps\.weight=CUDA0,blk\.(5[5-9]|6[0-9]|7[0-8])\.ffn_(gate|up|down)_exps\.weight=CUDA1"
P="The capital of France is Paris. Explain in three sentences why rivers are important to civilization."

if pgrep -af 'llama-completion' | grep -qv pgrep; then echo "llama running, abort"; exit 1; fi

run() { # tag, env
  local tag=$1 kv=$2 log=$OUT/pp_${tag}.log
  nvidia-smi dmon -s u -d 1 -o T > "$OUT/pp_dmon_${tag}.log" 2>&1 & local d=$!
  env ${kv:+$kv} $LCPP/llama-completion -m "$MODEL" -ngl 999 -fa 1 --fit off --no-mmap \
    -ot "$OT" -c 4096 --no-warmup --seed 1234 --temp 0 -n 96 -p "$P" > "$log" 2>&1
  kill $d 2>/dev/null
  echo ">>> $tag = $(grep -oE '[0-9.]+ tokens per second' "$log" | tail -1)"
  grep -m1 -iE "pipeline parallelism enabled" "$log" && echo "   PP ENABLED" || echo "   PP off"
  grep -m1 MOE_CPU_SPLIT "$log"
  grep -iE "out of memory|failed to|CUDA error" "$log" | head -2
  echo "   GPU util (steady decode, last 12 samples):"
  tail -25 "$OUT/pp_dmon_${tag}.log" | awk '$2==0||$2==1{print "    gpu"$2" sm="$3"%"}' | tail -12
}

echo "=== load 1/2: baseline-no-split + PP (control: does PP alone change baseline?) ==="
run base_pp "LLAMA_FORCE_PP=1"
echo "=== load 2/2: split k=$K + PP ==="
run split_pp "LLAMA_MOE_CPU_SPLIT=$K LLAMA_FORCE_PP=1"
echo "ANCHOR baseline(no-pp,19-22host) ~= 23.8 t/s"
echo "DONE"
