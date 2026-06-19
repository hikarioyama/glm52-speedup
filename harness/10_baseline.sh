#!/bin/bash
# baseline tok/s + VRAM配置確認。配置戦略は段階的に試す。
# 戦略: 非expert(attn/router/shared/norm)を全GPU常駐、溢れるexpert FFNだけCPUへ。
source ~/projects/glm52-speedup/harness/env.sh
NCPUMOE=${1:-0}   # CPUに逃がすMoE層数。0から上げて「VRAMに乗る最小offload」を二分探索
TAG=${2:-ncpumoe$NCPUMOE}
LOG=$OUT/10_baseline_$TAG.log

echo "### placement: --n-cpu-moe $NCPUMOE / -ngl 999 / 2GPU split ###" | tee $LOG
# decode律速を見るので prompt短/gen長。flash-attn と KV量子化も後で振る。
$LCPP/llama-bench \
  -m "$MODEL" \
  -ngl 999 \
  --n-cpu-moe $NCPUMOE \
  -fa 1 \
  -p 64 -n 128 \
  -r 3 \
  -o md 2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "### VRAM split (bench直後) ###" | tee -a $LOG
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | tee -a $LOG
