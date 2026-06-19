#!/bin/bash
# DL完了を待ち、自動で baseline → nsys を回す。ncu/expert-reuse は結果見て手動。
source ~/projects/glm52-speedup/harness/env.sh
echo "[$(date +%H:%M:%S)] waiting for model (7 shards >1G)..."
until model_ready; do sleep 60; done
echo "[$(date +%H:%M:%S)] MODEL READY. size=$(du -sh $MODEL_DIR|cut -f1)"

# 1) メタ確定
bash $H/00_metadata.sh

# 2) 配置 baseline: n-cpu-moe を 0→上げて VRAM に乗る最小offloadを探す
#    254GB>192GB なので 0 は OOM のはず。OOMなら自動で増やす二分探索。
for N in 0 8 16 24 32 40; do
  echo "[$(date +%H:%M:%S)] === try --n-cpu-moe $N ==="
  if bash $H/10_baseline.sh $N 2>&1 | grep -qiE 'out of memory|cudaMalloc|failed to allocate'; then
    echo "  OOM at $N, 上げる"
    continue
  fi
  echo "  OK at --n-cpu-moe $N → これを baseline 採用"
  echo "$N" > $OUT/baseline_ncpumoe.txt
  break
done

# 3) nsys で律速カーネル
N=$(cat $OUT/baseline_ncpumoe.txt 2>/dev/null || echo 32)
echo "[$(date +%H:%M:%S)] nsys profile @ n-cpu-moe=$N"
bash $H/20_nsys.sh $N

echo "[$(date +%H:%M:%S)] DONE. 結果: $OUT/ (baseline log, nsys kernsum, memops)"
