#!/bin/bash
# 1ロードで thread スイープ → decode が threads でスケールするか(=RAM並列不全, 安い勝ち)
# プラトーなら dequant/compute 律速(=GPUコピー正当)を判定。
source ~/projects/glm52-speedup/harness/env.sh
LOG=$OUT/sweep_threads.log
if pgrep -af 'llama-(server|bench|completion|cli)' | grep -qv pgrep; then
  echo "!!! llama 稼働中。中止" >&2; exit 1; fi
OT="blk\.([0-9]|1[0-9]|2[01])\.ffn_(gate|up|down)_exps\.weight=CPU,blk\.(2[2-9]|3[0-9]|4[0-9])\.ffn_(gate|up|down)_exps\.weight=CUDA0,blk\.(5[0-9]|6[0-9]|7[0-9])\.ffn_(gate|up|down)_exps\.weight=CUDA1"
echo "### thread sweep (decode t/s vs threads), ctx短, 1ロード ###" | tee $LOG
$LCPP/llama-bench -m "$MODEL" -ngl 999 -fa 1 \
  -ot "$OT" \
  --no-mmap 1 \
  -t 12,24,36,48 \
  -p 0 -n 96 \
  -r 2 -o md 2>&1 | tee -a $LOG
echo "DONE_SWEEP" | tee -a $LOG
