#!/bin/bash
# D1 de-risk: 現26.15 config の decode 中、GPU が遊休か / PCIe にトラフィックが有るかを直接観測。
# 期待(αが no-op の根拠): decode 中 GPU sm util ~0%, rxpci ~0 MB/s = expert を CPU 計算・GPU 手待ち。
# レバー検証: 律速が CPU RAM read で PCIe 経路が空いている = 両GPU 115GB/s コピーへの置換余地が物理的に空いている。
source ~/projects/glm52-speedup/harness/env.sh
LOG=$OUT/derisk_d1.log
DMON=$OUT/derisk_d1_dmon.log
PROMPT="The capital of France is Paris. Explain in three sentences why rivers are important to civilization."

# 衝突ガード
if pgrep -af 'llama-server|llama-bench|llama-completion|llama-cli' | grep -qv pgrep; then
  echo "!!! llama 稼働中。中止" >&2; pgrep -af 'llama-(server|bench|completion|cli)' | grep -v pgrep >&2; exit 1
fi

OT="blk\.([0-9]|1[0-9]|2[01])\.ffn_(gate|up|down)_exps\.weight=CPU,blk\.(2[2-9]|3[0-9]|4[0-9])\.ffn_(gate|up|down)_exps\.weight=CUDA0,blk\.(5[0-9]|6[0-9]|7[0-9])\.ffn_(gate|up|down)_exps\.weight=CUDA1"

# dmon を時刻付きでバックグラウンド記録(util + PCIe throughput)
nvidia-smi dmon -s put -d 1 -o DT > "$DMON" 2>&1 &
DMON_PID=$!
echo "dmon pid=$DMON_PID"

echo "### D1 de-risk: current 26 config, -n 160 ###" | tee $LOG
# 長めに生成して decode 区間を dmon で十分サンプル
$LCPP/llama-completion -m "$MODEL" -ngl 999 -fa 1 --fit off --no-mmap \
  -ot "$OT" -c 8192 --no-warmup --seed 1234 --temp 0 -n 160 -p "$PROMPT" 2>&1 | tee -a $LOG

kill $DMON_PID 2>/dev/null
echo "" | tee -a $LOG
echo "### perf ###" | tee -a $LOG
grep -E "eval time|tokens per second|graphs reused" $LOG | tee -a $LOG
echo "### dmon は $DMON ###" | tee -a $LOG
echo "DONE_DERISK_D1"
