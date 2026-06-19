#!/bin/bash
# 1コード変更あたり1モデルロードで「正しさ + 性能」を同時取得する反復テスト。
# 使い方: iter_test.sh <tag> [extra_env...]
#   例: iter_test.sh alpha_overlap GLM_PREFETCH=1
# - 決定論生成(temp0, fixed seed, 固定prompt)で出力テキストを ref と diff(正しさ回帰検知)
# - eval tok/s を抽出して 26.15 アンカーと比較
source ~/projects/glm52-speedup/harness/env.sh
TAG=${1:-iter}; shift || true
LOG=$OUT/iter_${TAG}.log
REF=$OUT/ref_output.txt           # 初回の baseline 出力を正解として保存
PROMPT="The capital of France is Paris. Explain in three sentences why rivers are important to civilization."
NPRED=96

# 衝突ガード(別ロード厳禁)
if pgrep -af 'llama-server|llama-bench|llama-completion|llama-cli' | grep -qv pgrep; then
  echo "!!! llama プロセス稼働中。衝突回避で中止" >&2
  pgrep -af 'llama-server|llama-bench|llama-completion|llama-cli' | grep -v pgrep >&2
  exit 1
fi

# 同じ -ot 配置(現行26t/s構成)を env から継承。追加 env は引数で渡す。
OT="blk\.([0-9]|1[0-9]|2[01])\.ffn_(gate|up|down)_exps\.weight=CPU,blk\.(2[2-9]|3[0-9]|4[0-9])\.ffn_(gate|up|down)_exps\.weight=CUDA0,blk\.(5[0-9]|6[0-9]|7[0-9])\.ffn_(gate|up|down)_exps\.weight=CUDA1"

echo "### iter=$TAG  extra_env: $* ###" | tee $LOG
env "$@" $LCPP/llama-completion -m "$MODEL" -ngl 999 -fa 1 --fit off --no-mmap \
  -ot "$OT" -c 8192 --no-warmup \
  --seed 1234 --temp 0 -n $NPRED -p "$PROMPT" 2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "### perf ###" | tee -a $LOG
grep -E "eval time|prompt eval" $LOG | tee -a $LOG.perf
TPS=$(grep -oE "[0-9.]+ tokens per second" $LOG | tail -1 | grep -oE "^[0-9.]+")
echo ">>> decode tok/s = ${TPS:-?}  (anchor 26.15)" | tee -a $LOG

# 正しさ: 生成本文だけ抽出して ref と比較
awk '/^The capital of France/{f=1} f{print}' $LOG | sed '/eval time/,$d' > $OUT/out_${TAG}.txt
if [ -f "$REF" ]; then
  if diff -q "$REF" "$OUT/out_${TAG}.txt" >/dev/null; then
    echo ">>> 正しさ: ref と一致 ✅" | tee -a $LOG
  else
    echo ">>> 正しさ: ref と差分あり ⚠️ (diff $REF $OUT/out_${TAG}.txt)" | tee -a $LOG
  fi
else
  cp "$OUT/out_${TAG}.txt" "$REF"
  echo ">>> ref 未存在 → この出力を ref として保存" | tee -a $LOG
fi
