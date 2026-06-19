#!/bin/bash
# モデルメタからルーフライン入力を確定 (active params / n_expert / top_k / 層数)
source ~/projects/glm52-speedup/harness/env.sh
set -e
DUMP=$LCPP/llama-gguf
# llama-gguf が無ければ gguf-dump 相当を探す
[ -x "$DUMP" ] || DUMP=$(ls $LCPP/../bin/llama-gguf 2>/dev/null || echo "")

echo "=== GGUF metadata (key hparams) ===" | tee $OUT/00_metadata.txt
if [ -n "$DUMP" ] && [ -x "$DUMP" ]; then
  "$DUMP" "$MODEL" 2>/dev/null | grep -iE 'expert|n_layer|block_count|embedding_length|feed_forward|head_count|n_head|context|rope|indexer|leading_dense|vocab' \
    | tee -a $OUT/00_metadata.txt
else
  echo "llama-gguf 無し → llama-cli 起動ログから読む(10_baseline のログ参照)" | tee -a $OUT/00_metadata.txt
fi
echo "" | tee -a $OUT/00_metadata.txt
echo "ルーフライン: active_GB/token = active_params * bpw(2.6) / 8" | tee -a $OUT/00_metadata.txt
echo "n_expert_used * per_expert_ffn + attn + shared 等から手計算 (起動ログの 'expert used' 参照)" | tee -a $OUT/00_metadata.txt
