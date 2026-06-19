#!/bin/bash
# GLM-5.2 を1回ロードして常駐させる。以後の実験は :8080 API へ → reload地獄回避。
# 起動前に衝突チェック必須(別エージェント並走対策)。
source ~/projects/glm52-speedup/harness/env.sh
PORT=8080
# 衝突ガード
if pgrep -af 'llama-server|llama-bench|llama-completion|llama-cli' | grep -qv pgrep; then
  echo "!!! 既に llama プロセスが居る。衝突回避のため起動中止。pgrep で確認せよ" >&2
  pgrep -af 'llama-server|llama-bench|llama-completion|llama-cli' | grep -v pgrep >&2
  exit 1
fi
# 配置: CPU=blk0-21(MoE19層) / CUDA0=22-49(28) / CUDA1=50-78(28) — 実証済(94/92GB, 26tok/s)
# -ot は3個別々に渡す(セミコロン連結はパース不可)
CTX=${1:-8192}
# -ot は comma 区切り単一引数 (複数フラグは最後しか効かない/セミコロンは不可)
OT="blk\.([0-9]|1[0-9]|2[01])\.ffn_(gate|up|down)_exps\.weight=CPU,blk\.(2[2-9]|3[0-9]|4[0-9])\.ffn_(gate|up|down)_exps\.weight=CUDA0,blk\.(5[0-9]|6[0-9]|7[0-9])\.ffn_(gate|up|down)_exps\.weight=CUDA1"
exec $LCPP/llama-server -m "$MODEL" -ngl 999 -fa 1 --fit off --no-mmap \
  -ot "$OT" \
  -c $CTX --host 127.0.0.1 --port $PORT \
  --no-warmup
