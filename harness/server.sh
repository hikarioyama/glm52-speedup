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
# A1 win (2026-06-20, +4% t/s / -9% cmp_cpu, confirmed n=3): offloaded experts default to
# cudaMallocHost pinned-shmem which is slow for CPU streaming reads. In split-OFF the experts
# are CPU-computed (pinning useless), so force plain anonymous (NO_PINNED) + THP (CPU_HUGEPAGE):
#   anon write-back 2 MiB pages read ~9% faster than pinned-shmem. NOTE: do NOT use with
#   LLAMA_MOE_CPU_SPLIT (NO_PINNED kills the split's pinned H2D copy benefit).
export GGML_CUDA_NO_PINNED=1
export GGML_CPU_HUGEPAGE=1
# A3 win (2026-06-20, +3.4% t/s, variance 1.06->0.04): pin the 24 compute threads 1:1 to
# physical cores 0-23 (NOT SMT siblings 24-47) to kill cross-CCX migration jitter. cmp_cpu
# 17.4->16.3ms, fully reproducible. (9965WX: logical 0-23 = phys cores, 24-47 = SMT.)
exec $LCPP/llama-server -m "$MODEL" -ngl 999 -fa 1 --fit off --no-mmap \
  -ctk q8_0 -ctv q8_0 \
  --cpu-mask FFFFFF --cpu-strict 1 \
  -ot "$OT" \
  -c $CTX --host 127.0.0.1 --port $PORT \
  --no-warmup
