#!/bin/bash
# Persistent-server sweep: load GLM-5.2 ONCE, then sweep (k,dual) configs via the runtime
# control file /tmp/moe_ctl (read per graph-build) — NO model reload between configs.
# Eliminates the ~4-5 min idle GPU reload between every measurement.
#   usage: server_sweep.sh            (launches server if not up, then sweeps)
#          server_sweep.sh sweeponly  (assumes server already up, just sweeps)
set -u
LCPP=/home/hikari/src/llama.cpp/build/bin
MODEL=/mnt/data/models/GLM-5.2-GGUF/UD-Q2_K_XL/GLM-5.2-UD-Q2_K_XL-00001-of-00007.gguf
OUT=/home/hikari/projects/glm52-speedup/harness/out
PORT=8080
CTL=/tmp/moe_ctl
export CUDA_VISIBLE_DEVICES=0,1
mkdir -p "$OUT"
# CUDA1-headroom config (room for dual-GPU staging on BOTH GPUs):
# host=blk3-24+75-78 (26 offloaded) / CUDA0=25-49 / CUDA1=50-74
OT="blk\.([3-9]|1[0-9]|2[0-4]|7[5-8])\.ffn_(gate|up|down)_exps\.weight=CUDA_Host,blk\.(2[5-9]|3[0-9]|4[0-9])\.ffn_(gate|up|down)_exps\.weight=CUDA0,blk\.(5[0-9]|6[0-9]|7[0-4])\.ffn_(gate|up|down)_exps\.weight=CUDA1"
P="The capital of France is Paris. Explain in three sentences why rivers are important to civilization."

if [ "${1:-}" != "sweeponly" ]; then
  if pgrep -af 'llama-server|llama-completion|llama-cli|llama-bench' | grep -qv pgrep; then
    echo "!!! llama process already running, abort"; pgrep -af 'llama-' | grep -v pgrep; exit 1
  fi
  echo "0 0" > "$CTL"   # start with split OFF
  echo "[server] launching (one load)..."
  LLAMA_MOE_CTL=$CTL nohup $LCPP/llama-server -m "$MODEL" -ngl 999 -fa 1 --fit off --no-mmap \
    -ot "$OT" -c 4096 -t 24 --host 127.0.0.1 --port $PORT --no-warmup > "$OUT/server.log" 2>&1 &
  echo "[server] waiting for ready (model load ~4-5min)..."
  for i in $(seq 1 600); do
    curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status":"ok"' && break
    grep -qiE "error|failed to load|out of memory" "$OUT/server.log" && { echo "[server] LOAD ERROR"; tail -5 "$OUT/server.log"; exit 1; }
    sleep 2
  done
  curl -s "http://127.0.0.1:$PORT/health" | grep -q ok || { echo "[server] not ready, abort"; tail -8 "$OUT/server.log"; exit 1; }
  echo "[server] READY"
fi

measure() { # $1=k $2=dual $3=npred  -> prints predicted_per_second
  local k=$1 d=$2 n=$3
  echo "$k $d" > "$CTL"
  curl -s "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
    -d "{\"prompt\":$(printf '%s' "$P" | jq -R -s .),\"n_predict\":$n,\"temperature\":0,\"seed\":1234,\"cache_prompt\":false}" \
    | jq -r '.timings.predicted_per_second // "FAIL"'
}

echo "===== server sweep (one load, runtime-tuned k/dual) ====="
printf "%-22s %s\n" "config" "decode t/s"
for cfg in "0 0:baseline" "4 0:single_k4" "5 0:single_k5" "6 0:single_k6" "3 1:dual_k3" "4 1:dual_k4" "5 1:dual_k5"; do
  kv="${cfg%%:*}"; name="${cfg##*:}"; k="${kv%% *}"; d="${kv##* }"
  measure "$k" "$d" 16 >/dev/null   # warmup (triggers graph realloc on config change)
  tps=$(measure "$k" "$d" 96)
  printf "%-22s %s\n" "$name (k=$k dual=$d)" "$tps"
done
echo "===== done (server still UP on :$PORT for more sweeps) ====="
