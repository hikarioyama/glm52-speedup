#!/bin/bash
# Diagnostic: run split k=4 with nvidia-smi dmon to observe GPU util + PCIe during decode.
# Goal: is the GPU branch actually offloading (PCIe RX>0, sm util>0) and overlapping with CPU?
LCPP=/home/hikari/src/llama.cpp/build/bin
MODEL=/mnt/data/models/GLM-5.2-GGUF/UD-Q2_K_XL/GLM-5.2-UD-Q2_K_XL-00001-of-00007.gguf
OUT=/home/hikari/projects/glm52-speedup/harness/out
export CUDA_VISIBLE_DEVICES=0,1
K=${1:-4}
EXTRA="${2:-}"
OT="blk\.([3-9]|1[0-9]|2[0-4])\.ffn_(gate|up|down)_exps\.weight=CUDA_Host,blk\.(2[5-9]|3[0-9]|4[0-9])\.ffn_(gate|up|down)_exps\.weight=CUDA0,blk\.(5[0-9]|6[0-9]|7[0-8])\.ffn_(gate|up|down)_exps\.weight=CUDA1"
P="Count slowly and explain each step: one, two, three. Now write a long detailed essay about the history of mathematics, covering ancient Babylon, Greece, the medieval Islamic world, and the European renaissance, with specific names and dates."

if pgrep -af 'llama-completion' | grep -qv pgrep; then echo "llama running, abort"; exit 1; fi

# start dmon in background (util sm/mem, pcie rx/tx, power)
nvidia-smi dmon -s pucm -d 1 -o T > "$OUT/diag_dmon_k${K}.log" 2>&1 &
DMON=$!

echo "### split k=$K  extra=[$EXTRA]  (dmon $DMON) ###"
env ${EXTRA:+$EXTRA} $LCPP/llama-completion -m "$MODEL" -ngl 999 -fa 1 --fit off --no-mmap \
  -ot "$OT" -c 4096 --no-warmup --seed 1234 --temp 0 -n 200 -p "$P" > "$OUT/diag_split_k${K}.log" 2>&1

kill $DMON 2>/dev/null
echo ">>> decode tok/s = $(grep -oE '[0-9.]+ tokens per second' $OUT/diag_split_k${K}.log | tail -1)"
grep -m1 MOE_CPU_SPLIT "$OUT/diag_split_k${K}.log"
echo "### dmon during decode (last 40 samples = steady decode) ###"
# columns: time gpu sm mem rxpci txpci ... ; show GPU0 and GPU1 rows
tail -45 "$OUT/diag_dmon_k${K}.log" | head -45
echo "DONE"
