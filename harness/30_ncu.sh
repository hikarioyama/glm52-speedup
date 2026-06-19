#!/bin/bash
# Nsight Compute: 律速カーネル(20_nsysで特定)の達成帯域比を測る。
# 目的 = dram throughput が peak の何%か。低ければ A群(MMVQ 128-bit/coalesce/smem)に伸び代。
source ~/projects/glm52-speedup/harness/env.sh
KERNEL_REGEX=${1:-".*mul_mat_vec.*|.*moe.*|.*mmvq.*|.*dequant.*"}   # 20_nsysの結果で絞る
NCPUMOE=${2:-0}
REP=$OUT/30_ncu_ncpumoe$NCPUMOE
PROMPT="Summarize the theory of computation."

# launch を絞って数カーネルだけ計測(decode 1-2 token分)。ncu は重いので -c で回数制限。
$NCU --target-processes all \
  --kernel-name-base demangled \
  --kernel-name "regex:$KERNEL_REGEX" \
  -c 30 \
  --set full \
  --metrics gpu__time_duration.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_bytes.sum,smsp__inst_executed.sum \
  -o $REP --force-overwrite \
  $LCPP/llama-cli -m "$MODEL" -ngl 999 --n-cpu-moe $NCPUMOE -fa 1 \
    -no-cnv -p "$PROMPT" -n 8 2>&1 | tail -15

echo "=== ncu summary (達成帯域比) ==="
$NCU --import $REP.ncu-rep --page raw --csv 2>/dev/null \
  | awk -F, 'NR==1 || /pct_of_peak/' | head -40 | tee $OUT/30_ncu_summary.txt
