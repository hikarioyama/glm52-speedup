#!/bin/bash
cd ~/projects/glm52-speedup/harness
for K in 5 7; do
  ./split_measure.sh $K 480 > "out/split_k${K}.console" 2>&1
  grep -aE "RESULT" "out/split_k${K}.console"
done
echo "SPLIT SWEEP DONE"
