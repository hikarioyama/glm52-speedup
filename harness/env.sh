#!/bin/bash
# GLM-5.2 speedup harness — shared env. source this.
export LC_ALL=C
H=~/projects/glm52-speedup/harness
OUT=$H/out
LCPP=~/src/llama.cpp/build/bin
MODEL_DIR=/mnt/data/models/GLM-5.2-GGUF/UD-Q2_K_XL
MODEL=$MODEL_DIR/GLM-5.2-UD-Q2_K_XL-00001-of-00007.gguf   # llama.cpp が残り6分割を自動ロード
NSYS=/usr/local/cuda-13.0/bin/nsys
NCU=~/miniforge3/envs/llamacpp-cu131/nsight-compute-2025.4.1/ncu

# 異種GPU(0=full WS / 1=Max-Q)。P2P は iommu=pt 前提。
export CUDA_VISIBLE_DEVICES=0,1
# llama.cpp ランタイムは host compiler 不要(ビルド時のみ gcc-15)。

mkdir -p $OUT
model_ready() {  # shard1はメタのみ(9MB)が正常。大shard6個+合計>230GBで完成判定
  [ "$(find $MODEL_DIR -name '*.gguf' -size +1G 2>/dev/null | wc -l)" -ge 6 ]
}
