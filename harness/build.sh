#!/bin/bash
# llama.cpp 増分ビルド (SM120, gcc-15 host)。変更→再ビルドの最短経路。
set -e
SRC=~/src/llama.cpp
cd "$SRC"
START=$(date +%s)
# CMake は configure 済前提。並列ビルド。失敗時は即 stderr。
cmake --build build -j "$(nproc)" 2>&1 | tail -25
END=$(date +%s)
echo "### build OK in $((END-START))s ###"
ls -la build/bin/llama-completion build/bin/llama-server 2>/dev/null
