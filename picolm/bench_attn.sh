#!/bin/bash
# Attention-tile F32 GEMM profiling on Fimbulvetr-11B IQ4_NL.
# --benchmark-ctx grows the context, forcing fresh prefills at each size.
# The d=64 n=128 batch=256 f32_gemm shape (GQA attention tiles) should
# dominate more and more as ctx grows.
M=/kv/Fimbulvetr-11B-v2.iq4_nl.gguf
cd .
PICOLM_PROFILE=1 ./picolm "$M" -c 4096 -n 4 -j 48 --prefault --benchmark-ctx \
    -p "$(head -c 6000 /data4/work/notes/picolm/main.txt | tr '\n' ' ')" 2>&1 | tail -60
