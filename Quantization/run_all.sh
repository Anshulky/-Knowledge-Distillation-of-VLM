#!/bin/bash
# The three configs that run on Blackwell (SM 12.0) with vLLM 0.30, one per GPU lane. Then: figures/make_quant_figure.py
cd /backup/Anshul/IE643
R=Data/quant/run_one.sh
( $R bf16 0 8100 ; $R fp8_llm 0 8100 --quantization online \
    --quantization-config '{"linear":"fp8_per_channel","ignore":["*visual*"]}' ) > Data/quant/lane0.log 2>&1 &
( $R fp8_tensor 1 8200 --quantization fp8_per_tensor ) > Data/quant/lane1.log 2>&1 &
wait
echo "ALL DONE $(date '+%F %T')" >> Data/quant/lane0.log
