vllm serve sonlam1102/falcon_explanation_vision --host 0.0.0.0 --port 8089  --load-format bitsandbytes --quantization bitsandbytes --dtype bfloat16 \
  --gpu-memory-utilization 0.3 \
  --enforce-eager \
  --max-num-seqs 8 \
  --kv-cache-dtype fp8 \
  --api-key "falcon_explanation_vision" \
  --max-model-len 8000 