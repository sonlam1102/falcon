vllm serve sonlam1102/falcon_augmentation_vision --host 0.0.0.0 --port 8086  --load-format bitsandbytes --quantization bitsandbytes --dtype bfloat16 \
  --gpu_memory_utilization 0.4 \
  --enforce-eager \
  --max-num-seqs 8 \
  --kv-cache-dtype fp8 \
  --api-key "falcon_augmentation_vision" \
  --max-model-len 8000 \