#!/bin/bash
# Stable llama.cpp-server configuration for RTX 1650 + 36GB RAM
# Prevents crashes during extended chat sessions
# Last updated: May 2026

MODEL_PATH="./models/gemma4-7b-q4_k_m.gguf"
PORT=8080

echo "=========================================="
echo "llama.cpp-server STABLE CONFIG"
echo "RTX 1650 + 36GB RAM"
echo "=========================================="
echo ""

# Key stability flags explained:
# --ctx-size 2048         : Reduced context (8K is too much for your VRAM/RAM combo with 36GB total)
# --np 1                  : Single parallel slot (prevents context fragmentation)
# --cont-batching         : Continuous batching (handles memory better)
# --batch-size 512        : Moderate batch for stable inference
# --flash-attn on         : Reduces memory usage significantly
# --cache-type-k q8_0     : Quantized KV cache (saves ~50% memory)
# --cache-type-v q8_0     : Quantized KV cache
# --ngl 20                : Offload 20 layers to GPU, rest to RAM (balanced)
# --cram 128              : Host-memory cache for prompt caching
# --mlock                 : Mlock model weights (prevents swapping crashes)
# --timeout 0             : No timeout (prevents crashes from long processing)

./llama-server \
  --model "$MODEL_PATH" \
  --ctx-size 2048 \
  --np 1 \
  --cont-batching \
  --batch-size 512 \
  --ubatch-size 512 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --ngl 20 \
  --cram 128 \
  --mlock \
  --timeout 0 \
  --host 127.0.0.1 \
  --port $PORT \
  --log-disable \
  --log-format json

echo ""
echo "Server running on http://127.0.0.1:$PORT"
echo "Use Ctrl+C to stop"
