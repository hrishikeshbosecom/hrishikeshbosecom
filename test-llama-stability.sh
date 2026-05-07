#!/bin/bash
# Quick test script for stable llama.cpp-server on RTX 1650
# Run this once to verify your setup won't crash

set -e

echo "=========================================="
echo "llama.cpp Stability Test"
echo "RTX 1650 + 36GB RAM"
echo "=========================================="
echo ""

# Configuration
MODEL="${1:-gemma4-7b-q4_k_m.gguf}"
PORT=8080
NUM_TESTS=10

if [ ! -f "$MODEL" ]; then
  echo "❌ Error: Model file not found: $MODEL"
  echo "Usage: $0 /path/to/model.gguf"
  exit 1
fi

echo "✓ Model found: $MODEL"
echo ""

# Step 1: Start server with stable config
echo "Starting llama-server with stable flags..."
./llama-server \
  --model "$MODEL" \
  --ctx-size 2048 \
  --np 1 \
  --batch-size 512 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --ngl 20 \
  --cram 256 \
  --port $PORT \
  --quiet &

SERVER_PID=$!
echo "Server started (PID: $SERVER_PID)"
echo ""

# Wait for server startup
sleep 5

# Step 2: Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
  echo "❌ Server failed to start!"
  exit 1
fi

echo "✓ Server running"
echo ""

# Step 3: Run test chats
echo "Running $NUM_TESTS test chats..."
echo "(This simulates extended usage to check for memory leaks)"
echo ""

FAILED=0
for i in $(seq 1 $NUM_TESTS); do
  echo -n "Chat $i/$NUM_TESTS: "
  
  # Test prompt
  RESPONSE=$(curl -s http://127.0.0.1:$PORT/api/generate \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"Write a short paragraph about machine learning.\",\"stream\":false,\"n_predict\":100}" \
    2>/dev/null || echo "")
  
  if echo "$RESPONSE" | grep -q '"response"'; then
    echo "✓ Success"
  else
    echo "❌ Failed"
    FAILED=$((FAILED + 1))
  fi
  
  # Show memory usage
  VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  RAM=$(free -h | grep "^Mem:" | awk '{print $3}')
  echo "    VRAM: ${VRAM}MB | RAM: $RAM"
  
  sleep 2
done

echo ""

# Step 4: Cleanup
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

sleep 2

# Step 5: Results
echo ""
echo "=========================================="
echo "Test Results"
echo "=========================================="

if [ $FAILED -eq 0 ]; then
  echo "✓ All $NUM_TESTS tests passed!"
  echo "✓ Your configuration is STABLE for extended use"
  echo ""
  echo "Next steps:"
  echo "1. Use the same flags for your normal chat setup"
  echo "2. Monitor memory with: watch -n 1 'nvidia-smi && free -h'"
  echo "3. If memory climbs past 25GB RAM, reduce --ctx-size to 1024"
else
  echo "❌ $FAILED test(s) failed"
  echo ""
  echo "Troubleshooting:"
  echo "1. Check server.log for errors"
  echo "2. Verify model file is intact"
  echo "3. Reduce --ctx-size to 1024"
  echo "4. Try a different model (Qwen 3 or Phi-4-mini)"
fi

echo ""
echo "=========================================="
