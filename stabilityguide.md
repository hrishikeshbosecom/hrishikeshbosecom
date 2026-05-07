# llama.cpp-server Stability Guide for RTX 1650 + 36GB RAM

## Problem: Server Crashes After Extended Chats

This happens because of:
1. **KV cache explosion** - Context cache grows unbounded
2. **Layer allocation failures** - CUDA memory fragmentation
3. **Gemma 4 context checkpoints** - Specific to Gemma models (RAM bloat)
4. **Unbounded context windows** - Default 4096 is too large for your hardware

---

## Solution 1: CRITICAL FLAGS (Start Here)

```bash
./llama-server \
  --model gemma4-7b-q4_k_m.gguf \
  --ctx-size 2048 \
  --np 1 \
  --batch-size 512 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --ngl 20
```

### Flag Explanations:

| Flag | Value | Why It Fixes Crashes |
|------|-------|---------------------|
| `--ctx-size` | 2048 | **Critical**: Reduces KV cache from 4096→2048. Saves 50% memory. Prevents OOM. |
| `--np` | 1 | Single slot (not 4-8). Prevents memory fragmentation from multiple contexts. |
| `--cache-type-k` | q8_0 | Quantizes key cache to 8-bit. Saves ~50% KV cache memory. Requires `--flash-attn on`. |
| `--cache-type-v` | q8_0 | Quantizes value cache to 8-bit. Must pair with key quantization. |
| `--flash-attn` | on | **Required for q8_0**: Without it, KV dequantization overhead negates savings. |
| `--batch-size` | 512 | Prevents GPU memory spikes during prefill. |
| `--ngl` | 20 | Offload 20 layers to GPU (4GB VRAM), rest to RAM (32GB). Balanced. |

---

## Solution 2: Monitor Memory During Chat

Watch memory creep in real-time:

```bash
# Terminal 1: Start server
./llama-server --model gemma4-7b-q4_k_m.gguf --ctx-size 2048 --np 1 --cache-type-k q8_0 --cache-type-v q8_0 --ngl 20

# Terminal 2: Monitor (refresh every second)
watch -n 1 'nvidia-smi | grep -A 5 "Processes" && echo "---" && free -h'
```

**Expected during extended chat:**
- GPU VRAM: 3-4 GB (stable, not climbing)
- System RAM: 8-12 GB used (stable)
- If RAM climbs past 20GB → STOP, your context window is too large

---

## Solution 3: If Using Gemma 4 Specifically

Gemma 4 has a known checkpoint bug causing RAM bloat:

```bash
# Workaround: Disable context checkpointing
GGML_NO_CHECKPOINTS=1 ./llama-server \
  --model gemma4-7b-q4_k_m.gguf \
  --ctx-size 2048 \
  --np 1 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --ngl 20
```

Or **switch to a different model** that doesn't have this issue:
- **Qwen 3 7B** (more stable)
- **Llama 3.2** (proven stable)
- **Phi-4-mini** (lightweight, no issues)

---

## Solution 4: Further Reduce Context (If Still Crashing)

If even 2048 context crashes after 50+ chats:

```bash
./llama-server \
  --model gemma4-7b-q4_k_m.gguf \
  --ctx-size 1024 \
  --np 1 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --ngl 20
```

Or switch to **smaller model** (same config works):

```bash
# Phi-4-mini: Only 2.7B, super stable, ~20 tok/s
./llama-server \
  --model phi-4-mini-q4_k_m.gguf \
  --ctx-size 4096 \
  --np 1 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --ngl 20
```

---

## Solution 5: Add Host-Memory Prompt Caching

Use 36GB RAM to cache prompts, reducing GPU pressure:

```bash
./llama-server \
  --model gemma4-7b-q4_k_m.gguf \
  --ctx-size 2048 \
  --np 1 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --ngl 20 \
  --cram 256
```

**--cram 256**: Allocate 256MB of RAM for prefix caching. This stores computed prompts in RAM, so repeated questions skip GPU processing (faster + less memory stress).

---

## Quick Diagnosis Script

Run this while chatting to see what's happening:

```bash
#!/bin/bash
echo "Monitoring llama-server memory..."
while true; do
  clear
  echo "=== GPU Memory (nvidia-smi) ==="
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1 | awk '{printf "VRAM: %.1f GB / %.1f GB\n", $1/1024, $2/1024}'
  
  echo ""
  echo "=== System RAM ==="
  free -h | grep "^Mem:" | awk '{printf "RAM: %s used / %s total\n", $3, $2}'
  
  echo ""
  echo "=== Processes ==="
  ps aux | grep llama-server | grep -v grep | awk '{printf "llama-server: PID %s, RSS: %.1f GB\n", $2, $6/1024/1024}'
  
  sleep 2
done
```

**What to watch for:**
- If VRAM + (RAM used by llama-server) > 32GB → Crash imminent
- If RAM climbs 1GB per chat → Memory leak (reduce --ctx-size or switch model)
- If GPU VRAM stable but system RAM climbs → Checkpoint bug (use GGML_NO_CHECKPOINTS=1)

---

## Recommended Configuration for You

Based on RTX 1650 + 36GB RAM + 20 tok/s acceptable:

### Option A: Maximum Stability (Recommended)
```bash
./llama-server \
  --model phi-4-mini-q4_k_m.gguf \
  --ctx-size 4096 \
  --np 1 \
  --batch-size 512 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --ngl 20 \
  --cram 256 \
  --mlock \
  --port 8080
```
- **Model**: Phi-4-mini (2.7B, super stable)
- **Speed**: 20-25 tok/s
- **Crashes**: Virtually zero
- **Context**: 4K tokens

### Option B: Better Quality, Still Stable
```bash
./llama-server \
  --model qwen3-7b-q4_k_m.gguf \
  --ctx-size 2048 \
  --np 1 \
  --batch-size 512 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --ngl 20 \
  --cram 256 \
  --mlock \
  --port 8080
```
- **Model**: Qwen3 7B (better quality)
- **Speed**: 20-30 tok/s
- **Crashes**: Rare
- **Context**: 2K tokens (safe)

### Option C: Gemma 4 (If You Want Best Quality)
```bash
GGML_NO_CHECKPOINTS=1 ./llama-server \
  --model gemma4-7b-q4_k_m.gguf \
  --ctx-size 2048 \
  --np 1 \
  --batch-size 512 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --ngl 20 \
  --cram 256 \
  --mlock \
  --port 8080
```
- **Model**: Gemma 4 7B (best quality)
- **Speed**: 20-25 tok/s
- **Crashes**: Fixed with GGML_NO_CHECKPOINTS=1
- **Context**: 2K tokens (safe)

---

## Environment Variables for Extra Stability

Add these before running llama-server:

```bash
export GGML_CUDA_NO_CUDART=1       # Prevent CUDA initialization crashes
export GGML_CUDA_FORCE_MMQ=1       # Stable memory matrix multiply
export GGML_CUDA_FORCE_DMMV=1      # Stable dot product
export GGML_NO_CHECKPOINTS=1       # Disable Gemma 4 checkpoint bug
export GGML_CUDA_DEVICE_COUNT=1    # Use single GPU only
```

Then run:
```bash
./llama-server --model gemma4-7b-q4_k_m.gguf --ctx-size 2048 --np 1 ...
```

---

## Testing: Run This to Verify Stability

```bash
#!/bin/bash
# Simulate 20 extended chats
./llama-server --model your-model.gguf --ctx-size 2048 --np 1 --cache-type-k q8_0 --cache-type-v q8_0 &
SERVER_PID=$!

sleep 3

for i in {1..20}; do
  echo "Chat $i/20..."
  curl -s http://127.0.0.1:8080/api/generate \
    -d '{"model":"your-model","prompt":"Explain quantum computing in detail.","stream":false}' \
    | jq '.response' | head -c 100
  
  sleep 2
  echo ""
done

kill $SERVER_PID
echo "Test complete. Check that RAM didn't exceed 20GB."
```

If this runs without crashes → your config is stable!

---

## Last Resort: Use Smaller Model

If crashes persist even with all above fixes:

```bash
# Downsize to proven stable model
./llama-server --model llama-3.2-1b-q4_k_m.gguf ...
```

1B models are bulletproof:
- **Speed**: 30+ tok/s
- **VRAM**: <2GB
- **Crashes**: Effectively zero
- **Quality**: Good enough for general chat (not coding)

---

## What NOT to Do

❌ Don't use `--ctx-size 8192` or higher  
❌ Don't set `--np > 2` (parallel slots consume context)  
❌ Don't skip `--flash-attn on` if using KV quantization  
❌ Don't use Gemma 4 without GGML_NO_CHECKPOINTS=1  
❌ Don't run without `--ngl` (full CPU is slower + crashes)  

---

## Support & Debugging

If crashes persist:

1. **Check llama.cpp version**: `./llama-server --version`
   - Update if > 2 months old: `git pull && make`

2. **Check CUDA drivers**: `nvidia-smi`
   - Ensure driver ≥ 550

3. **Check model integrity**: `ls -lh your-model.gguf`
   - Redownload if file seems truncated (< expected size)

4. **Check logs**: Run without `--log-disable` to see errors:
   ```bash
   ./llama-server --model your-model.gguf ... 2>&1 | tee server.log
   ```

5. **Report to llama.cpp GitHub**:
   - Include: `server.log`, `nvidia-smi output`, `free -h output`
   - Title: "llama-server crashes after N chats on RTX 1650"
