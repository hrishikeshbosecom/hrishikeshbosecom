# GPU, CUDA, Tensor cores, and the wall

Notes from the conversation. One pass, not repeated filler.

## 1. Apple GPU vs NVIDIA GPU

They are not the same kind of machine.

NVIDIA discrete GPUs (RTX 40/50, H100, B200) are separate accelerators. They have their own VRAM (GDDR or HBM), a CUDA software stack, Tensor cores, and a display controller on cards that have video outputs. The CPU copies data over PCIe (or the driver migrates pages). Training and inference libraries assume this model.

Apple Silicon GPUs (M-series, including M5 / M5 Ultra / M6) live on the same SoC as the CPU. Memory is unified: one DRAM pool, no separate GDDR brick. The programming model is Metal, plus MLX and Core ML. From M5, each GPU core also has a Neural Accelerator. Power is low. Capacity is high (128 GB on M5 Max, up to 512 GB on M5 Ultra).

Intel splits the same way. An iGPU uses shared system RAM. An Arc add-in card has its own GDDR and behaves like NVIDIA.

Use-case split:

- Gaming, CUDA training, peak tokens on models that fit in 24–32 GB: NVIDIA.
- Large local models, silence, battery, one box: Apple.
- Small models that fit in VRAM: NVIDIA is usually faster per token because of bandwidth and Tensor cores.
- 70B+ that will not fit in 32 GB: Apple (or a multi-GPU / high-memory NVIDIA box).

RTX 40-series street prices in late September 2026 were well above original MSRP on the high end (4090 often thousands of dollars new; 4070-class still hundreds to low thousands). India was higher and thinner on stock.

## 2. CUDA does not interchange

A library built only on CUDA will not run on a Mac GPU or a TPU.

- Mac path: Metal, MLX, llama.cpp Metal, PyTorch MPS, Core ML. Ollama on Apple Silicon uses MLX by default in recent versions.
- TPU path: XLA. Native language is JAX. PyTorch gets there through PyTorch/XLA, torchax, or vLLM TPU, which lowers models to JAX/XLA. Custom `.cu` kernels do not run.
- Weights can move (safetensors, GGUF, MLX). The engine cannot.

There can be a performance gap when leaving a tuned CUDA stack. It is not automatic. A mature TPU stack can beat a GPU on large-batch training or long-context inference. A naive bridge is often slower. Missing fused kernels (FlashAttention-class CUDA) are the usual loss.

Pandas itself is CPU. cuDF / cuPy are CUDA-only and stay on NVIDIA.

## 3. What a multiply actually does

Example: A is 70×1024, B is 1024×256, C = AB.

Each Cij is a dot product of length 1024. Work is about 37 million FLOPs (multiplies plus adds).

Discrete NVIDIA path:

1. CPU has A and B in RAM.
2. DMA copies them into VRAM (PCIe). This is a real copy between two chips, not a pointer swing.
3. CPU launches a kernel.
4. GPU writes C into VRAM.
5. CPU copies C back only if it needs it.

Inference of an LLM uploads weights once. Later GEMMs stay in VRAM. Prompt tokens and sampled ids are tiny copies. Training streams batches; prefetch usually hides that. The expensive part of training is compute, GPU memory, and GPU–GPU traffic, not the host copy.

Apple / Intel iGPU: that bulk copy is skipped. Same DRAM, different virtual addresses, coherent. The GPU still does the multiply.

Display is a different pipeline. The display controller scans a framebuffer. A GEMM does not call it. If C happens to be pixel colors in the right layout, you can present that buffer. On NVIDIA the framebuffer is usually VRAM. On Apple and Intel iGPU it is usually the shared pool. Intel Arc cards are back to separate VRAM.

## 4. CUDA cores

A CUDA core is an ALU. One instruction, one scalar result: add, multiply, or fused multiply-add.

FMA is one unit on one core: a*b+c in one pipe, counted as 2 FLOPs. It is not “core 0 multiplies, core 1 adds.”

Two CUDA cores do two independent FMAs at once. They do not cooperate on a single 1+1.

Each thread is sequential, CPU-like, inside itself. The GPU difference is how many of those threads run together.

Warps: NVIDIA issues threads in gangs of 32. That number is a hardware constant (SIMT width), not a batch-size cap and not a universal speed limit. AMD wavefronts are 32 or 64. Apple SIMD-groups are 32. A partial warp wastes lanes.

320 cores and 640 rows: launch 640 threads. At most ~320 scalar FMAs per clock. The rest wait. That turn-taking is a wave, not an ML batch. Batch size is chosen by you and limited by memory. You can batch far above core count.

1000 houses with 16 features is not one clock even with 1000 cores. Each house is 16 FMAs.

## 5. Tensor cores

A Tensor core does a small matrix stamp:

Y_tile = M_tile * X_tile + C_tile

A typical 16×16×16 MMA is 8192 FLOPs in one instruction, not 16 scalar FMAs. One Tensor core is not “16 CUDA cores.” It is closer to hundreds of CUDA-core-clocks for that tile.

16 loose scalars (16 separate y = mx + c) belong on CUDA cores. A 16×16 dense tile belongs on a Tensor core. A 10×10 multiply is padded toward 16 and is a toy.

Housing prices: 1000 rows, features sqft / beds / floors / baths.

- One house is one scalar y = w·x + b.
- 1000 houses are 1000 scalars and also one skinny GEMM Y = XW + b, X is 1000×4.
- 16 features still fine on CUDA cores. Tensor cores want fat W (thousands), which is an LLM layer.

Stacked GEMMs: Q, K, V, attention, output proj, FFN. One transformer block is a chain of Y = MX + C. A 70B model repeats that for many layers. That pile is why Tensor cores exist.

Tensor cores did not replace warp 32. A warp still drives the MMA. They were added because dense GEMMs waste scalar CUDA cores. Softmax, layernorm, RoPE, sampling stay on CUDA cores.

## 6. FLOPS

FLOP = one floating-point add or multiply.
FLOPS = how many per second.

Work is FLOPs. Speed is FLOPS. A job of 1 FLOP in 1 second is 1 FLOPS. The same job in 0.1 seconds is 10 FLOPS.

32,000 × 32,000 GEMM is 2 × 32000³ ≈ 6.55×10^13 FLOPs. 32,000 CUDA cores at one FMA per clock do about 64,000 FLOPs per clock, so about a billion clocks, not one. One clock would need on the order of 8 billion Tensor cores if each stamp is an 8192-FLOP MMA.

Brochure peaks (order of magnitude, late 2026):

- M5 Ultra GPU: tens of FP32 TFLOPS, higher if Neural Accelerators are counted.
- RTX 5090: ~100 TFLOPS FP32, much more on the Tensor path at low precision, ~575 W.
- H100: ~989 BF16 TFLOPS, ~1979 FP8, ~700 W.
- B200: ~2250 BF16, ~4500 FP8, ~1000 W.

NVIDIA wins FLOPS because the die is almost all GPU, Tensor cores are specialized, power is allowed to be ugly, HBM/GDDR feeds them, and you can glue many cards. Apple wins joules and gigabytes.

## 7. The wall

Limits that matter: power, memory bandwidth, capacity, distance between chips, manufacturing, utilization. Warp size is a detail.

Quantum machines can represent linear algebra. They do not replace a Tensor-core GEMM. Loading the matrix and reading out every entry cancels the usual theoretical speedups. LLMs are classical streaming GEMMs.

Low precision (FP16, FP8, FP4, Q4) does not solve exact Y = MX + C. It changes the problem to an approximate GEMM that is often good enough for next-token quality. That is frugal engineering.

Split:

- Business may redefine the problem. That is still a real problem. Engineering is then measured against the new contract.
- Engineering redefining the problem because it cannot beat the constraints is frugal. Acceptable. There is no lamp. It cannot hold the ground forever.
- Elevating engineering beats a constraint and creates a new class of problems (Tensor cores → feed the MMA; jets → materials; transistors → lithography).

Last 10–15 years: real product advance (transformers, Tensor cores, working large models) and a wall on exact ops per watt per package. The world rejoices at outputs. GPU engineers are still on joules, packaging, and wires. Free energy would rename the wall (bandwidth, fabric, algorithms), not delete it.

## 8. One-line versions

- CUDA and Metal/XLA do not interchange. Weights can.
- Copy is real on discrete NVIDIA; skipped on Apple and Intel iGPU.
- Display is not the GEMM path.
- CUDA core: one scalar FMA. Tensor core: one tile Y = MX + C.
- 32 is warp width, not max batch.
- FLOPs is work. FLOPS is speed.
- Fewer bits is a spec change, not a broken wall.
- Frugal is allowed. It is not the same as overcoming the constraint.
