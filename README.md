# tinyinfer

A transformer inference runtime written from scratch in C++17. Runs GPT-2 and Llama-style models (tested on Qwen2.5-0.5B-Instruct) with a KV cache, sampling, and multithreaded SIMD kernels. No PyTorch, no BLAS, no ML libraries in the runtime; Python is only used to export weights and produce reference outputs.

Every model is checked against Hugging Face: logits within tolerance and greedy generation token for token.

**Headline:** on Qwen2.5-0.5B-Instruct (F32, CPU only, Apple M5 Pro), tinyinfer decodes at **~100% of llama.cpp's speed** (98.6 vs 96.4 tok/s best-vs-best, 102–104% across two head-to-head runs) and prefills at **78–95%** of llama.cpp, which hands prefill matmuls to Apple Accelerate. See [Performance](#performance-v8).

![tinyinfer vs llama.cpp](bench/results/compare.png)

## Roadmap

| Milestone | Scope | Status |
|---|---|---|
| v1 | Repo scaffold, CMake, GPT-2 weight export + reference outputs | done |
| v2 | Tensor utilities, weight loader, naive matmul | done |
| v3 | Full GPT-2 forward pass, logits match Hugging Face | done |
| v4 | BPE tokenizer, greedy generation matches Hugging Face token for token | done |
| v5 | Sampling: temperature, top-k, top-p, seeding, CLI flags | done |
| v6 | KV cache, prefill vs. decode, tokens/sec with and without cache | done |
| v7 | Llama-style models (RMSNorm, SwiGLU, GQA, RoPE) | done |
| v8 | CPU performance: OpenMP, NEON kernels, fused matmuls, benchmarks vs llama.cpp | done |
| v9+ | CUDA backend | |

## Layout

```
src/       runtime
tests/     unit and reference tests (CTest)
scripts/   weight export and tooling (Python)
bench/     benchmarks
models/    exported weights and references (gitignored)
```

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r scripts/requirements.txt
.venv/bin/python scripts/export.py --model gpt2 --out models/gpt2
.venv/bin/python scripts/export.py --model Qwen/Qwen2.5-0.5B-Instruct --out models/qwen2.5-0.5b
cmake -B build && cmake --build build -j
ctest --test-dir build --output-on-failure
```

Generate text:

```bash
./build/tinyinfer -p "In a shocking finding, scientists discovered a herd of unicorns living in" -n 40
```

Sample instead of greedy decoding:

```bash
./build/tinyinfer -p "Once upon a time" -n 40 --temp 0.8 --top-p 0.9 --seed 42
```

Flags: `--temp` (0 = greedy), `--top-k`, `--top-p`, `--seed` (printed at the end if not given). The pipeline is temperature → top-k → softmax → top-p → sample, the same order Hugging Face uses.

The tokenizer reads `vocab.json` and `merges.txt` directly and reimplements GPT-2's byte-level BPE, including its pre-tokenizer regex. Unicode letter and number classes cover common scripts rather than the full Unicode tables.

`test_generate` checks the first 10 greedy tokens by default; `TINYINFER_FULL=1 ./build/test_generate` checks all 32.

Chat with the instruct model:

```bash
./build/tinyinfer -m models/qwen2.5-0.5b --chat -p "Explain what a KV cache is in one sentence." --temp 0.7 --top-p 0.9
```

## Performance (v8)

### What changed

| Change | Why it matters |
|---|---|
| NEON matmul with register tiling (`src/kernels.cpp`) | 4x4 tiles of x rows by weight rows keep 16 accumulators in registers: 16 FMAs per 8 loads. Decode uses a 1x4 variant that loads x once per step. |
| OpenMP over output rows, dynamic scheduling | Apple chips mix performance and efficiency cores; small chunks let fast cores take more work instead of waiting at the barrier. |
| Threads spin between parallel regions | Decode launches ~100 small matmuls per token. With libomp's default, parked threads took ~60 µs to wake for a matmul that needs 4 µs. Spinning cut a 128x896 matmul at 10 threads from 61 µs to 7 µs. |
| Fewer threads for small matmuls | Roughly one thread per MiB of weights, so a 0.5 MiB projection does not wake 12 cores. |
| Fused Q/K/V and gate/up weights | 7 matmuls per layer become 4, fewer synchronization points per token. |

The original single-threaded naive matmul stays as `matmul_naive`, the reference the fast kernel is tested against on shapes that hit every tail path.

### Results vs llama.cpp

Same model, same F32 weights (official Qwen GGUF expanded with `llama-quantize ... F32`), `llama-bench -dev none -ngl 0` for CPU only, and `bench_model`, which measures the same two things: `pp128` processes a 128-token prompt in one batch, `tg64` decodes 64 tokens one at a time. Two back-to-back runs, 5 repetitions each ([run 1](bench/results/compare_run1.csv), [run 2](bench/results/compare.csv)):

| test | tinyinfer best | llama.cpp best | run 1 | run 2 |
|---|---:|---:|---:|---:|
| decode tg64 | 98.6 tok/s (10 threads) | 96.4 tok/s (12 threads) | 104% | 102% |
| prefill pp128 | 911 tok/s (12 threads) | 993 tok/s (5 threads) | 95% | 92% |

llama.cpp varies run to run on this machine; a separate solo run reached 95.9 tok/s decode and 1164 tok/s prefill, which puts prefill at 78%. Decode stays at roughly parity in every run. llama.cpp's prefill uses Apple Accelerate (the CPU's matrix coprocessor); tinyinfer's is hand-written NEON.

Reproduce with `scripts/bench_vs_llamacpp.sh` (needs `brew install llama.cpp`).

GPT-2 small, same machine (`bench_model models/gpt2`):

| | naive, no cache (v4) | naive + KV cache (v6) | v8, 8–10 threads |
|---|---:|---:|---:|
| decode | 0.87 tok/s | 19.6 tok/s | 331 tok/s |
| prefill | ~27 tok/s | ~27 tok/s | 1441 tok/s |

### Why decode is memory-bandwidth bound

Each decode step multiplies one token's activations (a vector) by every weight matrix in the model. Every weight is read from memory once and used for exactly one multiply-add:

- Bytes per token: all of Qwen2.5-0.5B's weights, 1.98 GB in F32 (the tied embedding is read in full as the output head).
- FLOPs per token: about 2 per weight, so ~1 GFLOP.
- Arithmetic intensity: 2 FLOPs per 4 bytes = **0.5 FLOP/byte**.

At 98.6 tok/s tinyinfer streams 98.6 x 1.98 GB = **~195 GB/s** of weights, which is close to the ~235 GB/s the fastest single streaming matmul reached on this machine. The arithmetic actually performed is only ~97 GFLOP/s, about a tenth of what the same kernels sustain in prefill (~1000 GFLOP/s). The cores mostly wait for memory, so decode speed is set by bytes per token divided by memory bandwidth, and two well-written engines land at the same number.

Prefill is the opposite. With 128 tokens, every weight loaded from memory is reused 128 times, giving ~64 FLOP/byte, so prefill is limited by compute and the quality of the matmul kernel matters. That's where llama.cpp's use of Apple's matrix hardware shows.

What this implies:

- **Smaller weights are faster.** F16 would halve the bytes per token and roughly double decode speed; 4-bit quantization would give ~8x. That's why llama.cpp users run quantized models.
- **More threads stop helping** once memory is saturated (decode peaks at 8–10 threads; 15 is slower).
- **Batching helps servers.** Decoding 8 users at once reads the weights once for 8 tokens, raising intensity 8x. This is the main idea behind vLLM-style serving.
- **The KV cache trades memory for compute.** It turns O(n) recomputation per token into reading the cache, which is itself a memory cost that grows with context length.

## Llama-style models (v7)

`LlamaModel` implements the decoder used by Llama, Qwen2 and most current open models. Differences from GPT-2:

| | GPT-2 | Llama-style |
|---|---|---|
| Normalization | LayerNorm (mean, variance, bias) | RMSNorm (scale only) |
| Positions | learned `wpe` added to the input | RoPE: rotate q and k by position-dependent angles inside every layer |
| Attention | 12 heads, each with its own K/V | grouped-query: 14 query heads share 2 K/V heads, so the KV cache is 7x smaller |
| MLP | `gelu(x W1) W2` | SwiGLU: `(silu(x Wg) * x Wu) Wd` |
| Biases | everywhere | none, except Qwen2's q/k/v |

The tokenizer reuses the byte-level BPE with Qwen2's pre-tokenizer regex (single digits, case-insensitive contractions, punctuation attached to the following word) and splits out special tokens like `<|im_start|>` before BPE. `test_llama` checks the tokenizer, prefill logits, cached decode, and 32 greedy tokens per prompt against Hugging Face.

## KV cache (v6)

Without a cache, generating each token reruns the whole sequence through every layer, so step `n` costs O(n) full forward passes worth of work. Under the causal mask, a token's keys and values never change once computed, so the cache stores them per layer (`[n_layer, n_head, max_seq, head_size]`, one head's history contiguous) and each decode step only processes the new token.

GPT-2 small, greedy, 32 new tokens, single thread, naive matmul (`./build/bench_generate`):

| prompt | cache | time to first token (s) | decode tok/s | total (s) | cache MiB |
|---:|:---:|---:|---:|---:|---:|
| 16 | off | 0.60 | 0.87 | 36.2 | — |
| 16 | on | 0.58 | 19.62 | 2.2 | 3.4 |
| 128 | off | 4.67 | 0.19 | 168.7 | — |
| 128 | on | 4.58 | 20.20 | 6.1 | 11.2 |

Decode is 22x faster at a 16-token prompt and 106x faster at 128. Without the cache, decode speed falls as the sequence grows; with it, decode speed is flat because each step does the same matmul work and only attention grows with length.

## Weight format (`.tinf`)

Little-endian, one file per model. Reference outputs use the same container.

```
"TINF"              4 bytes magic
version             u32 (= 1)
n_config            u32
  key_len u32, key bytes, value i64           (repeated n_config times)
n_tensors           u32
  name_len u32, name bytes,
  dtype u32 (0 = f32, 1 = i32), ndim u32,
  dims i64[ndim], offset u64, nbytes u64      (repeated n_tensors times)
tensor data, each tensor starting at a 64-byte aligned absolute offset
```

Conventions:

- All weights are f32, row-major.
- Every linear weight is stored `[out, in]`, so a linear layer is `y = x @ Wᵀ + b`. GPT-2's Conv1D weights (`[in, out]`) are transposed during export.
- `c_attn` packs Q, K, V along the output dimension: `[3 * n_embd, n_embd]`.
- `lm_head` is tied to `wte` and not stored.
- `layer_norm_eps_x1e9` stores epsilon as an integer (1e-5 → 10000).

GPT-2 tensor names: `wte [n_vocab, n_embd]`, `wpe [n_ctx, n_embd]`, `h.{i}.ln_1.{weight,bias}`, `h.{i}.attn.c_attn.{weight,bias}`, `h.{i}.attn.c_proj.{weight,bias}`, `h.{i}.ln_2.{weight,bias}`, `h.{i}.mlp.c_fc.{weight,bias}`, `h.{i}.mlp.c_proj.{weight,bias}`, `ln_f.{weight,bias}`.

## Reference outputs (`reference.tinf`)

For each fixed prompt `i` (listed in `prompts.json`):

| Tensor | Shape | Meaning |
|---|---|---|
| `p{i}.text` | `[bytes]` i32 | prompt as UTF-8 bytes |
| `p{i}.tokens` | `[T]` i32 | Hugging Face token ids for the prompt |
| `p{i}.logits` | `[T, n_vocab]` f32 | logits at every position |
| `p{i}.hidden` | `[n_layer + 1, T, n_embd]` f32 | residual stream after embeddings and after each block (last entry is after `ln_f`) |
| `p{i}.greedy` | `[32]` i32 | greedy continuation tokens |

`tok{i}.text` / `tok{i}.ids` hold extra tokenizer cases (whitespace, contractions, Unicode, emoji).

The per-layer hidden states make it possible to find exactly which block diverges when the forward pass is wrong.
