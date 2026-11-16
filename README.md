# tinyinfer

A transformer inference runtime written from scratch in C++17. Runs GPT-2 and Llama-style models (tested on Qwen2.5-0.5B-Instruct). No PyTorch, no BLAS, no ML libraries in the runtime. Python is only used to export weights and produce reference outputs.

Starts with GPT-2 small, adds a KV cache, then Llama-style models, then a CPU performance pass. A CUDA backend comes later.

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
| v8 | CPU performance: OpenMP, cache-friendly layouts, benchmark tables and charts | |
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
