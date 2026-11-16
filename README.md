# tinyinfer

A transformer inference runtime written from scratch in C++17. No PyTorch, no BLAS, no ML libraries in the runtime. Python is only used to export weights and produce reference outputs.

Starts with GPT-2 small, adds a KV cache, then Llama-style models, then a CPU performance pass. A CUDA backend comes later.

## Roadmap

| Milestone | Scope | Status |
|---|---|---|
| v1 | Repo scaffold, CMake, GPT-2 weight export + reference outputs | done |
| v2 | Tensor utilities, weight loader, naive matmul | done |
| v3 | Full GPT-2 forward pass, logits match Hugging Face | done |
| v4 | BPE tokenizer, greedy generation matches Hugging Face token for token | done |
| v5 | Sampling: temperature, top-k, top-p, seeding, CLI flags | done |
| v6 | KV cache, prefill vs. decode, tokens/sec with and without cache | |
| v7 | Llama-style models (RMSNorm, SwiGLU, GQA, RoPE) | |
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
