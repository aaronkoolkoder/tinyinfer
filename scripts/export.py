#!/usr/bin/env python3
"""Export a Hugging Face checkpoint and reference outputs to .tinf files.

Supports GPT-2 and Llama-style models (Llama, Qwen2).

Usage:
    python scripts/export.py --model gpt2 --out models/gpt2
    python scripts/export.py --model Qwen/Qwen2.5-0.5B-Instruct --out models/qwen2.5-0.5b
"""
import argparse
import json
import os
import shutil
import struct
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MAGIC = b"TINF"
VERSION = 1
ALIGN = 64
DTYPE_F32, DTYPE_I32 = 0, 1
ARCH_GPT2, ARCH_LLAMA = 1, 2

PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
    "In a shocking finding, scientists discovered a herd of unicorns living in",
    "def fibonacci(n):",
]
GEN_TOKENS = 32

TOKENIZER_CASES = [
    "Hello world",
    "  leading spaces and trailing   ",
    "Tabs\tand\nnewlines\n\nhere",
    "It's we'll they've I'm you'd she's don't",
    "Numbers 12345 and 3.14159, plus 1,000,000!",
    "Punctuation!!! ...?? (parens) [brackets] {braces} <angle>",
    "caf\u00e9 na\u00efve r\u00e9sum\u00e9 Z\u00fcrich",
    "\u041f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440",
    "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8",
    "emoji \U0001f642\U0001f44d mixed",
    "def f(x):\n    return x * 2\n",
    "CamelCaseWords and snake_case_words",
    "a  b   c    d",
    "end with newline\n",
]


def text_bytes(s):
    return np.frombuffer(s.encode("utf-8"), dtype=np.uint8).astype(np.int32)


def write_tinf(path, config, tensors):
    """config: dict[str, int]; tensors: list of (name, np.ndarray)."""
    entries = []
    for name, arr in tensors:
        arr = np.ascontiguousarray(arr)
        if arr.dtype == np.float32:
            dtype = DTYPE_F32
        elif arr.dtype == np.int32:
            dtype = DTYPE_I32
        else:
            raise TypeError(f"{name}: unsupported dtype {arr.dtype}")
        entries.append((name, dtype, arr))

    header = bytearray()
    header += MAGIC
    header += struct.pack("<I", VERSION)
    header += struct.pack("<I", len(config))
    for k, v in config.items():
        kb = k.encode()
        header += struct.pack("<I", len(kb)) + kb + struct.pack("<q", int(v))
    header += struct.pack("<I", len(entries))

    index_size = sum(4 + len(n.encode()) + 4 + 4 + 8 * a.ndim + 8 + 8 for n, _, a in entries)
    offset = _align(len(header) + index_size)
    index = bytearray()
    for name, dtype, arr in entries:
        nb = name.encode()
        index += struct.pack("<I", len(nb)) + nb
        index += struct.pack("<II", dtype, arr.ndim)
        index += struct.pack(f"<{arr.ndim}q", *arr.shape)
        index += struct.pack("<QQ", offset, arr.nbytes)
        offset = _align(offset + arr.nbytes)

    with open(path, "wb") as f:
        f.write(header)
        f.write(index)
        for _, _, arr in entries:
            f.write(b"\0" * (_align(f.tell()) - f.tell()))
            f.write(arr.tobytes())


def _align(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


def export_gpt2(model, out):
    cfg = model.config
    sd = {k: v.detach().float().numpy() for k, v in model.state_dict().items()}
    config = {
        "arch": ARCH_GPT2,
        "n_vocab": cfg.vocab_size,
        "n_ctx": cfg.n_positions,
        "n_embd": cfg.n_embd,
        "n_head": cfg.n_head,
        "n_layer": cfg.n_layer,
        "n_inner": cfg.n_inner or 4 * cfg.n_embd,
        "layer_norm_eps_x1e9": round(cfg.layer_norm_epsilon * 1e9),
    }

    # GPT-2 uses Conv1D with weights stored [in, out]; transpose to [out, in]
    # so every linear layer in the runtime is y = x @ W^T.
    conv1d = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")
    tensors = [("wte", sd["transformer.wte.weight"]), ("wpe", sd["transformer.wpe.weight"])]
    for i in range(cfg.n_layer):
        p = f"transformer.h.{i}."
        for k in ("ln_1.weight", "ln_1.bias", "attn.c_attn.weight", "attn.c_attn.bias",
                  "attn.c_proj.weight", "attn.c_proj.bias", "ln_2.weight", "ln_2.bias",
                  "mlp.c_fc.weight", "mlp.c_fc.bias", "mlp.c_proj.weight", "mlp.c_proj.bias"):
            w = sd[p + k]
            if k in conv1d:
                w = w.T
            tensors.append((f"h.{i}.{k}", w))
    tensors += [("ln_f.weight", sd["transformer.ln_f.weight"]), ("ln_f.bias", sd["transformer.ln_f.bias"])]
    # lm_head is tied to wte, so it is not stored separately.

    write_tinf(out / "model.tinf", config, tensors)
    return config


def export_llama(model, out):
    cfg = model.config
    sd = {k: v.detach().float().numpy() for k, v in model.state_dict().items()}
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    has_bias = "model.layers.0.self_attn.q_proj.bias" in sd
    tied = bool(getattr(cfg, "tie_word_embeddings", False))
    rope = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
    if rope.get("rope_type", "default") != "default":
        raise NotImplementedError(f"rope type {rope['rope_type']} is not supported")
    rope_theta = rope.get("rope_theta") or getattr(cfg, "rope_theta", 10000.0)
    config = {
        "arch": ARCH_LLAMA,
        "n_vocab": cfg.vocab_size,
        "n_ctx": cfg.max_position_embeddings,
        "n_embd": cfg.hidden_size,
        "n_head": cfg.num_attention_heads,
        "n_kv_head": cfg.num_key_value_heads,
        "head_dim": head_dim,
        "n_layer": cfg.num_hidden_layers,
        "n_inner": cfg.intermediate_size,
        "rope_theta": round(rope_theta),
        "rms_eps_x1e9": round(cfg.rms_norm_eps * 1e9),
        "attn_bias": int(has_bias),
        "tied_embeddings": int(tied),
    }
    tensors = [("embed", sd["model.embed_tokens.weight"])]
    names = {
        "input_layernorm.weight": "attn_norm",
        "self_attn.q_proj.weight": "wq",
        "self_attn.k_proj.weight": "wk",
        "self_attn.v_proj.weight": "wv",
        "self_attn.o_proj.weight": "wo",
        "post_attention_layernorm.weight": "ffn_norm",
        "mlp.gate_proj.weight": "w_gate",
        "mlp.up_proj.weight": "w_up",
        "mlp.down_proj.weight": "w_down",
    }
    if has_bias:
        names.update({"self_attn.q_proj.bias": "bq", "self_attn.k_proj.bias": "bk", "self_attn.v_proj.bias": "bv"})
    for i in range(cfg.num_hidden_layers):
        for hf, ours in names.items():
            tensors.append((f"l.{i}.{ours}", sd[f"model.layers.{i}.{hf}"]))
    tensors.append(("norm", sd["model.norm.weight"]))
    if not tied:
        tensors.append(("lm_head", sd["lm_head.weight"]))
    write_tinf(out / "model.tinf", config, tensors)
    return config


@torch.no_grad()
def greedy(model, ids, n):
    # Plain argmax loop: no eos stopping, no repetition penalty from generation_config.
    seq = ids
    for _ in range(n):
        nxt = model(seq).logits[:, -1].argmax(-1, keepdim=True)
        seq = torch.cat([seq, nxt], dim=1)
    return seq[0, ids.shape[1]:]


@torch.no_grad()
def export_reference(model, tok, out):
    tensors = []
    for i, prompt in enumerate(PROMPTS):
        ids = tok(prompt, return_tensors="pt").input_ids
        res = model(ids, output_hidden_states=True)
        gen = greedy(model, ids, GEN_TOKENS)
        tensors += [
            (f"p{i}.text", text_bytes(prompt)),
            (f"p{i}.tokens", ids[0].numpy().astype(np.int32)),
            (f"p{i}.logits", res.logits[0].numpy().astype(np.float32)),
            (f"p{i}.hidden", torch.stack(res.hidden_states)[:, 0].numpy().astype(np.float32)),
            (f"p{i}.greedy", gen.numpy().astype(np.int32)),
        ]
    # Strings are stored as i32 arrays of UTF-8 bytes.
    for i, text in enumerate(TOKENIZER_CASES):
        tensors += [
            (f"tok{i}.text", text_bytes(text)),
            (f"tok{i}.ids", np.array(tok(text).input_ids, dtype=np.int32)),
        ]
    config = {"n_prompts": len(PROMPTS), "gen_tokens": GEN_TOKENS, "n_tok_cases": len(TOKENIZER_CASES)}
    write_tinf(out / "reference.tinf", config, tensors)
    with open(out / "prompts.json", "w") as f:
        json.dump(PROMPTS, f, indent=2)


def export_tokenizer(model_id, tok, model_type, out):
    for name in ("vocab.json", "merges.txt"):  # consumed by the C++ BPE tokenizer
        shutil.copy(hf_hub_download(model_id, name), out / name)
    (out / "pretokenizer.txt").write_text("gpt2\n" if model_type == "gpt2" else "qwen2\n")
    with open(out / "special_tokens.txt", "w") as f:
        for tid, added in sorted(tok.added_tokens_decoder.items()):
            if added.special:
                f.write(f"{tid}\t{added.content}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--out", default="models/gpt2")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model_type = AutoConfig.from_pretrained(args.model).model_type
    if model_type not in ("gpt2", "llama", "qwen2"):
        raise SystemExit(f"unsupported model type {model_type}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(args.model)
    export_tokenizer(args.model, tok, model_type, out)

    config = export_gpt2(model, out) if model_type == "gpt2" else export_llama(model, out)
    export_reference(model, tok, out)
    size = os.path.getsize(out / "model.tinf") / 2**20
    print(f"wrote {out}/model.tinf ({size:.0f} MiB), reference.tinf, tokenizer files")
    print(json.dumps(config))


if __name__ == "__main__":
    main()
