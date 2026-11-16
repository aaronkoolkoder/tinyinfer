#pragma once

#include <cstdint>

#include "tensor.h"

namespace tinyinfer {

// out[M, N] = x[M, K] @ w[N, K]^T
void matmul(float* out, const float* x, const float* w, int64_t M, int64_t N, int64_t K);

// Shape-checked wrapper. x: [..., K], w: [N, K] -> out: [..., N]
void matmul(Tensor& out, const Tensor& x, const Tensor& w);

// out[M, N] = x[M, K] @ w[N, K]^T + b[N]
void linear(float* out, const float* x, const float* w, const float* b, int64_t M, int64_t N, int64_t K);

// out[t] = wte[tokens[t]] + wpe[t], out: [T, C]
void embed(float* out, const float* wte, const float* wpe, const int32_t* tokens, int64_t T, int64_t C);

// Per-row LayerNorm over C. x, out: [T, C]; gamma, beta: [C]. out may alias x.
void layernorm(float* out, const float* x, const float* gamma, const float* beta, int64_t T, int64_t C, float eps);

// GPT-2's tanh approximation of GELU, in place.
void gelu(float* x, int64_t n);

// x += y
void add_inplace(float* x, const float* y, int64_t n);

// qkv: [T, 3C] rows laid out as [q | k | v], out: [T, C]. See attention.cpp.
void causal_self_attention(float* out, const float* qkv, int64_t T, int64_t C, int64_t n_head);

// Per-row RMSNorm: out = x / sqrt(mean(x^2) + eps) * gamma. out may alias x.
void rmsnorm(float* out, const float* x, const float* gamma, int64_t T, int64_t C, float eps);

// gate = silu(gate) * up, in place.
void swiglu(float* gate, const float* up, int64_t n);

// Rotary position embedding, in place, Hugging Face "rotate_half" layout:
// within each head, element i pairs with element i + D/2.
// x: [T, n_heads * D]; row t is at absolute position pos + t.
void rope(float* x, int64_t T, int64_t n_heads, int64_t D, int64_t pos, float theta);

// Grouped-query causal attention over a cache.
//   q: [T, H*D]; k_cache, v_cache: [KVH, max_seq, D] already holding positions 0..pos+T-1
//   query head h reads kv head h / (H / KVH). out: [T, H*D]
void gqa_attention(float* out, const float* q, const float* k_cache, const float* v_cache, int64_t pos, int64_t T,
                   int64_t H, int64_t KVH, int64_t D, int64_t max_seq);

}  // namespace tinyinfer
