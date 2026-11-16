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

}  // namespace tinyinfer
