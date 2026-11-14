#pragma once

#include <cstdint>

#include "tensor.h"

namespace tinyinfer {

// out[M, N] = x[M, K] @ w[N, K]^T
void matmul(float* out, const float* x, const float* w, int64_t M, int64_t N, int64_t K);

// Shape-checked wrapper. x: [..., K], w: [N, K] -> out: [..., N]
void matmul(Tensor& out, const Tensor& x, const Tensor& w);

}  // namespace tinyinfer
