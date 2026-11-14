#include "ops.h"

#include <stdexcept>

namespace tinyinfer {

void matmul(Tensor& out, const Tensor& x, const Tensor& w) {
    if (w.ndim() != 2 || x.ndim() < 1 || out.ndim() != x.ndim())
        throw std::invalid_argument("matmul: bad ranks");
    const int64_t K = x.dim(-1), N = w.dim(0);
    const int64_t M = x.numel() / K;
    if (w.dim(1) != K || out.dim(-1) != N || out.numel() != M * N)
        throw std::invalid_argument("matmul: " + out.shape_str() + " = " + x.shape_str() + " @ " + w.shape_str() + "^T");
    matmul(out.f32(), x.f32(), w.f32(), M, N, K);
}

}  // namespace tinyinfer
