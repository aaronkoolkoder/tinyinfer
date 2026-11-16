#include "ops.h"

#include <cmath>
#include <cstring>
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

void linear(float* out, const float* x, const float* w, const float* b, int64_t M, int64_t N, int64_t K) {
    matmul(out, x, w, M, N, K);
    if (!b) return;
    for (int64_t m = 0; m < M; ++m) {
        float* row = out + m * N;
        for (int64_t n = 0; n < N; ++n) row[n] += b[n];
    }
}

void embed(float* out, const float* wte, const float* wpe, const int32_t* tokens, int64_t T, int64_t C) {
    for (int64_t t = 0; t < T; ++t) {
        const float* te = wte + static_cast<int64_t>(tokens[t]) * C;
        const float* pe = wpe + t * C;
        float* o = out + t * C;
        for (int64_t c = 0; c < C; ++c) o[c] = te[c] + pe[c];
    }
}

void layernorm(float* out, const float* x, const float* gamma, const float* beta, int64_t T, int64_t C, float eps) {
    for (int64_t t = 0; t < T; ++t) {
        const float* xr = x + t * C;
        float* o = out + t * C;
        double mean = 0.0;
        for (int64_t c = 0; c < C; ++c) mean += xr[c];
        mean /= C;
        double var = 0.0;
        for (int64_t c = 0; c < C; ++c) {
            double d = xr[c] - mean;
            var += d * d;
        }
        var /= C;
        const float m = static_cast<float>(mean);
        const float rstd = static_cast<float>(1.0 / std::sqrt(var + eps));
        for (int64_t c = 0; c < C; ++c) o[c] = (xr[c] - m) * rstd * gamma[c] + beta[c];
    }
}

void gelu(float* x, int64_t n) {
    const float k = 0.7978845608f;  // sqrt(2 / pi)
    for (int64_t i = 0; i < n; ++i) {
        const float v = x[i];
        x[i] = 0.5f * v * (1.0f + std::tanh(k * (v + 0.044715f * v * v * v)));
    }
}

void add_inplace(float* x, const float* y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) x[i] += y[i];
}

}  // namespace tinyinfer
