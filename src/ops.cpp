#include "ops.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
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

void rmsnorm(float* out, const float* x, const float* gamma, int64_t T, int64_t C, float eps) {
    for (int64_t t = 0; t < T; ++t) {
        const float* xr = x + t * C;
        float* o = out + t * C;
        double ss = 0.0;
        for (int64_t c = 0; c < C; ++c) ss += static_cast<double>(xr[c]) * xr[c];
        const float rms = static_cast<float>(1.0 / std::sqrt(ss / C + eps));
        for (int64_t c = 0; c < C; ++c) o[c] = xr[c] * rms * gamma[c];
    }
}

void swiglu(float* gate, const float* up, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        const float g = gate[i];
        gate[i] = g / (1.0f + std::exp(-g)) * up[i];
    }
}

void gqa_attention(float* out, const float* q, const float* k_cache, const float* v_cache, int64_t pos, int64_t T,
                   int64_t H, int64_t KVH, int64_t D, int64_t max_seq) {
    const int64_t group = H / KVH;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    std::vector<float> scores(pos + T);
    for (int64_t h = 0; h < H; ++h) {
        const float* kh = k_cache + (h / group) * max_seq * D;
        const float* vh = v_cache + (h / group) * max_seq * D;
        for (int64_t t = 0; t < T; ++t) {
            const float* qr = q + t * H * D + h * D;
            float* o = out + t * H * D + h * D;
            const int64_t n_keys = pos + t + 1;
            float mx = -std::numeric_limits<float>::infinity();
            for (int64_t s = 0; s < n_keys; ++s) {
                const float* kr = kh + s * D;
                float dot = 0.0f;
                for (int64_t i = 0; i < D; ++i) dot += qr[i] * kr[i];
                scores[s] = dot * scale;
                mx = std::max(mx, scores[s]);
            }
            double sum = 0.0;
            for (int64_t s = 0; s < n_keys; ++s) {
                scores[s] = std::exp(scores[s] - mx);
                sum += scores[s];
            }
            const float inv = static_cast<float>(1.0 / sum);
            std::fill(o, o + D, 0.0f);
            for (int64_t s = 0; s < n_keys; ++s) {
                const float* vr = vh + s * D;
                const float w = scores[s] * inv;
                for (int64_t i = 0; i < D; ++i) o[i] += w * vr[i];
            }
        }
    }
}

}  // namespace tinyinfer
