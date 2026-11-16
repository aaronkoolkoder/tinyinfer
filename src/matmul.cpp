#include "ops.h"

namespace tinyinfer {

// Reference version: out[M, N] = x[M, K] @ w[N, K]^T, one dot product per output.
// The fast path in kernels.cpp is tested against this.
void matmul_naive(float* out, const float* x, const float* w, int64_t M, int64_t N, int64_t K) {
    for (int64_t m = 0; m < M; ++m) {
        const float* xr = x + m * K;
        for (int64_t n = 0; n < N; ++n) {
            const float* wr = w + n * K;
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) sum += xr[k] * wr[k];
            out[m * N + n] = sum;
        }
    }
}

}  // namespace tinyinfer
