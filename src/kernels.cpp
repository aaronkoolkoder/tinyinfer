#include <algorithm>
#include <cstdlib>

#include "ops.h"

#if defined(_OPENMP)
#include <omp.h>
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace tinyinfer {

namespace {

int g_threads = 0;

// libomp parks idle threads quickly by default; decode launches ~100 small
// parallel regions per token, so waking threads each time costs more than the
// work. Keep them spinning between regions (user env vars still win).
struct OmpSpin {
    OmpSpin() {
        setenv("OMP_WAIT_POLICY", "active", 0);
        setenv("KMP_BLOCKTIME", "1000", 0);
    }
} g_omp_spin;

#if defined(__ARM_NEON)

inline float dot(const float* a, const float* b, int64_t K) {
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
    int64_t k = 0;
    for (; k + 16 <= K; k += 16) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + k), vld1q_f32(b + k));
        s1 = vfmaq_f32(s1, vld1q_f32(a + k + 4), vld1q_f32(b + k + 4));
        s2 = vfmaq_f32(s2, vld1q_f32(a + k + 8), vld1q_f32(b + k + 8));
        s3 = vfmaq_f32(s3, vld1q_f32(a + k + 12), vld1q_f32(b + k + 12));
    }
    for (; k + 4 <= K; k += 4) s0 = vfmaq_f32(s0, vld1q_f32(a + k), vld1q_f32(b + k));
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; k < K; ++k) s += a[k] * b[k];
    return s;
}

// One x row against four w rows: x is loaded once per step.
inline void dot_1x4(const float* x, const float* w, int64_t K, float* out) {
    const float *w0 = w, *w1 = w + K, *w2 = w + 2 * K, *w3 = w + 3 * K;
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
    float32x4_t b0 = a0, b1 = a0, b2 = a0, b3 = a0;
    int64_t k = 0;
    for (; k + 8 <= K; k += 8) {
        const float32x4_t x0 = vld1q_f32(x + k), x1 = vld1q_f32(x + k + 4);
        a0 = vfmaq_f32(a0, x0, vld1q_f32(w0 + k));
        a1 = vfmaq_f32(a1, x0, vld1q_f32(w1 + k));
        a2 = vfmaq_f32(a2, x0, vld1q_f32(w2 + k));
        a3 = vfmaq_f32(a3, x0, vld1q_f32(w3 + k));
        b0 = vfmaq_f32(b0, x1, vld1q_f32(w0 + k + 4));
        b1 = vfmaq_f32(b1, x1, vld1q_f32(w1 + k + 4));
        b2 = vfmaq_f32(b2, x1, vld1q_f32(w2 + k + 4));
        b3 = vfmaq_f32(b3, x1, vld1q_f32(w3 + k + 4));
    }
    float r0 = vaddvq_f32(vaddq_f32(a0, b0)), r1 = vaddvq_f32(vaddq_f32(a1, b1));
    float r2 = vaddvq_f32(vaddq_f32(a2, b2)), r3 = vaddvq_f32(vaddq_f32(a3, b3));
    for (; k < K; ++k) {
        r0 += x[k] * w0[k];
        r1 += x[k] * w1[k];
        r2 += x[k] * w2[k];
        r3 += x[k] * w3[k];
    }
    out[0] = r0;
    out[1] = r1;
    out[2] = r2;
    out[3] = r3;
}

// Four x rows against four w rows: 16 accumulators, 8 loads per 16 FMAs.
inline void dot_4x4(const float* x, const float* w, int64_t K, float* out, int64_t ldo) {
    float32x4_t acc[4][4];
    for (auto& row : acc)
        for (auto& a : row) a = vdupq_n_f32(0);
    int64_t k = 0;
    for (; k + 4 <= K; k += 4) {
        float32x4_t xv[4], wv[4];
        for (int i = 0; i < 4; ++i) xv[i] = vld1q_f32(x + i * K + k);
        for (int j = 0; j < 4; ++j) wv[j] = vld1q_f32(w + j * K + k);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) acc[i][j] = vfmaq_f32(acc[i][j], xv[i], wv[j]);
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            float s = vaddvq_f32(acc[i][j]);
            for (int64_t kk = k; kk < K; ++kk) s += x[i * K + kk] * w[j * K + kk];
            out[i * ldo + j] = s;
        }
}

#else

inline float dot(const float* a, const float* b, int64_t K) {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int64_t k = 0;
    for (; k + 4 <= K; k += 4) {
        s0 += a[k] * b[k];
        s1 += a[k + 1] * b[k + 1];
        s2 += a[k + 2] * b[k + 2];
        s3 += a[k + 3] * b[k + 3];
    }
    float s = (s0 + s1) + (s2 + s3);
    for (; k < K; ++k) s += a[k] * b[k];
    return s;
}

inline void dot_1x4(const float* x, const float* w, int64_t K, float* out) {
    for (int j = 0; j < 4; ++j) out[j] = dot(x, w + j * K, K);
}

inline void dot_4x4(const float* x, const float* w, int64_t K, float* out, int64_t ldo) {
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) out[i * ldo + j] = dot(x + i * K, w + j * K, K);
}

#endif

// Computes rows [n0, n1) of the output for all M.
void matmul_cols(float* out, const float* x, const float* w, int64_t M, int64_t N, int64_t K, int64_t n0,
                 int64_t n1) {
    int64_t m = 0;
    for (; m + 4 <= M; m += 4) {
        int64_t n = n0;
        for (; n + 4 <= n1; n += 4) dot_4x4(x + m * K, w + n * K, K, out + m * N + n, N);
        for (; n < n1; ++n)
            for (int64_t i = 0; i < 4; ++i) out[(m + i) * N + n] = dot(x + (m + i) * K, w + n * K, K);
    }
    for (; m < M; ++m) {
        const float* xr = x + m * K;
        float* o = out + m * N;
        int64_t n = n0;
        for (; n + 4 <= n1; n += 4) dot_1x4(xr, w + n * K, K, o + n);
        for (; n < n1; ++n) o[n] = dot(xr, w + n * K, K);
    }
}

}  // namespace

void set_num_threads(int n) {
#if defined(_OPENMP)
    static const int default_threads = omp_get_max_threads();
    omp_set_num_threads(n > 0 ? n : default_threads);
#endif
    g_threads = n;
}

int num_threads() {
#if defined(_OPENMP)
    return g_threads > 0 ? g_threads : omp_get_max_threads();
#else
    return 1;
#endif
}

void matmul(float* out, const float* x, const float* w, int64_t M, int64_t N, int64_t K) {
    // Waking a thread costs microseconds, so small matmuls use fewer of them:
    // roughly one thread per MiB of weights, more when M amortizes the reads.
    const double work_mib = static_cast<double>(N) * K * 4 * std::max<int64_t>(1, M / 4) / (1 << 20);
    const int64_t threads = std::clamp<int64_t>(static_cast<int64_t>(work_mib), 1, num_threads());
    // Small chunks so faster cores pick up more work (Apple chips mix P and E cores).
    int64_t chunk = std::max<int64_t>(4, N / (threads * 8));
    chunk = (chunk + 3) / 4 * 4;
    const int64_t n_chunks = (N + chunk - 1) / chunk;
    if (threads == 1 || n_chunks == 1) {
        matmul_cols(out, x, w, M, N, K, 0, N);
        return;
    }
#pragma omp parallel for schedule(dynamic, 1) num_threads(static_cast<int>(threads))
    for (int64_t c = 0; c < n_chunks; ++c) {
        const int64_t n0 = c * chunk;
        matmul_cols(out, x, w, M, N, K, n0, std::min(N, n0 + chunk));
    }
}

}  // namespace tinyinfer
