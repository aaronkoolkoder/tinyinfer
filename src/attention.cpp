#include "ops.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace tinyinfer {

void causal_self_attention(float* out, const float* qkv, int64_t T, int64_t C, int64_t n_head) {
    const int64_t hs = C / n_head;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hs));
    std::vector<float> scores(T);

    for (int64_t h = 0; h < n_head; ++h) {
        for (int64_t t = 0; t < T; ++t) {
            const float* q = qkv + t * 3 * C + h * hs;
            float* out_t = out + t * C + h * hs;
            float max_score = -std::numeric_limits<float>::infinity();

            for (int64_t s = 0; s <= t; ++s) {
                const float* k = qkv + s * 3 * C + C + h * hs;
                float dot = 0.0f;

                for (int64_t i = 0; i < hs; ++i) {
                    dot += q[i] * k[i];
                }

                scores[s] = dot * scale;
                max_score = std::max(max_score, scores[s]);
            }

            float sum = 0.0f;

            for (int64_t s = 0; s <= t; ++s) {
                scores[s] = std::exp(scores[s] - max_score);
                sum += scores[s];
            }

            for (int64_t s = 0; s <= t; ++s) {
                scores[s] /= sum;
            }

            for (int64_t i = 0; i < hs; ++i) {
                out_t[i] = 0.0f;
            }

            for (int64_t s = 0; s <= t; ++s) {
                const float* v = qkv + s * 3 * C + 2 * C + h * hs;

                for (int64_t i = 0; i < hs; ++i) {
                    out_t[i] += scores[s] * v[i];
                }
            }
        }
    }
}

}
