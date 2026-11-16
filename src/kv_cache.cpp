#include "kv_cache.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace tinyinfer {

KVCache::KVCache(int64_t n_layer_, int64_t n_head_, int64_t head_size_, int64_t max_seq_)
    : n_layer(n_layer_), n_head(n_head_), head_size(head_size_), max_seq(max_seq_),
      k({n_layer_, n_head_, max_seq_, head_size_}), v({n_layer_, n_head_, max_seq_, head_size_}) {}

void kv_append(float* k_cache, float* v_cache, const float* qkv, int64_t pos, int64_t T_new, int64_t C,
               int64_t n_head, int64_t max_seq) {
    const int64_t hs = C / n_head;
    for (int64_t t = 0; t < T_new; ++t) {
        const float* k = qkv + t * 3 * C + C;
        const float* v = qkv + t * 3 * C + 2 * C;
        for (int64_t h = 0; h < n_head; ++h) {
            float* kd = k_cache + (h * max_seq + pos + t) * hs;
            float* vd = v_cache + (h * max_seq + pos + t) * hs;
            for (int64_t i = 0; i < hs; ++i) {
                kd[i] = k[h * hs + i];
                vd[i] = v[h * hs + i];
            }
        }
    }
}

void cached_attention(float* out, const float* qkv, const float* k_cache, const float* v_cache, int64_t pos,
                      int64_t T_new, int64_t C, int64_t n_head, int64_t max_seq) {
    const int64_t hs = C / n_head;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hs));
    std::vector<float> scores(pos + T_new);

    for (int64_t h = 0; h < n_head; ++h) {
        const float* kh = k_cache + h * max_seq * hs;
        const float* vh = v_cache + h * max_seq * hs;
        for (int64_t t = 0; t < T_new; ++t) {
            const float* q = qkv + t * 3 * C + h * hs;
            float* o = out + t * C + h * hs;
            const int64_t n_keys = pos + t + 1;

            float max_score = -std::numeric_limits<float>::infinity();
            for (int64_t s = 0; s < n_keys; ++s) {
                const float* k = kh + s * hs;
                float dot = 0.0f;
                for (int64_t i = 0; i < hs; ++i) dot += q[i] * k[i];
                scores[s] = dot * scale;
                max_score = std::max(max_score, scores[s]);
            }

            double sum = 0.0;
            for (int64_t s = 0; s < n_keys; ++s) {
                scores[s] = std::exp(scores[s] - max_score);
                sum += scores[s];
            }
            const float inv = static_cast<float>(1.0 / sum);

            for (int64_t i = 0; i < hs; ++i) o[i] = 0.0f;
            for (int64_t s = 0; s < n_keys; ++s) {
                const float* v = vh + s * hs;
                const float wgt = scores[s] * inv;
                for (int64_t i = 0; i < hs; ++i) o[i] += wgt * v[i];
            }
        }
    }
}

}  // namespace tinyinfer
