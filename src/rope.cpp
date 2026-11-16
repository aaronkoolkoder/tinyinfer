#include <cmath>
#include <vector>

#include "ops.h"

namespace tinyinfer {

// Each pair (x[i], x[i + D/2]) is rotated by angle pos * theta^(-2i/D):
// low i spin fast and encode fine position, high i spin slowly.
void rope(float* x, int64_t T, int64_t n_heads, int64_t D, int64_t pos, float theta) {
    const int64_t half = D / 2;
    std::vector<float> inv_freq(half);
    for (int64_t i = 0; i < half; ++i)
        inv_freq[i] = static_cast<float>(1.0 / std::pow(static_cast<double>(theta), (2.0 * i) / D));
    std::vector<float> cos_t(half), sin_t(half);
    for (int64_t t = 0; t < T; ++t) {
        const float p = static_cast<float>(pos + t);
        for (int64_t i = 0; i < half; ++i) {
            const float a = p * inv_freq[i];
            cos_t[i] = std::cos(a);
            sin_t[i] = std::sin(a);
        }
        for (int64_t h = 0; h < n_heads; ++h) {
            float* r = x + (t * n_heads + h) * D;
            for (int64_t i = 0; i < half; ++i) {
                const float a = r[i], b = r[i + half];
                r[i] = a * cos_t[i] - b * sin_t[i];
                r[i + half] = b * cos_t[i] + a * sin_t[i];
            }
        }
    }
}

}  // namespace tinyinfer
