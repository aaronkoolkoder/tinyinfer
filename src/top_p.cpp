#include "sampling.h"

#include <algorithm>
#include <numeric>
#include <vector>

namespace tinyinfer {

void top_p_filter(float* probs, int64_t n, float p) {
    if (n <= 0) return;

    std::vector<int64_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(),
        [&](int64_t a, int64_t b) {
            return probs[a] > probs[b];
        });

    double cumulative = 0.0;
    int64_t keep = 0;

    for (int64_t i = 0; i < n; ++i) {
        cumulative += probs[indices[i]];
        ++keep;
        if (cumulative >= p) break;
    }

    for (int64_t i = keep; i < n; ++i) {
        probs[indices[i]] = 0.0f;
    }

    const float inv = static_cast<float>(1.0 / cumulative);
    for (int64_t i = 0; i < keep; ++i) {
        probs[indices[i]] *= inv;
    }
}

}  // namespace tinyinfer
