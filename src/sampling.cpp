#include "sampling.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "generate.h"

namespace tinyinfer {

uint64_t Rng::next_u64() {
    // splitmix64
    uint64_t z = (state_ += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

float Rng::next_float() { return static_cast<float>(next_u64() >> 40) * (1.0f / 16777216.0f); }

void apply_temperature(float* logits, int64_t n, float temperature) {
    if (temperature <= 0.0f) throw std::invalid_argument("temperature must be > 0");
    const float inv = 1.0f / temperature;
    for (int64_t i = 0; i < n; ++i) logits[i] *= inv;
}

void top_k_filter(float* logits, int64_t n, int k) {
    if (k <= 0 || k >= n) return;
    std::vector<float> tmp(logits, logits + n);
    std::nth_element(tmp.begin(), tmp.begin() + (k - 1), tmp.end(), std::greater<float>());
    const float kth = tmp[k - 1];
    // Ties at the threshold are kept, matching Hugging Face.
    for (int64_t i = 0; i < n; ++i)
        if (logits[i] < kth) logits[i] = -std::numeric_limits<float>::infinity();
}

void softmax(float* x, int64_t n) {
    float mx = -std::numeric_limits<float>::infinity();
    for (int64_t i = 0; i < n; ++i) mx = std::max(mx, x[i]);
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - mx);
        sum += x[i];
    }
    const float inv = static_cast<float>(1.0 / sum);
    for (int64_t i = 0; i < n; ++i) x[i] *= inv;
}

int32_t sample_index(const float* probs, int64_t n, float u) {
    double cdf = 0.0;
    int32_t last_nonzero = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (probs[i] <= 0.0f) continue;
        cdf += probs[i];
        last_nonzero = static_cast<int32_t>(i);
        if (u < cdf) return last_nonzero;
    }
    return last_nonzero;  // rounding left cdf slightly below 1
}

int32_t Sampler::sample(const float* logits, int64_t n) {
    if (params_.temperature <= 0.0f) return argmax(logits, n);
    buf_.assign(logits, logits + n);
    apply_temperature(buf_.data(), n, params_.temperature);
    top_k_filter(buf_.data(), n, params_.top_k);
    softmax(buf_.data(), n);
    if (params_.top_p < 1.0f) top_p_filter(buf_.data(), n, params_.top_p);
    return sample_index(buf_.data(), n, rng_.next_float());
}

}  // namespace tinyinfer
