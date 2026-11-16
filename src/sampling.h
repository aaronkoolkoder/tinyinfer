#pragma once

#include <cstdint>
#include <vector>

namespace tinyinfer {

struct SamplingParams {
    float temperature = 1.0f;  // 0 means greedy
    int top_k = 0;             // 0 disables
    float top_p = 1.0f;        // 1 disables
    uint64_t seed = 0;
};

// Deterministic across platforms, unlike std::uniform_real_distribution.
class Rng {
public:
    explicit Rng(uint64_t seed) : state_(seed) {}
    uint64_t next_u64();
    float next_float();  // [0, 1)

private:
    uint64_t state_;
};

void apply_temperature(float* logits, int64_t n, float temperature);

// Sets every logit outside the k largest to -inf.
void top_k_filter(float* logits, int64_t n, int k);

// In place, numerically stable.
void softmax(float* x, int64_t n);

// probs: [n] summing to 1. Zeros everything outside the nucleus and renormalizes.
void top_p_filter(float* probs, int64_t n, float p);

// Inverse-CDF draw: smallest i with probs[0] + ... + probs[i] > u.
int32_t sample_index(const float* probs, int64_t n, float u);

class Sampler {
public:
    explicit Sampler(const SamplingParams& params) : params_(params), rng_(params.seed) {}

    // logits: [n]. Not modified.
    int32_t sample(const float* logits, int64_t n);

private:
    SamplingParams params_;
    Rng rng_;
    std::vector<float> buf_;
};

}  // namespace tinyinfer
