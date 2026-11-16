#include <cmath>
#include <limits>
#include <vector>

#include "sampling.h"
#include "test.h"

using namespace tinyinfer;

TEST(rng_is_deterministic_and_in_range) {
    Rng a(42), b(42), c(43);
    bool differs = false;
    for (int i = 0; i < 1000; ++i) {
        float x = a.next_float();
        CHECK(x >= 0.0f && x < 1.0f);
        CHECK(x == b.next_float());
        differs |= x != c.next_float();
    }
    CHECK(differs);
}

TEST(softmax_is_stable_and_normalized) {
    float x[3] = {1000.0f, 1001.0f, 999.0f};
    softmax(x, 3);
    CHECK_NEAR(x[0] + x[1] + x[2], 1.0, 1e-6);
    CHECK_NEAR(x[1], 0.6652409, 1e-6);
}

TEST(temperature_scales_logits) {
    float x[2] = {2.0f, -4.0f};
    apply_temperature(x, 2, 0.5f);
    CHECK_NEAR(x[0], 4.0, 1e-6);
    CHECK_NEAR(x[1], -8.0, 1e-6);
}

TEST(top_k_keeps_k_largest) {
    float x[5] = {0.1f, 3.0f, -1.0f, 2.0f, 0.5f};
    top_k_filter(x, 5, 2);
    CHECK(x[1] == 3.0f && x[3] == 2.0f);
    CHECK(std::isinf(x[0]) && std::isinf(x[2]) && std::isinf(x[4]));
}

TEST(sample_index_inverts_cdf) {
    const float p[4] = {0.1f, 0.0f, 0.6f, 0.3f};
    CHECK(sample_index(p, 4, 0.0f) == 0);
    CHECK(sample_index(p, 4, 0.099f) == 0);
    CHECK(sample_index(p, 4, 0.1f) == 2);
    CHECK(sample_index(p, 4, 0.69f) == 2);
    CHECK(sample_index(p, 4, 0.71f) == 3);
    CHECK(sample_index(p, 4, 0.9999999f) == 3);
}

TEST(temperature_zero_is_greedy) {
    SamplingParams sp;
    sp.temperature = 0.0f;
    Sampler s(sp);
    const float logits[4] = {0.0f, 5.0f, 4.9f, -2.0f};
    for (int i = 0; i < 20; ++i) CHECK(s.sample(logits, 4) == 1);
}

TEST(same_seed_same_tokens) {
    SamplingParams sp;
    sp.seed = 7;
    Sampler a(sp), b(sp);
    const float logits[6] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    for (int i = 0; i < 200; ++i) CHECK(a.sample(logits, 6) == b.sample(logits, 6));
}

TEST(samples_follow_distribution) {
    SamplingParams sp;
    sp.seed = 1;
    Sampler s(sp);
    const float logits[3] = {std::log(0.2f), std::log(0.3f), std::log(0.5f)};
    int counts[3] = {0, 0, 0};
    const int N = 20000;
    for (int i = 0; i < N; ++i) ++counts[s.sample(logits, 3)];
    CHECK_NEAR(counts[0] / double(N), 0.2, 0.02);
    CHECK_NEAR(counts[1] / double(N), 0.3, 0.02);
    CHECK_NEAR(counts[2] / double(N), 0.5, 0.02);
}

TEST(top_k_one_is_greedy) {
    SamplingParams sp;
    sp.top_k = 1;
    sp.seed = 3;
    Sampler s(sp);
    const float logits[4] = {1.0f, 2.0f, 3.0f, 2.5f};
    for (int i = 0; i < 50; ++i) CHECK(s.sample(logits, 4) == 2);
}

TEST_MAIN()
