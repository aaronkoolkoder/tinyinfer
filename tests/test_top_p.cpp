#include <algorithm>
#include <cmath>
#include <vector>

#include "sampling.h"
#include "tensor.h"
#include "test.h"

using namespace tinyinfer;

namespace {

void check_filter(std::vector<float> probs, float p, const std::vector<float>& want) {
    top_p_filter(probs.data(), static_cast<int64_t>(probs.size()), p);
    for (size_t i = 0; i < want.size(); ++i) CHECK_NEAR(probs[i], want[i], 1e-6);
}

std::vector<float> random_probs(int64_t n, uint64_t seed) {
    Tensor t({n});
    fill_randn(t, seed, 2.0f);
    std::vector<float> v(t.f32(), t.f32() + n);
    softmax(v.data(), n);
    return v;
}

}  // namespace

TEST(example_from_stub) { check_filter({0.15f, 0.50f, 0.05f, 0.30f}, 0.7f, {0.0f, 0.625f, 0.0f, 0.375f}); }

TEST(p_one_keeps_everything) { check_filter({0.15f, 0.50f, 0.05f, 0.30f}, 1.0f, {0.15f, 0.50f, 0.05f, 0.30f}); }

TEST(reaching_p_exactly_stops) { check_filter({0.15f, 0.50f, 0.05f, 0.30f}, 0.5f, {0.0f, 1.0f, 0.0f, 0.0f}); }

TEST(high_p_needs_every_token) { check_filter({0.15f, 0.50f, 0.05f, 0.30f}, 0.99f, {0.15f, 0.50f, 0.05f, 0.30f}); }

TEST(tiny_p_keeps_only_the_top_token) { check_filter({0.2f, 0.1f, 0.4f, 0.3f}, 1e-4f, {0.0f, 0.0f, 1.0f, 0.0f}); }

TEST(smallest_nucleus_on_random_distribution) {
    const int64_t n = 1000;
    const float p = 0.9f;
    const std::vector<float> orig = random_probs(n, 11);
    std::vector<float> probs = orig;
    top_p_filter(probs.data(), n, p);

    double kept_mass = 0.0, out_sum = 0.0;
    float min_kept = 1.0f, max_dropped = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        out_sum += probs[i];
        if (probs[i] > 0.0f) {
            kept_mass += orig[i];
            min_kept = std::min(min_kept, orig[i]);
        } else {
            max_dropped = std::max(max_dropped, orig[i]);
        }
    }
    CHECK_NEAR(out_sum, 1.0, 1e-5);
    CHECK(min_kept >= max_dropped);             // kept the most likely tokens
    CHECK(kept_mass >= p - 1e-6);               // reached p
    CHECK(kept_mass - min_kept < p);            // and no more than needed
    for (int64_t i = 0; i < n; ++i)
        if (probs[i] > 0.0f) CHECK_NEAR(probs[i], orig[i] / kept_mass, 1e-6);  // rescaled in place
}

TEST(handles_full_gpt2_vocab) {
    const int64_t n = 50257;
    std::vector<float> probs = random_probs(n, 12);
    top_p_filter(probs.data(), n, 0.95f);
    double sum = 0.0;
    int64_t kept = 0;
    for (float v : probs) {
        sum += v;
        kept += v > 0.0f;
    }
    CHECK_NEAR(sum, 1.0, 1e-4);
    CHECK(kept > 0 && kept < n);
}

TEST(sampler_never_leaves_the_nucleus) {
    SamplingParams sp;
    sp.top_p = 0.7f;
    sp.seed = 5;
    Sampler s(sp);
    // probs 0.5, 0.3, 0.15, 0.05 -> nucleus {0, 1}
    const float logits[4] = {std::log(0.5f), std::log(0.3f), std::log(0.15f), std::log(0.05f)};
    int counts[4] = {0, 0, 0, 0};
    for (int i = 0; i < 5000; ++i) ++counts[s.sample(logits, 4)];
    CHECK(counts[2] == 0 && counts[3] == 0);
    CHECK_NEAR(counts[0] / 5000.0, 0.625, 0.03);
}

TEST_MAIN()
