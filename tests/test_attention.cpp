#include <cmath>

#include "ops.h"
#include "test.h"

using namespace tinyinfer;

namespace {

// Pointers into row t of a [T, 3C] qkv buffer.
float* q_at(Tensor& qkv, int64_t t, int64_t C) { return qkv.f32() + t * 3 * C; }
float* k_at(Tensor& qkv, int64_t t, int64_t C) { return qkv.f32() + t * 3 * C + C; }
float* v_at(Tensor& qkv, int64_t t, int64_t C) { return qkv.f32() + t * 3 * C + 2 * C; }

Tensor run(const Tensor& qkv, int64_t T, int64_t C, int64_t n_head) {
    Tensor out({T, C});
    fill(out, 1e6f);
    causal_self_attention(out.f32(), qkv.f32(), T, C, n_head);
    return out;
}

}  // namespace

TEST(hand_computed_two_tokens) {
    const int64_t T = 2, C = 2;
    Tensor qkv({T, 3 * C});
    const float rows[2][6] = {{1, 0, 1, 0, 1, 2},     // q0 | k0 | v0
                              {0, 2, 0, 1, 3, -1}};   // q1 | k1 | v1
    for (int t = 0; t < 2; ++t)
        for (int i = 0; i < 6; ++i) qkv.f32()[t * 6 + i] = rows[t][i];
    Tensor out = run(qkv, T, C, 1);
    CHECK_NEAR(out.f32()[0], 1.0, 1e-5);
    CHECK_NEAR(out.f32()[1], 2.0, 1e-5);
    CHECK_NEAR(out.f32()[2], 2.6088593, 1e-5);
    CHECK_NEAR(out.f32()[3], -0.4132891, 1e-5);
}

TEST(single_token_returns_its_value) {
    const int64_t T = 1, C = 8;
    Tensor qkv({T, 3 * C});
    fill_randn(qkv, 3);
    Tensor out = run(qkv, T, C, 2);
    for (int64_t c = 0; c < C; ++c) CHECK_NEAR(out.f32()[c], v_at(qkv, 0, C)[c], 1e-6);
}

TEST(zero_queries_average_visible_values) {
    const int64_t T = 5, C = 6;
    Tensor qkv({T, 3 * C});
    fill_randn(qkv, 4);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t c = 0; c < C; ++c) q_at(qkv, t, C)[c] = 0.0f;
    Tensor out = run(qkv, T, C, 3);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t c = 0; c < C; ++c) {
            double mean = 0;
            for (int64_t s = 0; s <= t; ++s) mean += v_at(qkv, s, C)[c];
            CHECK_NEAR(out.f32()[t * C + c], mean / (t + 1), 1e-5);
        }
}

TEST(future_tokens_do_not_leak) {
    const int64_t T = 6, C = 8;
    Tensor qkv({T, 3 * C});
    fill_randn(qkv, 5);
    Tensor a = run(qkv, T, C, 2);
    for (int64_t i = 0; i < 3 * C; ++i) qkv.f32()[(T - 1) * 3 * C + i] += 5.0f;
    Tensor b = run(qkv, T, C, 2);
    for (int64_t i = 0; i < (T - 1) * C; ++i) CHECK(a.f32()[i] == b.f32()[i]);
    bool last_changed = false;
    for (int64_t c = 0; c < C; ++c) last_changed |= a.f32()[(T - 1) * C + c] != b.f32()[(T - 1) * C + c];
    CHECK(last_changed);
}

TEST(heads_are_independent) {
    const int64_t T = 4, C = 8, n_head = 2, hs = C / n_head;
    Tensor qkv({T, 3 * C});
    fill_randn(qkv, 6);
    Tensor a = run(qkv, T, C, n_head);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t c = hs; c < C; ++c) {
            q_at(qkv, t, C)[c] *= -3.0f;
            v_at(qkv, t, C)[c] = 0.0f;
        }
    Tensor b = run(qkv, T, C, n_head);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t c = 0; c < hs; ++c) CHECK(a.f32()[t * C + c] == b.f32()[t * C + c]);
        for (int64_t c = hs; c < C; ++c) CHECK_NEAR(b.f32()[t * C + c], 0.0, 0.0);
    }
}

TEST(scores_are_scaled_by_sqrt_head_size) {
    // One head, hs = 4: q.k = 4 for s=1 and 0 for s=0, so after scaling the
    // scores are {0, 2} and the weights are softmax({0, 2}).
    const int64_t T = 2, C = 4;
    Tensor qkv({T, 3 * C});
    for (int64_t c = 0; c < C; ++c) {
        q_at(qkv, 1, C)[c] = 1.0f;
        k_at(qkv, 1, C)[c] = 1.0f;
    }
    v_at(qkv, 0, C)[0] = 1.0f;
    v_at(qkv, 1, C)[0] = 0.0f;
    Tensor out = run(qkv, T, C, 1);
    const double p0 = 1.0 / (1.0 + std::exp(2.0));
    CHECK_NEAR(out.f32()[1 * C + 0], p0, 1e-5);
}

TEST(large_scores_stay_finite) {
    const int64_t T = 3, C = 4;
    Tensor qkv({T, 3 * C});
    for (int64_t t = 0; t < T; ++t)
        for (int64_t c = 0; c < C; ++c) {
            q_at(qkv, t, C)[c] = 100.0f;
            k_at(qkv, t, C)[c] = 100.0f * (t + 1);
            v_at(qkv, t, C)[c] = static_cast<float>(t * 10 + c);
        }
    Tensor out = run(qkv, T, C, 1);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t c = 0; c < C; ++c) {
            CHECK(std::isfinite(out.f32()[t * C + c]));
            CHECK_NEAR(out.f32()[t * C + c], v_at(qkv, t, C)[c], 1e-4);
        }
}

TEST_MAIN()
