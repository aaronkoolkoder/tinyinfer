#include <vector>

#include "kv_cache.h"
#include "ops.h"
#include "test.h"

using namespace tinyinfer;

namespace {

constexpr int64_t C = 8, n_head = 2, hs = C / n_head, max_seq = 10;
constexpr float kSentinel = -777.0f;

KVCache make_cache() {
    KVCache c(1, n_head, hs, max_seq);
    fill(c.k, kSentinel);
    fill(c.v, kSentinel);
    return c;
}

// qkv where every element encodes (token, part, channel) so misplaced copies are visible.
Tensor coded_qkv(int64_t T, int64_t first_token) {
    Tensor qkv({T, 3 * C});
    for (int64_t t = 0; t < T; ++t)
        for (int64_t i = 0; i < 3 * C; ++i) qkv.f32()[t * 3 * C + i] = static_cast<float>((first_token + t) * 1000 + i);
    return qkv;
}

float cache_at(KVCache& c, const Tensor& which, int64_t h, int64_t p, int64_t i) {
    (void)c;
    return which.f32()[(h * max_seq + p) * hs + i];
}

// Runs cached attention over a random sequence fed in chunks and compares
// against the uncached causal_self_attention.
void check_chunks(const std::vector<int64_t>& chunks, uint64_t seed) {
    int64_t T = 0;
    for (int64_t n : chunks) T += n;
    Tensor qkv({T, 3 * C});
    fill_randn(qkv, seed);
    Tensor want({T, C});
    causal_self_attention(want.f32(), qkv.f32(), T, C, n_head);

    KVCache cache = make_cache();
    Tensor got({T, C});
    fill(got, 1e6f);
    int64_t pos = 0;
    for (int64_t n : chunks) {
        const float* q = qkv.f32() + pos * 3 * C;
        kv_append(cache.k_layer(0), cache.v_layer(0), q, pos, n, C, n_head, max_seq);
        cached_attention(got.f32() + pos * C, q, cache.k_layer(0), cache.v_layer(0), pos, n, C, n_head, max_seq);
        pos += n;
    }
    CHECK_NEAR(max_abs_diff(got, want), 0.0, 1e-5);
}

}  // namespace

TEST(append_writes_keys_and_values_per_head) {
    KVCache c = make_cache();
    Tensor qkv = coded_qkv(2, 0);
    kv_append(c.k_layer(0), c.v_layer(0), qkv.f32(), 0, 2, C, n_head, max_seq);
    for (int64_t t = 0; t < 2; ++t)
        for (int64_t h = 0; h < n_head; ++h)
            for (int64_t i = 0; i < hs; ++i) {
                CHECK(cache_at(c, c.k, h, t, i) == qkv.f32()[t * 3 * C + C + h * hs + i]);
                CHECK(cache_at(c, c.v, h, t, i) == qkv.f32()[t * 3 * C + 2 * C + h * hs + i]);
            }
}

TEST(append_at_offset_touches_nothing_else) {
    KVCache c = make_cache();
    Tensor qkv = coded_qkv(3, 4);
    kv_append(c.k_layer(0), c.v_layer(0), qkv.f32(), 4, 3, C, n_head, max_seq);
    for (int64_t h = 0; h < n_head; ++h)
        for (int64_t p = 0; p < max_seq; ++p)
            for (int64_t i = 0; i < hs; ++i) {
                const bool filled = p >= 4 && p < 7;
                const int64_t t = p - 4;
                const float k_want = filled ? qkv.f32()[t * 3 * C + C + h * hs + i] : kSentinel;
                const float v_want = filled ? qkv.f32()[t * 3 * C + 2 * C + h * hs + i] : kSentinel;
                CHECK(cache_at(c, c.k, h, p, i) == k_want);
                CHECK(cache_at(c, c.v, h, p, i) == v_want);
            }
}

TEST(prefill_matches_uncached_attention) { check_chunks({7}, 21); }

TEST(one_token_at_a_time_matches_uncached_attention) { check_chunks({1, 1, 1, 1, 1, 1, 1}, 22); }

TEST(prefill_then_decode_matches_uncached_attention) { check_chunks({4, 1, 1, 1}, 23); }

TEST(chunks_after_prefill_stay_causal) { check_chunks({3, 2, 4}, 24); }

TEST(decode_ignores_stale_cache_slots) {
    // Slots past the current position hold garbage (sentinels); they must not be read.
    check_chunks({2, 1}, 25);
}

TEST_MAIN()
