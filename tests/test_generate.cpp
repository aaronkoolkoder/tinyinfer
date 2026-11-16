#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "generate.h"
#include "gpt2.h"
#include "test.h"
#include "tinf.h"

using namespace tinyinfer;

namespace {

const std::string kDir = TINYINFER_MODEL_DIR;

struct Fixture {
    GPT2Model model{GPT2Weights::load(kDir + "/model.tinf")};
    const GPT2Weights& w = model.weights;
    TinfFile ref = TinfFile::open(kDir + "/reference.tinf");
};

Fixture& fixture() {
    static Fixture f;
    return f;
}

std::vector<int32_t> prompt_tokens(int64_t i) {
    Tensor tok = fixture().ref.get("p" + std::to_string(i) + ".tokens");
    return std::vector<int32_t>(tok.i32(), tok.i32() + tok.dim(0));
}

void check_greedy(bool use_cache, int64_t max_tokens) {
    Fixture& f = fixture();
    for (int64_t i = 0; i < f.ref.config_int("n_prompts"); ++i) {
        Tensor want = f.ref.get("p" + std::to_string(i) + ".greedy");
        GenerateOptions opts;
        opts.use_cache = use_cache;
        opts.max_new_tokens = static_cast<int>(std::min(want.dim(0), max_tokens));
        std::vector<int32_t> got = generate_greedy(f.model, prompt_tokens(i), opts);
        CHECK(static_cast<int>(got.size()) == opts.max_new_tokens);
        for (size_t t = 0; t < got.size(); ++t) {
            if (got[t] != want.i32()[t])
                std::printf("  prompt %lld: token %zu is %d, want %d\n", (long long)i, t, got[t], want.i32()[t]);
            CHECK(got[t] == want.i32()[t]);
        }
    }
}

}  // namespace

TEST(last_only_matches_last_row) {
    Fixture& f = fixture();
    std::vector<int32_t> tok = prompt_tokens(0);
    const int64_t T = static_cast<int64_t>(tok.size());
    Tensor all = gpt2_forward(f.w, tok.data(), T);
    Tensor last = gpt2_forward(f.w, tok.data(), T, nullptr, true);
    CHECK(last.shape_str() == "[1, 50257]");
    CHECK(max_abs_diff(last[0], all[T - 1]) == 0.0f);
}

TEST(cached_prefill_logits_match_reference) {
    Fixture& f = fixture();
    for (int64_t i = 0; i < f.ref.config_int("n_prompts"); ++i) {
        std::vector<int32_t> tok = prompt_tokens(i);
        KVCache cache = make_kv_cache(f.w.config, 64);
        Tensor logits = gpt2_forward_cached(f.w, cache, tok.data(), static_cast<int64_t>(tok.size()), false);
        CHECK(cache.len == static_cast<int64_t>(tok.size()));
        CHECK_NEAR(max_abs_diff(logits, f.ref.get("p" + std::to_string(i) + ".logits")), 0.0, 1e-2);
    }
}

TEST(cached_decode_matches_full_recompute) {
    Fixture& f = fixture();
    std::vector<int32_t> seq = prompt_tokens(2);
    const int32_t extra[] = {257, 7779, 287, 262, 3963};
    KVCache cache = make_kv_cache(f.w.config, 64);
    gpt2_forward_cached(f.w, cache, seq.data(), static_cast<int64_t>(seq.size()));
    for (int32_t id : extra) {
        seq.push_back(id);
        Tensor cached = gpt2_forward_cached(f.w, cache, &id, 1);
        Tensor full = gpt2_forward(f.w, seq.data(), static_cast<int64_t>(seq.size()), nullptr, true);
        CHECK_NEAR(max_abs_diff(cached, full), 0.0, 1e-3);
    }
}

// Without a cache every step reruns the whole sequence, so only a prefix is checked.
TEST(greedy_without_cache_matches_hugging_face) { check_greedy(false, 10); }

TEST(greedy_with_cache_matches_hugging_face) { check_greedy(true, 1 << 30); }

TEST_MAIN()
