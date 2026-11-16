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
    GPT2Weights w = GPT2Weights::load(kDir + "/model.tinf");
    TinfFile ref = TinfFile::open(kDir + "/reference.tinf");
};

Fixture& fixture() {
    static Fixture f;
    return f;
}

// Without a KV cache every step reruns the whole sequence, so the default
// checks a prefix. Set TINYINFER_FULL=1 to check all reference tokens.
int tokens_to_check(int64_t available) {
    const char* full = std::getenv("TINYINFER_FULL");
    return static_cast<int>(full && std::string(full) == "1" ? available : std::min<int64_t>(available, 10));
}

}  // namespace

TEST(last_only_matches_last_row) {
    Fixture& f = fixture();
    Tensor tok = f.ref.get("p0.tokens");
    const int64_t T = tok.dim(0);
    Tensor all = gpt2_forward(f.w, tok.i32(), T);
    Tensor last = gpt2_forward(f.w, tok.i32(), T, nullptr, true);
    CHECK(last.shape_str() == "[1, 50257]");
    CHECK(max_abs_diff(last[0], all[T - 1]) == 0.0f);
}

TEST(greedy_matches_hugging_face) {
    Fixture& f = fixture();
    for (int64_t i = 0; i < f.ref.config_int("n_prompts"); ++i) {
        const std::string p = "p" + std::to_string(i) + ".";
        Tensor tok = f.ref.get(p + "tokens");
        Tensor want = f.ref.get(p + "greedy");
        GenerateOptions opts;
        opts.max_new_tokens = tokens_to_check(want.dim(0));
        std::vector<int32_t> prompt(tok.i32(), tok.i32() + tok.dim(0));
        std::vector<int32_t> got = generate_greedy(f.w, prompt, opts);
        CHECK(static_cast<int>(got.size()) == opts.max_new_tokens);
        for (size_t t = 0; t < got.size(); ++t) {
            if (got[t] != want.i32()[t]) std::printf("  prompt %lld: token %zu is %d, want %d\n", (long long)i, t, got[t], want.i32()[t]);
            CHECK(got[t] == want.i32()[t]);
        }
    }
}

TEST_MAIN()
