#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "gpt2.h"
#include "test.h"
#include "tinf.h"

using namespace tinyinfer;

namespace {

const std::string kDir = TINYINFER_MODEL_DIR;

float max_abs(const Tensor& t) {
    float m = 0;
    for (int64_t i = 0; i < t.numel(); ++i) m = std::max(m, std::fabs(t.f32()[i]));
    return m;
}

int64_t argmax(const float* x, int64_t n) { return std::max_element(x, x + n) - x; }

struct Fixture {
    GPT2Weights w = GPT2Weights::load(kDir + "/model.tinf");
    TinfFile ref = TinfFile::open(kDir + "/reference.tinf");
    int64_t n_prompts = ref.config_int("n_prompts");
};

Fixture& fixture() {
    static Fixture f;
    return f;
}

}  // namespace

TEST(embeddings_match_reference) {
    Fixture& f = fixture();
    for (int64_t i = 0; i < f.n_prompts; ++i) {
        const std::string p = "p" + std::to_string(i) + ".";
        Tensor tok = f.ref.get(p + "tokens");
        std::vector<Tensor> hidden;
        gpt2_forward(f.w, tok.i32(), tok.dim(0), &hidden);
        CHECK_NEAR(max_abs_diff(hidden[0], f.ref.get(p + "hidden")[0]), 0.0, 1e-6);
    }
}

// Reports the first block whose output diverges, which is where to look.
TEST(every_layer_matches_reference) {
    Fixture& f = fixture();
    for (int64_t i = 0; i < f.n_prompts; ++i) {
        const std::string p = "p" + std::to_string(i) + ".";
        Tensor tok = f.ref.get(p + "tokens");
        Tensor want = f.ref.get(p + "hidden");
        std::vector<Tensor> hidden;
        gpt2_forward(f.w, tok.i32(), tok.dim(0), &hidden);
        for (int64_t l = 0; l < want.dim(0); ++l) {
            Tensor ref_l = want[l];
            const float rel = max_abs_diff(hidden[l], ref_l) / std::max(1.0f, max_abs(ref_l));
            if (rel > 1e-4f) {
                std::printf("  prompt %lld: hidden[%lld] diverges (rel err %g)%s\n", (long long)i, (long long)l, rel,
                            l == 0 ? "" : ", check block l-1");
            }
            CHECK(rel <= 1e-4f);
        }
    }
}

TEST(logits_match_reference) {
    Fixture& f = fixture();
    for (int64_t i = 0; i < f.n_prompts; ++i) {
        const std::string p = "p" + std::to_string(i) + ".";
        Tensor tok = f.ref.get(p + "tokens");
        Tensor want = f.ref.get(p + "logits");
        Tensor logits = gpt2_forward(f.w, tok.i32(), tok.dim(0));
        CHECK_NEAR(max_abs_diff(logits, want), 0.0, 1e-2);
        const int64_t T = tok.dim(0), V = f.w.config.n_vocab;
        for (int64_t t = 0; t < T; ++t) CHECK(argmax(logits[t].f32(), V) == argmax(want[t].f32(), V));
    }
}

TEST_MAIN()
