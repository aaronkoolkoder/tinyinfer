#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "generate.h"
#include "llama.h"
#include "ops.h"
#include "test.h"
#include "tinf.h"
#include "tokenizer.h"

using namespace tinyinfer;

namespace {

const std::string kDir = TINYINFER_LLAMA_DIR;

bool model_present() { return std::ifstream(kDir + "/model.tinf").good(); }

struct Fixture {
    LlamaModel model = LlamaModel::load(TinfFile::open(kDir + "/model.tinf"));
    Tokenizer tok = Tokenizer::load(kDir);
    TinfFile ref = TinfFile::open(kDir + "/reference.tinf");
};

Fixture& fixture() {
    static Fixture f;
    return f;
}

std::string as_text(const Tensor& bytes) {
    std::string s;
    for (int64_t i = 0; i < bytes.numel(); ++i) s += static_cast<char>(bytes.i32()[i]);
    return s;
}

std::vector<int32_t> as_ids(const Tensor& t) { return std::vector<int32_t>(t.i32(), t.i32() + t.numel()); }

float dot(const float* a, const float* b, int64_t n) {
    float s = 0;
    for (int64_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

}  // namespace

TEST(rope_position_zero_is_identity) {
    Tensor x({1, 2 * 8});
    fill_randn(x, 1);
    Tensor y = x.clone();
    rope(y.f32(), 1, 2, 8, 0, 10000.0f);
    CHECK(max_abs_diff(x, y) == 0.0f);
}

TEST(rope_preserves_norm_and_relative_position) {
    const int64_t D = 64;
    Tensor q({1, D}), k({1, D});
    fill_randn(q, 2);
    fill_randn(k, 3);
    auto rotated = [&](const Tensor& v, int64_t p) {
        Tensor r = v.clone();
        rope(r.f32(), 1, 1, D, p, 1e6f);
        return r;
    };
    Tensor q5 = rotated(q, 5);
    CHECK_NEAR(dot(q5.f32(), q5.f32(), D), dot(q.f32(), q.f32(), D), 1e-3);
    // q.k after rotation depends only on the distance between positions.
    const float a = dot(rotated(q, 7).f32(), rotated(k, 3).f32(), D);
    const float b = dot(rotated(q, 104).f32(), rotated(k, 100).f32(), D);
    CHECK_NEAR(a, b, 1e-3);
}

TEST(tokenizer_matches_hugging_face) {
    if (!model_present()) return;
    Fixture& f = fixture();
    for (int64_t i = 0; i < f.ref.config_int("n_tok_cases"); ++i) {
        const std::string p = "tok" + std::to_string(i) + ".";
        const std::string text = as_text(f.ref.get(p + "text"));
        const std::vector<int32_t> want = as_ids(f.ref.get(p + "ids"));
        const std::vector<int32_t> got = f.tok.encode(text);
        if (got != want) std::printf("  case %lld differs: \"%s\"\n", (long long)i, text.c_str());
        CHECK(got == want);
        CHECK(f.tok.decode(got) == text);
    }
    for (int64_t i = 0; i < f.ref.config_int("n_prompts"); ++i) {
        const std::string p = "p" + std::to_string(i) + ".";
        CHECK(f.tok.encode(as_text(f.ref.get(p + "text"))) == as_ids(f.ref.get(p + "tokens")));
    }
}

TEST(special_tokens_encode_as_single_ids) {
    if (!model_present()) return;
    Fixture& f = fixture();
    std::vector<int32_t> ids = f.tok.encode("<|im_start|>user\nhi<|im_end|>");
    CHECK(ids.front() == f.tok.special_id("<|im_start|>"));
    CHECK(ids.back() == f.tok.special_id("<|im_end|>"));
    CHECK(f.tok.decode(ids) == "<|im_start|>user\nhi<|im_end|>");
}

TEST(prefill_logits_match_reference) {
    if (!model_present()) return;
    Fixture& f = fixture();
    for (int64_t i = 0; i < f.ref.config_int("n_prompts"); ++i) {
        const std::string p = "p" + std::to_string(i) + ".";
        std::vector<int32_t> tok = as_ids(f.ref.get(p + "tokens"));
        KVCache cache = f.model.make_cache(64);
        Tensor logits = f.model.forward(cache, tok.data(), static_cast<int64_t>(tok.size()), false);
        const float err = max_abs_diff(logits, f.ref.get(p + "logits"));
        if (err > 2e-2f) std::printf("  prompt %lld: max logit error %g\n", (long long)i, err);
        CHECK(err <= 2e-2f);
    }
}

TEST(decode_matches_full_recompute) {
    if (!model_present()) return;
    Fixture& f = fixture();
    std::vector<int32_t> seq = as_ids(f.ref.get("p1.tokens"));
    std::vector<int32_t> extra = as_ids(f.ref.get("p1.greedy"));
    KVCache cache = f.model.make_cache(64);
    f.model.forward(cache, seq.data(), static_cast<int64_t>(seq.size()));
    for (int i = 0; i < 4; ++i) {
        seq.push_back(extra[i]);
        Tensor cached = f.model.forward(cache, &extra[i], 1);
        Tensor full = f.model.forward_uncached(seq.data(), static_cast<int64_t>(seq.size()));
        CHECK_NEAR(max_abs_diff(cached, full), 0.0, 1e-3);
    }
}

TEST(greedy_matches_hugging_face) {
    if (!model_present()) return;
    Fixture& f = fixture();
    for (int64_t i = 0; i < f.ref.config_int("n_prompts"); ++i) {
        const std::string p = "p" + std::to_string(i) + ".";
        std::vector<int32_t> want = as_ids(f.ref.get(p + "greedy"));
        GenerateOptions opts;
        opts.max_new_tokens = static_cast<int>(want.size());
        std::vector<int32_t> got = generate_greedy(f.model, as_ids(f.ref.get(p + "tokens")), opts);
        for (size_t t = 0; t < want.size() && t < got.size(); ++t)
            if (got[t] != want[t]) {
                std::printf("  prompt %lld: first mismatch at token %zu (%d vs %d)\n", (long long)i, t, got[t], want[t]);
                break;
            }
        CHECK(got == want);
    }
}

int main() {
    if (!model_present())
        std::printf("SKIP model tests: %s/model.tinf not found (run scripts/export.py --model "
                    "Qwen/Qwen2.5-0.5B-Instruct --out models/qwen2.5-0.5b)\n",
                    kDir.c_str());
    return test::run_all();
}
