#include <cmath>
#include <string>

#include "gpt2.h"
#include "test.h"
#include "tinf.h"

using namespace tinyinfer;

static const std::string kDir = TINYINFER_MODEL_DIR;

TEST(model_file_parses) {
    TinfFile f = TinfFile::open(kDir + "/model.tinf");
    CHECK(f.config_int("n_vocab") == 50257);
    CHECK(f.config_int("n_layer") == 12);
    CHECK(f.names().size() == 2 + 12 * 12 + 2);
}

TEST(gpt2_weights_load_with_expected_shapes) {
    GPT2Weights w = GPT2Weights::load(kDir + "/model.tinf");
    CHECK(w.config.n_embd == 768 && w.config.n_head == 12);
    CHECK_NEAR(w.config.eps, 1e-5, 1e-12);
    CHECK(w.layers.size() == 12);
    CHECK(w.wte.shape_str() == "[50257, 768]");
    CHECK(w.layers[11].fc_proj_w.shape_str() == "[768, 3072]");
    float s = 0;
    for (int64_t i = 0; i < w.lnf_w.numel(); ++i) s += w.lnf_w.f32()[i];
    CHECK(std::isfinite(s) && s != 0.0f);
}

TEST(reference_file_parses) {
    TinfFile r = TinfFile::open(kDir + "/reference.tinf");
    const int64_t n = r.config_int("n_prompts");
    CHECK(n >= 1);
    for (int64_t i = 0; i < n; ++i) {
        const std::string p = "p" + std::to_string(i) + ".";
        Tensor tok = r.get(p + "tokens");
        CHECK(tok.dtype() == DType::I32);
        const int64_t T = tok.dim(0);
        CHECK(r.get(p + "logits").shape_str() == "[" + std::to_string(T) + ", 50257]");
        CHECK(r.get(p + "hidden").shape_str() == "[13, " + std::to_string(T) + ", 768]");
        CHECK(r.get(p + "greedy").dim(0) == r.config_int("gen_tokens"));
    }
}

TEST(missing_file_throws) {
    bool threw = false;
    try {
        TinfFile::open(kDir + "/nope.tinf");
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

TEST_MAIN()
