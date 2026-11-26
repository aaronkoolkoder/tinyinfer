#include <fstream>
#include <string>

#include "gpt2.h"
#include "ops.h"
#include "test.h"
#include "tinf.h"

using namespace tinyinfer;

static const std::string kDir = TINYINFER_MODEL_DIR;

static bool model_present() { return std::ifstream(kDir + "/model.tinf").good(); }

TEST(small_known_answer) {
    const float x[2 * 3] = {1, 2, 3,
                            4, 5, 6};
    const float w[4 * 3] = {1, 0, -1,
                            2, 1, 0,
                            0, 0, 1,
                            1, 1, 1};
    const float want[2 * 4] = {-2, 4, 3, 6,
                               -2, 13, 6, 15};
    float out[2 * 4];
    matmul(out, x, w, 2, 4, 3);
    for (int i = 0; i < 8; ++i) CHECK_NEAR(out[i], want[i], 1e-6);
}

TEST(overwrites_output) {
    const float x[2] = {1, 1};
    const float w[2] = {2, 3};
    float out[1] = {1000.0f};
    matmul(out, x, w, 1, 1, 2);
    CHECK_NEAR(out[0], 5.0, 1e-6);
}

TEST(one_hot_rows_select_columns) {
    const int64_t M = 7, N = 5, K = 11;
    Tensor x({M, K}), w({N, K}), out({M, N});
    fill_randn(x, 1);
    for (int64_t n = 0; n < N; ++n) w.f32()[n * K + (3 * n + 1) % K] = 1.0f;
    matmul(out, x, w);
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n)
            CHECK_NEAR(out.f32()[m * N + n], x.f32()[m * K + (3 * n + 1) % K], 0.0);
}

TEST(ones_input_sums_weight_rows) {
    const int64_t M = 3, N = 13, K = 37;
    Tensor x({M, K}), w({N, K}), out({M, N});
    fill(x, 1.0f);
    fill_randn(w, 2);
    matmul(out, x, w);
    for (int64_t n = 0; n < N; ++n) {
        double s = 0;
        for (int64_t k = 0; k < K; ++k) s += w.f32()[n * K + k];
        for (int64_t m = 0; m < M; ++m) CHECK_NEAR(out.f32()[m * N + n], s, 1e-4);
    }
}

// Shapes that hit every tail path of the tiled kernel.
TEST(fast_matches_naive_on_odd_shapes) {
    const int64_t shapes[][3] = {{1, 1, 1}, {1, 7, 3}, {3, 5, 17}, {4, 4, 4}, {5, 9, 33}, {13, 31, 70},
                                 {1, 4099, 129}, {128, 300, 96}, {7, 1000, 768}};
    for (const auto& sh : shapes) {
        const int64_t M = sh[0], N = sh[1], K = sh[2];
        Tensor x({M, K}), w({N, K}), fast({M, N}), ref({M, N});
        fill_randn(x, M * 31 + N);
        fill_randn(w, K * 17 + 3);
        fill(fast, 1e6f);
        matmul(fast.f32(), x.f32(), w.f32(), M, N, K);
        matmul_naive(ref.f32(), x.f32(), w.f32(), M, N, K);
        CHECK_NEAR(max_abs_diff(fast, ref), 0.0, 1e-3);
    }
}

TEST(thread_count_does_not_change_results) {
    const int64_t M = 9, N = 777, K = 300;
    Tensor x({M, K}), w({N, K}), a({M, N}), b({M, N});
    fill_randn(x, 1);
    fill_randn(w, 2);
    set_num_threads(1);
    matmul(a.f32(), x.f32(), w.f32(), M, N, K);
    set_num_threads(0);
    matmul(b.f32(), x.f32(), w.f32(), M, N, K);
    CHECK(max_abs_diff(a, b) == 0.0f);
}

// The final hidden state times wte^T is exactly GPT-2's lm_head, so this
// checks matmul against Hugging Face logits on real weights.
TEST(gpt2_lm_head_matches_reference) {
    if (!model_present()) return;
    GPT2Weights w = GPT2Weights::load(kDir + "/model.tinf");
    TinfFile ref = TinfFile::open(kDir + "/reference.tinf");
    for (int64_t i = 0; i < ref.config_int("n_prompts"); ++i) {
        const std::string p = "p" + std::to_string(i) + ".";
        Tensor hidden = ref.get(p + "hidden")[w.config.n_layer];
        Tensor want = ref.get(p + "logits");
        Tensor out({hidden.dim(0), w.config.n_vocab});
        matmul(out, hidden, w.wte);
        CHECK_NEAR(max_abs_diff(out, want), 0.0, 2e-3);
    }
}

TEST_MAIN()
