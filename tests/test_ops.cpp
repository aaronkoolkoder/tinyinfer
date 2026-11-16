#include "ops.h"
#include "test.h"

using namespace tinyinfer;

TEST(gelu_matches_hugging_face) {
    float x[5] = {1, -1, 0, 3, -3};
    const float want[5] = {0.8411920f, -0.1588080f, 0.0f, 2.9963627f, -0.0036374f};
    gelu(x, 5);
    for (int i = 0; i < 5; ++i) CHECK_NEAR(x[i], want[i], 1e-6);
}

TEST(layernorm_known_answer) {
    const float x[4] = {1, 2, 3, 4};
    const float g[4] = {2, 2, 2, 2};
    const float b[4] = {1, 1, 1, 1};
    const float want[4] = {-1.6832814f, 0.1055728f, 1.8944272f, 3.6832814f};
    float out[4];
    layernorm(out, x, g, b, 1, 4, 0.0f);
    for (int i = 0; i < 4; ++i) CHECK_NEAR(out[i], want[i], 1e-5);
}

TEST(layernorm_rows_are_independent_and_alias_safe) {
    float x[2 * 3] = {1, 2, 3, 10, 10, 40};
    const float g[3] = {1, 1, 1}, b[3] = {0, 0, 0};
    layernorm(x, x, g, b, 2, 3, 1e-5f);
    for (int r = 0; r < 2; ++r) {
        double mean = 0, var = 0;
        for (int c = 0; c < 3; ++c) mean += x[r * 3 + c];
        mean /= 3;
        for (int c = 0; c < 3; ++c) var += (x[r * 3 + c] - mean) * (x[r * 3 + c] - mean);
        CHECK_NEAR(mean, 0.0, 1e-6);
        CHECK_NEAR(var / 3, 1.0, 1e-4);
    }
}

TEST(embed_adds_token_and_position) {
    const float wte[3 * 2] = {0, 0, 10, 20, 100, 200};
    const float wpe[2 * 2] = {1, 2, 3, 4};
    const int32_t tok[2] = {2, 1};
    float out[4];
    embed(out, wte, wpe, tok, 2, 2);
    CHECK(out[0] == 101 && out[1] == 202 && out[2] == 13 && out[3] == 24);
}

TEST(linear_adds_bias) {
    const float x[2] = {1, 2};
    const float w[2 * 2] = {1, 1, 0, 1};
    const float b[2] = {10, -10};
    float out[2];
    linear(out, x, w, b, 1, 2, 2);
    CHECK_NEAR(out[0], 13.0, 1e-6);
    CHECK_NEAR(out[1], -8.0, 1e-6);
}

TEST_MAIN()
