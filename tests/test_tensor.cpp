#include <cstdint>

#include "tensor.h"
#include "test.h"

using namespace tinyinfer;

TEST(alloc_is_zeroed_and_aligned) {
    Tensor t({3, 5});
    CHECK(t.numel() == 15);
    CHECK(t.nbytes() == 60);
    CHECK(reinterpret_cast<uintptr_t>(t.f32()) % 64 == 0);
    for (int64_t i = 0; i < t.numel(); ++i) CHECK(t.f32()[i] == 0.0f);
}

TEST(reshape_shares_storage) {
    Tensor t({2, 6});
    Tensor r = t.reshape({3, 4});
    r.f32()[7] = 42.0f;
    CHECK(t.f32()[7] == 42.0f);
    CHECK(r.dim(0) == 3 && r.dim(-1) == 4);
}

TEST(index_returns_row_view) {
    Tensor t({4, 3});
    for (int64_t i = 0; i < 12; ++i) t.f32()[i] = static_cast<float>(i);
    Tensor row = t[2];
    CHECK(row.ndim() == 1 && row.dim(0) == 3);
    CHECK(row.f32()[0] == 6.0f && row.f32()[2] == 8.0f);
    row.f32()[1] = -1.0f;
    CHECK(t.f32()[7] == -1.0f);
}

TEST(clone_is_independent) {
    Tensor t({4});
    fill(t, 1.0f);
    Tensor c = t.clone();
    c.f32()[0] = 5.0f;
    CHECK(t.f32()[0] == 1.0f);
    CHECK_NEAR(max_abs_diff(t, c), 4.0, 0.0);
}

TEST(randn_is_deterministic) {
    Tensor a({1000}), b({1000});
    fill_randn(a, 7);
    fill_randn(b, 7);
    CHECK(max_abs_diff(a, b) == 0.0f);
    double mean = 0;
    for (int64_t i = 0; i < a.numel(); ++i) mean += a.f32()[i];
    CHECK_NEAR(mean / a.numel(), 0.0, 0.1);
}

TEST(dtype_is_checked) {
    Tensor t({2}, DType::I32);
    bool threw = false;
    try {
        t.f32();
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

TEST_MAIN()
