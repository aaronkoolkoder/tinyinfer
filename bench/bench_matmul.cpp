#include <chrono>
#include <cstdio>

#include "ops.h"
#include "tensor.h"

using namespace tinyinfer;

struct Shape {
    const char* name;
    int64_t M, N, K;
};

int main() {
    const Shape shapes[] = {
        {"decode  c_fc     1x768 @ 3072x768", 1, 3072, 768},
        {"decode  lm_head  1x768 @ 50257x768", 1, 50257, 768},
        {"prefill c_fc   128x768 @ 3072x768", 128, 3072, 768},
        {"prefill c_attn 128x768 @ 2304x768", 128, 2304, 768},
    };
    std::printf("%-38s %10s %10s\n", "shape", "ms", "GFLOP/s");
    for (const Shape& s : shapes) {
        Tensor x({s.M, s.K}), w({s.N, s.K}), out({s.M, s.N});
        fill_randn(x, 1);
        fill_randn(w, 2);
        matmul(out, x, w);
        int iters = 0;
        auto t0 = std::chrono::steady_clock::now();
        double elapsed = 0;
        while (elapsed < 0.5 || iters < 3) {
            matmul(out, x, w);
            ++iters;
            elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
        double ms = elapsed / iters * 1e3;
        double gflops = 2.0 * s.M * s.N * s.K / (ms * 1e-3) / 1e9;
        std::printf("%-38s %10.3f %10.2f\n", s.name, ms, gflops);
    }
}
