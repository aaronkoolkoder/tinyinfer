#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "gpt2.h"

namespace tinyinfer {

struct GenerateOptions {
    int max_new_tokens = 32;
    int32_t stop_token = -1;  // stop after emitting this token; -1 disables
};

// Greedy decoding without a KV cache: every step reruns the full sequence.
// Returns only the new tokens. on_token, if set, is called as each one is chosen.
std::vector<int32_t> generate_greedy(const GPT2Weights& w, const std::vector<int32_t>& prompt,
                                     const GenerateOptions& opts,
                                     const std::function<void(int32_t)>& on_token = {});

int32_t argmax(const float* x, int64_t n);

}  // namespace tinyinfer
