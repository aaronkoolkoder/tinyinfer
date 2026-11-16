#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "model.h"
#include "sampling.h"

namespace tinyinfer {

struct GenerateOptions {
    int max_new_tokens = 32;
    std::vector<int32_t> stop_tokens;  // stop after emitting any of these
    bool use_cache = true;
};

struct GenerateStats {
    double prefill_seconds = 0;  // time to first token
    double decode_seconds = 0;   // time for the remaining tokens
    int64_t prompt_tokens = 0;
    int64_t generated = 0;
    size_t cache_bytes = 0;
};

// With use_cache, the prompt is prefilled once and each step decodes one token.
// Without it, every step reruns the full sequence.
// Returns only the new tokens. on_token, if set, is called as each one is chosen.
std::vector<int32_t> generate(const Model& model, const std::vector<int32_t>& prompt, const GenerateOptions& opts,
                              Sampler& sampler, const std::function<void(int32_t)>& on_token = {},
                              GenerateStats* stats = nullptr);

std::vector<int32_t> generate_greedy(const Model& model, const std::vector<int32_t>& prompt,
                                     const GenerateOptions& opts,
                                     const std::function<void(int32_t)>& on_token = {},
                                     GenerateStats* stats = nullptr);

int32_t argmax(const float* x, int64_t n);

}  // namespace tinyinfer
