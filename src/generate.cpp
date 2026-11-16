#include "generate.h"

#include <stdexcept>

namespace tinyinfer {

int32_t argmax(const float* x, int64_t n) {
    int64_t best = 0;
    for (int64_t i = 1; i < n; ++i)
        if (x[i] > x[best]) best = i;
    return static_cast<int32_t>(best);
}

std::vector<int32_t> generate(const GPT2Weights& w, const std::vector<int32_t>& prompt, const GenerateOptions& opts,
                              Sampler& sampler, const std::function<void(int32_t)>& on_token) {
    if (prompt.empty()) throw std::invalid_argument("empty prompt");
    std::vector<int32_t> seq = prompt;
    std::vector<int32_t> out;
    for (int i = 0; i < opts.max_new_tokens && static_cast<int64_t>(seq.size()) < w.config.n_ctx; ++i) {
        Tensor logits = gpt2_forward(w, seq.data(), static_cast<int64_t>(seq.size()), nullptr, true);
        const int32_t next = sampler.sample(logits.f32(), w.config.n_vocab);
        seq.push_back(next);
        out.push_back(next);
        if (on_token) on_token(next);
        if (next == opts.stop_token) break;
    }
    return out;
}

std::vector<int32_t> generate_greedy(const GPT2Weights& w, const std::vector<int32_t>& prompt,
                                     const GenerateOptions& opts, const std::function<void(int32_t)>& on_token) {
    SamplingParams greedy;
    greedy.temperature = 0.0f;
    Sampler sampler(greedy);
    return generate(w, prompt, opts, sampler, on_token);
}

}  // namespace tinyinfer
