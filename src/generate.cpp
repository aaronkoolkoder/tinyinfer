#include "generate.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace tinyinfer {

int32_t argmax(const float* x, int64_t n) {
    int64_t best = 0;
    for (int64_t i = 1; i < n; ++i)
        if (x[i] > x[best]) best = i;
    return static_cast<int32_t>(best);
}

namespace {

double since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

std::vector<int32_t> generate(const Model& model, const std::vector<int32_t>& prompt, const GenerateOptions& opts,
                              Sampler& sampler, const std::function<void(int32_t)>& on_token, GenerateStats* stats) {
    if (prompt.empty()) throw std::invalid_argument("empty prompt");
    const int64_t V = model.n_vocab();
    const int64_t limit = std::min<int64_t>(model.n_ctx(), static_cast<int64_t>(prompt.size()) + opts.max_new_tokens);
    std::vector<int32_t> seq = prompt;
    std::vector<int32_t> out;
    GenerateStats st;
    st.prompt_tokens = static_cast<int64_t>(prompt.size());

    KVCache cache;
    if (opts.use_cache) cache = model.make_cache(limit);
    st.cache_bytes = cache.bytes();

    auto t0 = std::chrono::steady_clock::now();
    auto t_first = t0;
    Tensor logits;
    while (static_cast<int64_t>(seq.size()) < limit) {
        if (!opts.use_cache) {
            logits = model.forward_uncached(seq.data(), static_cast<int64_t>(seq.size()));
        } else if (cache.len == 0) {
            logits = model.forward(cache, seq.data(), static_cast<int64_t>(seq.size()));
        } else {
            logits = model.forward(cache, &seq.back(), 1);
        }
        const int32_t next = sampler.sample(logits.f32(), V);
        if (out.empty()) {
            t_first = std::chrono::steady_clock::now();
            st.prefill_seconds = since(t0);
        }
        seq.push_back(next);
        out.push_back(next);
        if (on_token) on_token(next);
        if (std::find(opts.stop_tokens.begin(), opts.stop_tokens.end(), next) != opts.stop_tokens.end()) break;
    }
    st.decode_seconds = out.empty() ? 0.0 : since(t_first);
    st.generated = static_cast<int64_t>(out.size());
    if (stats) *stats = st;
    return out;
}

std::vector<int32_t> generate_greedy(const Model& model, const std::vector<int32_t>& prompt,
                                     const GenerateOptions& opts, const std::function<void(int32_t)>& on_token,
                                     GenerateStats* stats) {
    SamplingParams greedy;
    greedy.temperature = 0.0f;
    Sampler sampler(greedy);
    return generate(model, prompt, opts, sampler, on_token, stats);
}

}  // namespace tinyinfer
