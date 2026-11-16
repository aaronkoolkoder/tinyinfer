#include "model.h"

#include <stdexcept>

#include "gpt2.h"
#include "llama.h"
#include "tinf.h"

namespace tinyinfer {

Tensor Model::forward_uncached(const int32_t* tokens, int64_t T) const {
    KVCache cache = make_cache(T);
    return forward(cache, tokens, T, true);
}

size_t GPT2Model::weight_bytes() const {
    size_t b = weights.wte.nbytes() + weights.wpe.nbytes() + weights.lnf_w.nbytes() + weights.lnf_b.nbytes();
    for (const auto& L : weights.layers)
        for (const Tensor* t : {&L.ln1_w, &L.ln1_b, &L.attn_w, &L.attn_b, &L.proj_w, &L.proj_b, &L.ln2_w, &L.ln2_b,
                                &L.fc_w, &L.fc_b, &L.fc_proj_w, &L.fc_proj_b})
            b += t->nbytes();
    return b;
}

std::unique_ptr<Model> Model::load(const std::string& dir) {
    TinfFile f = TinfFile::open(dir + "/model.tinf");
    const int64_t arch = f.config().count("arch") ? f.config_int("arch") : 1;
    if (arch == 1) return std::make_unique<GPT2Model>(GPT2Weights::load(f));
    if (arch == 2) return std::make_unique<LlamaModel>(LlamaModel::load(f));
    throw std::runtime_error(dir + ": unknown arch " + std::to_string(arch));
}

}  // namespace tinyinfer
