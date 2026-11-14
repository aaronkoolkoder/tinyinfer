#include "gpt2.h"

#include <stdexcept>

namespace tinyinfer {

namespace {

Tensor take(const TinfFile& f, const std::string& name, std::vector<int64_t> shape) {
    Tensor t = f.get(name);
    if (t.dtype() != DType::F32 || t.shape() != shape) {
        Tensor want = Tensor::view(nullptr, shape, DType::F32, nullptr);
        throw std::runtime_error(name + ": expected " + want.shape_str() + ", got " + t.shape_str());
    }
    return t;
}

}  // namespace

GPT2Weights GPT2Weights::load(const std::string& path) { return load(TinfFile::open(path)); }

GPT2Weights GPT2Weights::load(const TinfFile& f) {
    GPT2Weights w;
    GPT2Config& c = w.config;
    c.n_vocab = f.config_int("n_vocab");
    c.n_ctx = f.config_int("n_ctx");
    c.n_embd = f.config_int("n_embd");
    c.n_head = f.config_int("n_head");
    c.n_layer = f.config_int("n_layer");
    c.n_inner = f.config_int("n_inner");
    c.eps = static_cast<float>(f.config_int("layer_norm_eps_x1e9")) * 1e-9f;
    if (c.n_embd % c.n_head != 0) throw std::runtime_error("n_embd not divisible by n_head");

    const int64_t C = c.n_embd, H = c.n_inner;
    w.wte = take(f, "wte", {c.n_vocab, C});
    w.wpe = take(f, "wpe", {c.n_ctx, C});
    for (int64_t i = 0; i < c.n_layer; ++i) {
        const std::string p = "h." + std::to_string(i) + ".";
        GPT2Layer L;
        L.ln1_w = take(f, p + "ln_1.weight", {C});
        L.ln1_b = take(f, p + "ln_1.bias", {C});
        L.attn_w = take(f, p + "attn.c_attn.weight", {3 * C, C});
        L.attn_b = take(f, p + "attn.c_attn.bias", {3 * C});
        L.proj_w = take(f, p + "attn.c_proj.weight", {C, C});
        L.proj_b = take(f, p + "attn.c_proj.bias", {C});
        L.ln2_w = take(f, p + "ln_2.weight", {C});
        L.ln2_b = take(f, p + "ln_2.bias", {C});
        L.fc_w = take(f, p + "mlp.c_fc.weight", {H, C});
        L.fc_b = take(f, p + "mlp.c_fc.bias", {H});
        L.fc_proj_w = take(f, p + "mlp.c_proj.weight", {C, H});
        L.fc_proj_b = take(f, p + "mlp.c_proj.bias", {C});
        w.layers.push_back(std::move(L));
    }
    w.lnf_w = take(f, "ln_f.weight", {C});
    w.lnf_b = take(f, "ln_f.bias", {C});
    return w;
}

}  // namespace tinyinfer
