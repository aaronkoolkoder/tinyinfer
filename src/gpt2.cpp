#include "gpt2.h"

#include <stdexcept>

#include "ops.h"

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

Tensor gpt2_forward(const GPT2Weights& w, const int32_t* tokens, int64_t T, std::vector<Tensor>* hidden) {
    const GPT2Config& c = w.config;
    const int64_t C = c.n_embd, H = c.n_inner, V = c.n_vocab;
    if (T < 1 || T > c.n_ctx) throw std::invalid_argument("sequence length out of range");
    for (int64_t t = 0; t < T; ++t)
        if (tokens[t] < 0 || tokens[t] >= V) throw std::invalid_argument("token id out of range");

    Tensor x({T, C}), h({T, C}), qkv({T, 3 * C}), att({T, C}), proj({T, C}), fc({T, H});
    if (hidden) hidden->clear();

    embed(x.f32(), w.wte.f32(), w.wpe.f32(), tokens, T, C);
    for (int64_t l = 0; l < c.n_layer; ++l) {
        if (hidden) hidden->push_back(x.clone());
        const GPT2Layer& L = w.layers[l];

        layernorm(h.f32(), x.f32(), L.ln1_w.f32(), L.ln1_b.f32(), T, C, c.eps);
        linear(qkv.f32(), h.f32(), L.attn_w.f32(), L.attn_b.f32(), T, 3 * C, C);
        causal_self_attention(att.f32(), qkv.f32(), T, C, c.n_head);
        linear(proj.f32(), att.f32(), L.proj_w.f32(), L.proj_b.f32(), T, C, C);
        add_inplace(x.f32(), proj.f32(), T * C);

        layernorm(h.f32(), x.f32(), L.ln2_w.f32(), L.ln2_b.f32(), T, C, c.eps);
        linear(fc.f32(), h.f32(), L.fc_w.f32(), L.fc_b.f32(), T, H, C);
        gelu(fc.f32(), T * H);
        linear(proj.f32(), fc.f32(), L.fc_proj_w.f32(), L.fc_proj_b.f32(), T, C, H);
        add_inplace(x.f32(), proj.f32(), T * C);
    }
    layernorm(x.f32(), x.f32(), w.lnf_w.f32(), w.lnf_b.f32(), T, C, c.eps);
    if (hidden) hidden->push_back(x.clone());

    Tensor logits({T, V});
    matmul(logits.f32(), x.f32(), w.wte.f32(), T, V, C);
    return logits;
}

}  // namespace tinyinfer
