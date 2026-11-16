#include "llama.h"

#include <algorithm>
#include <cmath>
#include <limits>
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

// Stacks row-major [rows_i, C] matrices into one [sum rows_i, C] tensor.
Tensor stack_rows(std::initializer_list<const Tensor*> parts) {
    int64_t rows = 0, cols = (*parts.begin())->dim(-1);
    for (const Tensor* p : parts) rows += p->ndim() == 1 ? 1 : p->dim(0);
    const bool vector = (*parts.begin())->ndim() == 1;
    Tensor out(vector ? std::vector<int64_t>{0} : std::vector<int64_t>{rows, cols});
    if (vector) {
        int64_t n = 0;
        for (const Tensor* p : parts) n += p->numel();
        out = Tensor({n});
    }
    float* dst = out.f32();
    for (const Tensor* p : parts) dst = std::copy(p->f32(), p->f32() + p->numel(), dst);
    return out;
}

// k, v: [T, KV*D] -> cache [KV, max_seq, D] at positions pos..pos+T-1
void append_kv(float* k_cache, float* v_cache, const float* k, const float* v, int64_t pos, int64_t T,
               int64_t n_kv, int64_t D, int64_t max_seq) {
    for (int64_t t = 0; t < T; ++t)
        for (int64_t h = 0; h < n_kv; ++h) {
            const float* ks = k + t * n_kv * D + h * D;
            const float* vs = v + t * n_kv * D + h * D;
            std::copy(ks, ks + D, k_cache + (h * max_seq + pos + t) * D);
            std::copy(vs, vs + D, v_cache + (h * max_seq + pos + t) * D);
        }
}

}  // namespace

LlamaModel LlamaModel::load(const TinfFile& f) {
    LlamaModel m;
    LlamaConfig& c = m.config;
    c.n_vocab = f.config_int("n_vocab");
    c.n_ctx = f.config_int("n_ctx");
    c.n_embd = f.config_int("n_embd");
    c.n_head = f.config_int("n_head");
    c.n_kv_head = f.config_int("n_kv_head");
    c.head_dim = f.config_int("head_dim");
    c.n_layer = f.config_int("n_layer");
    c.n_inner = f.config_int("n_inner");
    c.rope_theta = static_cast<float>(f.config_int("rope_theta"));
    c.eps = static_cast<float>(f.config_int("rms_eps_x1e9")) * 1e-9f;
    c.attn_bias = f.config_int("attn_bias") != 0;
    if (c.n_head % c.n_kv_head != 0) throw std::runtime_error("n_head must be a multiple of n_kv_head");

    const int64_t C = c.n_embd, Q = c.n_head * c.head_dim, KV = c.n_kv_head * c.head_dim, F = c.n_inner;
    m.embed = take(f, "embed", {c.n_vocab, C});
    m.lm_head = f.config_int("tied_embeddings") ? m.embed : take(f, "lm_head", {c.n_vocab, C});
    for (int64_t i = 0; i < c.n_layer; ++i) {
        const std::string p = "l." + std::to_string(i) + ".";
        LlamaLayer L;
        L.attn_norm = take(f, p + "attn_norm", {C});
        L.wq = take(f, p + "wq", {Q, C});
        L.wk = take(f, p + "wk", {KV, C});
        L.wv = take(f, p + "wv", {KV, C});
        L.wo = take(f, p + "wo", {C, Q});
        if (c.attn_bias) {
            L.bq = take(f, p + "bq", {Q});
            L.bk = take(f, p + "bk", {KV});
            L.bv = take(f, p + "bv", {KV});
        }
        L.ffn_norm = take(f, p + "ffn_norm", {C});
        L.w_gate = take(f, p + "w_gate", {F, C});
        L.w_up = take(f, p + "w_up", {F, C});
        L.w_down = take(f, p + "w_down", {C, F});
        L.wqkv = stack_rows({&L.wq, &L.wk, &L.wv});
        if (c.attn_bias) L.bqkv = stack_rows({&L.bq, &L.bk, &L.bv});
        L.w_gate_up = stack_rows({&L.w_gate, &L.w_up});
        m.layers.push_back(std::move(L));
    }
    m.norm = take(f, "norm", {C});
    return m;
}

size_t LlamaModel::weight_bytes() const {
    size_t b = embed.nbytes() + norm.nbytes();
    if (lm_head.f32() != embed.f32()) b += lm_head.nbytes();
    for (const auto& L : layers)
        for (const Tensor* t : {&L.attn_norm, &L.wq, &L.wk, &L.wv, &L.wo, &L.bq, &L.bk, &L.bv, &L.ffn_norm,
                                &L.w_gate, &L.w_up, &L.w_down})
            if (!t->empty()) b += t->nbytes();
    return b;
}

KVCache LlamaModel::make_cache(int64_t max_seq) const {
    if (max_seq <= 0 || max_seq > config.n_ctx) max_seq = config.n_ctx;
    return KVCache(config.n_layer, config.n_kv_head, config.head_dim, max_seq);
}

Tensor LlamaModel::forward(KVCache& cache, const int32_t* tokens, int64_t T, bool last_only) const {
    const LlamaConfig& c = config;
    const int64_t C = c.n_embd, D = c.head_dim, H = c.n_head, KVH = c.n_kv_head;
    const int64_t Q = H * D, KV = KVH * D, F = c.n_inner, V = c.n_vocab;
    const int64_t pos = cache.len;
    if (T < 1 || pos + T > cache.max_seq) throw std::invalid_argument("KV cache is full");
    if (cache.n_layer != c.n_layer || cache.n_head != KVH || cache.head_size != D)
        throw std::invalid_argument("KV cache does not match model");
    for (int64_t t = 0; t < T; ++t)
        if (tokens[t] < 0 || tokens[t] >= V) throw std::invalid_argument("token id out of range");

    const int64_t QKV = Q + 2 * KV;
    Tensor x({T, C}), h({T, C}), qkv({T, QKV}), q({T, Q}), k({T, KV}), v({T, KV}), att({T, Q}), proj({T, C});
    Tensor gate_up({T, 2 * F}), act({T, F});

    for (int64_t t = 0; t < T; ++t) {
        const float* e = embed.f32() + static_cast<int64_t>(tokens[t]) * C;
        std::copy(e, e + C, x.f32() + t * C);
    }

    for (int64_t l = 0; l < c.n_layer; ++l) {
        const LlamaLayer& L = layers[l];
        rmsnorm(h.f32(), x.f32(), L.attn_norm.f32(), T, C, c.eps);
        linear(qkv.f32(), h.f32(), L.wqkv.f32(), c.attn_bias ? L.bqkv.f32() : nullptr, T, QKV, C);
        for (int64_t t = 0; t < T; ++t) {
            const float* row = qkv.f32() + t * QKV;
            std::copy(row, row + Q, q.f32() + t * Q);
            std::copy(row + Q, row + Q + KV, k.f32() + t * KV);
            std::copy(row + Q + KV, row + QKV, v.f32() + t * KV);
        }
        rope(q.f32(), T, H, D, pos, c.rope_theta);
        rope(k.f32(), T, KVH, D, pos, c.rope_theta);

        float* kc = cache.k_layer(l);
        float* vc = cache.v_layer(l);
        append_kv(kc, vc, k.f32(), v.f32(), pos, T, KVH, D, cache.max_seq);
        gqa_attention(att.f32(), q.f32(), kc, vc, pos, T, H, KVH, D, cache.max_seq);
        linear(proj.f32(), att.f32(), L.wo.f32(), nullptr, T, C, Q);
        add_inplace(x.f32(), proj.f32(), T * C);

        rmsnorm(h.f32(), x.f32(), L.ffn_norm.f32(), T, C, c.eps);
        linear(gate_up.f32(), h.f32(), L.w_gate_up.f32(), nullptr, T, 2 * F, C);
        for (int64_t t = 0; t < T; ++t) {
            const float* g = gate_up.f32() + t * 2 * F;
            float* a = act.f32() + t * F;
            std::copy(g, g + F, a);
            swiglu(a, g + F, F);
        }
        linear(proj.f32(), act.f32(), L.w_down.f32(), nullptr, T, C, F);
        add_inplace(x.f32(), proj.f32(), T * C);
    }
    cache.len += T;

    const int64_t rows = last_only ? 1 : T;
    rmsnorm(h.f32(), x.f32() + (T - rows) * C, norm.f32(), rows, C, c.eps);
    Tensor logits({rows, V});
    matmul(logits.f32(), h.f32(), lm_head.f32(), rows, V, C);
    return logits;
}

}  // namespace tinyinfer
