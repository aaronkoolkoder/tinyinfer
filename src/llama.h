#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "model.h"
#include "tensor.h"
#include "tinf.h"

namespace tinyinfer {

struct LlamaConfig {
    int64_t n_vocab = 0, n_ctx = 0, n_embd = 0;
    int64_t n_head = 0, n_kv_head = 0, head_dim = 0;
    int64_t n_layer = 0, n_inner = 0;
    float rope_theta = 10000.0f;
    float eps = 1e-6f;
    bool attn_bias = false;
};

struct LlamaLayer {
    Tensor attn_norm;            // [C]
    Tensor wq, wk, wv, wo;       // [H*D, C], [KV*D, C], [KV*D, C], [C, H*D]
    Tensor bq, bk, bv;           // optional (Qwen2)
    Tensor ffn_norm;             // [C]
    Tensor w_gate, w_up, w_down; // [F, C], [F, C], [C, F]
    Tensor wqkv, bqkv;           // [H*D + 2*KV*D, C]: q, k, v fused into one matmul
    Tensor w_gate_up;            // [2F, C]: gate and up fused
};

// Llama-style decoder: RMSNorm, RoPE, grouped-query attention, SwiGLU MLP.
class LlamaModel : public Model {
public:
    static LlamaModel load(const TinfFile& f);

    const char* arch_name() const override { return "llama"; }
    int64_t n_vocab() const override { return config.n_vocab; }
    int64_t n_ctx() const override { return config.n_ctx; }
    size_t weight_bytes() const override;
    KVCache make_cache(int64_t max_seq) const override;
    Tensor forward(KVCache& cache, const int32_t* tokens, int64_t T, bool last_only = true) const override;

    LlamaConfig config;
    Tensor embed;    // [V, C]
    Tensor lm_head;  // [V, C], same storage as embed when tied
    std::vector<LlamaLayer> layers;
    Tensor norm;     // [C]
};

}  // namespace tinyinfer
