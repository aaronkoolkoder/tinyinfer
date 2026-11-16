#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tensor.h"
#include "tinf.h"

namespace tinyinfer {

struct GPT2Config {
    int64_t n_vocab = 0;
    int64_t n_ctx = 0;
    int64_t n_embd = 0;
    int64_t n_head = 0;
    int64_t n_layer = 0;
    int64_t n_inner = 0;
    float eps = 1e-5f;
};

struct GPT2Layer {
    Tensor ln1_w, ln1_b;      // [C]
    Tensor attn_w, attn_b;    // [3C, C], [3C]
    Tensor proj_w, proj_b;    // [C, C], [C]
    Tensor ln2_w, ln2_b;      // [C]
    Tensor fc_w, fc_b;        // [4C, C], [4C]
    Tensor fc_proj_w, fc_proj_b;  // [C, 4C], [C]
};

struct GPT2Weights {
    GPT2Config config;
    Tensor wte;  // [V, C], also the lm_head
    Tensor wpe;  // [n_ctx, C]
    std::vector<GPT2Layer> layers;
    Tensor lnf_w, lnf_b;  // [C]

    static GPT2Weights load(const TinfFile& file);
    static GPT2Weights load(const std::string& path);
};

// Full forward pass over T tokens, returns logits [T, n_vocab].
// If hidden is given, it receives n_layer + 1 tensors [T, C] laid out like
// Hugging Face's hidden_states: the embeddings, the residual stream after
// blocks 0..n_layer-2, and the final ln_f output.
// With last_only, only the final position's logits are computed: [1, n_vocab].
Tensor gpt2_forward(const GPT2Weights& w, const int32_t* tokens, int64_t T, std::vector<Tensor>* hidden = nullptr,
                    bool last_only = false);

}  // namespace tinyinfer
