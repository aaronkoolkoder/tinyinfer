#pragma once

#include <cstdint>

#include "tensor.h"

namespace tinyinfer {

// Keys and values for every layer, laid out [n_layer, n_head, max_seq, hs] so
// that one head's history is contiguous in memory.
struct KVCache {
    int64_t n_layer = 0, n_head = 0, head_size = 0, max_seq = 0;
    int64_t len = 0;  // positions filled so far
    Tensor k, v;

    KVCache() = default;
    KVCache(int64_t n_layer, int64_t n_head, int64_t head_size, int64_t max_seq);

    float* k_layer(int64_t l) { return k.f32() + l * n_head * max_seq * head_size; }
    float* v_layer(int64_t l) { return v.f32() + l * n_head * max_seq * head_size; }
    size_t bytes() const { return k.nbytes() + v.nbytes(); }
    void clear() { len = 0; }
};

// Copy the keys and values of T_new new tokens into one layer of the cache.
//   qkv:      [T_new, 3C], rows laid out [q | k | v] as in causal_self_attention
//   k_cache, v_cache: this layer's [n_head, max_seq, hs] blocks
//   pos:      absolute position of the first new token (tokens already cached)
void kv_append(float* k_cache, float* v_cache, const float* qkv, int64_t pos, int64_t T_new, int64_t C,
               int64_t n_head, int64_t max_seq);

// Causal attention for T_new new queries against everything in the cache,
// including the new tokens (kv_append runs first).
//   out: [T_new, C]
void cached_attention(float* out, const float* qkv, const float* k_cache, const float* v_cache, int64_t pos,
                      int64_t T_new, int64_t C, int64_t n_head, int64_t max_seq);

}  // namespace tinyinfer
