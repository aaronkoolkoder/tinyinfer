#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "kv_cache.h"
#include "tensor.h"

namespace tinyinfer {

// What generation needs from any architecture.
class Model {
public:
    virtual ~Model() = default;

    virtual const char* arch_name() const = 0;
    virtual int64_t n_vocab() const = 0;
    virtual int64_t n_ctx() const = 0;
    virtual size_t weight_bytes() const = 0;

    virtual KVCache make_cache(int64_t max_seq) const = 0;

    // Runs T new tokens at positions cache.len.., appends their K/V, advances cache.len.
    // Returns logits [T, n_vocab], or [1, n_vocab] with last_only.
    virtual Tensor forward(KVCache& cache, const int32_t* tokens, int64_t T, bool last_only = true) const = 0;

    // Full recompute with no persistent cache.
    virtual Tensor forward_uncached(const int32_t* tokens, int64_t T) const;

    // Reads the arch key from <dir>/model.tinf.
    static std::unique_ptr<Model> load(const std::string& dir);
};

}  // namespace tinyinfer
