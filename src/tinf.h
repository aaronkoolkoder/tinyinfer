#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tensor.h"

namespace tinyinfer {

// Memory-mapped .tinf file. Tensors returned by get() are zero-copy views
// that keep the mapping alive.
class TinfFile {
public:
    static TinfFile open(const std::string& path);

    const std::map<std::string, int64_t>& config() const { return config_; }
    int64_t config_int(const std::string& key) const;

    bool has(const std::string& name) const { return index_.count(name) > 0; }
    Tensor get(const std::string& name) const;
    std::vector<std::string> names() const;

private:
    struct Entry {
        DType dtype;
        std::vector<int64_t> shape;
        uint64_t offset;
        uint64_t nbytes;
    };

    std::string path_;
    std::shared_ptr<void> mapping_;
    size_t size_ = 0;
    std::map<std::string, int64_t> config_;
    std::map<std::string, Entry> index_;
};

}  // namespace tinyinfer
