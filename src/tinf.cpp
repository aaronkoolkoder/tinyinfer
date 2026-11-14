#include "tinf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>

namespace tinyinfer {

namespace {

class Reader {
public:
    Reader(const std::byte* data, size_t size, const std::string& path) : data_(data), size_(size), path_(path) {}

    template <typename T>
    T read() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, data_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return v;
    }

    std::string read_string() {
        uint32_t n = read<uint32_t>();
        need(n);
        std::string s(reinterpret_cast<const char*>(data_ + pos_), n);
        pos_ += n;
        return s;
    }

private:
    void need(size_t n) {
        if (pos_ + n > size_) throw std::runtime_error(path_ + ": truncated header");
    }

    const std::byte* data_;
    size_t size_;
    size_t pos_ = 0;
    const std::string& path_;
};

}  // namespace

TinfFile TinfFile::open(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path + " (run scripts/export.py first)");
    struct stat st {};
    if (fstat(fd, &st) != 0) {
        ::close(fd);
        throw std::runtime_error("cannot stat " + path);
    }
    size_t size = static_cast<size_t>(st.st_size);
    void* addr = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (addr == MAP_FAILED) throw std::runtime_error("mmap failed for " + path);

    TinfFile f;
    f.path_ = path;
    f.size_ = size;
    f.mapping_ = std::shared_ptr<void>(addr, [size](void* p) { munmap(p, size); });

    const auto* bytes = static_cast<const std::byte*>(addr);
    Reader r(bytes, size, path);
    char magic[4];
    for (char& c : magic) c = r.read<char>();
    if (std::memcmp(magic, "TINF", 4) != 0) throw std::runtime_error(path + ": bad magic");
    uint32_t version = r.read<uint32_t>();
    if (version != 1) throw std::runtime_error(path + ": unsupported version " + std::to_string(version));

    uint32_t n_config = r.read<uint32_t>();
    for (uint32_t i = 0; i < n_config; ++i) {
        std::string key = r.read_string();
        f.config_[key] = r.read<int64_t>();
    }

    uint32_t n_tensors = r.read<uint32_t>();
    for (uint32_t i = 0; i < n_tensors; ++i) {
        std::string name = r.read_string();
        Entry e;
        uint32_t dtype = r.read<uint32_t>();
        if (dtype > 1) throw std::runtime_error(path + ": " + name + " has unknown dtype");
        e.dtype = static_cast<DType>(dtype);
        uint32_t ndim = r.read<uint32_t>();
        int64_t numel = 1;
        for (uint32_t d = 0; d < ndim; ++d) {
            e.shape.push_back(r.read<int64_t>());
            numel *= e.shape.back();
        }
        e.offset = r.read<uint64_t>();
        e.nbytes = r.read<uint64_t>();
        if (e.nbytes != static_cast<uint64_t>(numel) * dtype_size(e.dtype))
            throw std::runtime_error(path + ": " + name + " size does not match shape");
        if (e.offset + e.nbytes > size) throw std::runtime_error(path + ": " + name + " out of bounds");
        if (e.offset % 64 != 0) throw std::runtime_error(path + ": " + name + " is not 64-byte aligned");
        f.index_[name] = std::move(e);
    }
    return f;
}

int64_t TinfFile::config_int(const std::string& key) const {
    auto it = config_.find(key);
    if (it == config_.end()) throw std::runtime_error(path_ + ": missing config key " + key);
    return it->second;
}

Tensor TinfFile::get(const std::string& name) const {
    auto it = index_.find(name);
    if (it == index_.end()) throw std::runtime_error(path_ + ": missing tensor " + name);
    const Entry& e = it->second;
    auto* base = static_cast<std::byte*>(mapping_.get());
    return Tensor::view(base + e.offset, e.shape, e.dtype, mapping_);
}

std::vector<std::string> TinfFile::names() const {
    std::vector<std::string> out;
    for (const auto& kv : index_) out.push_back(kv.first);
    return out;
}

}  // namespace tinyinfer
