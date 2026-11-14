#include "tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>

namespace tinyinfer {

namespace {

constexpr size_t kAlign = 64;

int64_t product(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) {
        if (d < 0) throw std::invalid_argument("negative dimension");
        n *= d;
    }
    return n;
}

void check_dtype(DType have, DType want) {
    if (have != want) throw std::runtime_error("tensor dtype mismatch");
}

}  // namespace

size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::F32: return 4;
        case DType::I32: return 4;
    }
    throw std::invalid_argument("unknown dtype");
}

Tensor::Tensor(std::vector<int64_t> shape, DType dtype)
    : shape_(std::move(shape)), numel_(product(shape_)), dtype_(dtype) {
    size_t bytes = std::max<size_t>(nbytes(), 1);
    bytes = (bytes + kAlign - 1) / kAlign * kAlign;
    void* p = std::aligned_alloc(kAlign, bytes);
    if (!p) throw std::bad_alloc();
    std::memset(p, 0, bytes);
    owner_ = std::shared_ptr<void>(p, std::free);
    data_ = static_cast<std::byte*>(p);
}

Tensor Tensor::view(void* data, std::vector<int64_t> shape, DType dtype, std::shared_ptr<void> owner) {
    Tensor t;
    t.shape_ = std::move(shape);
    t.numel_ = product(t.shape_);
    t.dtype_ = dtype;
    t.data_ = static_cast<std::byte*>(data);
    t.owner_ = std::move(owner);
    return t;
}

int64_t Tensor::dim(int i) const {
    if (i < 0) i += ndim();
    if (i < 0 || i >= ndim()) throw std::out_of_range("dim index");
    return shape_[i];
}

float* Tensor::f32() { check_dtype(dtype_, DType::F32); return reinterpret_cast<float*>(data_); }
const float* Tensor::f32() const { check_dtype(dtype_, DType::F32); return reinterpret_cast<const float*>(data_); }
int32_t* Tensor::i32() { check_dtype(dtype_, DType::I32); return reinterpret_cast<int32_t*>(data_); }
const int32_t* Tensor::i32() const { check_dtype(dtype_, DType::I32); return reinterpret_cast<const int32_t*>(data_); }

Tensor Tensor::reshape(std::vector<int64_t> shape) const {
    if (product(shape) != numel_) throw std::invalid_argument("reshape " + shape_str() + ": numel mismatch");
    return view(data_, std::move(shape), dtype_, owner_);
}

Tensor Tensor::operator[](int64_t i) const {
    if (ndim() == 0 || i < 0 || i >= shape_[0]) throw std::out_of_range("index " + std::to_string(i) + " into " + shape_str());
    std::vector<int64_t> sub(shape_.begin() + 1, shape_.end());
    size_t stride = static_cast<size_t>(product(sub)) * dtype_size(dtype_);
    return view(data_ + i * stride, std::move(sub), dtype_, owner_);
}

Tensor Tensor::clone() const {
    Tensor t(shape_, dtype_);
    std::memcpy(t.data_, data_, nbytes());
    return t;
}

std::string Tensor::shape_str() const {
    std::string s = "[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(shape_[i]);
    }
    return s + "]";
}

void fill(Tensor& t, float value) {
    std::fill(t.f32(), t.f32() + t.numel(), value);
}

void fill_randn(Tensor& t, uint64_t seed, float stddev) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, stddev);
    float* p = t.f32();
    for (int64_t i = 0; i < t.numel(); ++i) p[i] = dist(rng);
}

bool same_shape(const Tensor& a, const Tensor& b) { return a.shape() == b.shape(); }

float max_abs_diff(const Tensor& a, const Tensor& b) {
    if (!same_shape(a, b)) throw std::invalid_argument("shape mismatch " + a.shape_str() + " vs " + b.shape_str());
    const float* x = a.f32();
    const float* y = b.f32();
    float m = 0.0f;
    for (int64_t i = 0; i < a.numel(); ++i) m = std::max(m, std::fabs(x[i] - y[i]));
    return m;
}

}  // namespace tinyinfer
