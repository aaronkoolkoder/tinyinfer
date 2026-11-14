#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace tinyinfer {

enum class DType : uint32_t { F32 = 0, I32 = 1 };

size_t dtype_size(DType dt);

// Row-major, contiguous. Copies share storage; use clone() for a deep copy.
class Tensor {
public:
    Tensor() = default;
    explicit Tensor(std::vector<int64_t> shape, DType dtype = DType::F32);

    static Tensor view(void* data, std::vector<int64_t> shape, DType dtype, std::shared_ptr<void> owner);

    const std::vector<int64_t>& shape() const { return shape_; }
    int64_t dim(int i) const;
    int ndim() const { return static_cast<int>(shape_.size()); }
    int64_t numel() const { return numel_; }
    size_t nbytes() const { return static_cast<size_t>(numel_) * dtype_size(dtype_); }
    DType dtype() const { return dtype_; }
    bool empty() const { return data_ == nullptr; }

    float* f32();
    const float* f32() const;
    int32_t* i32();
    const int32_t* i32() const;

    Tensor reshape(std::vector<int64_t> shape) const;
    Tensor operator[](int64_t i) const;  // view of index i along dim 0
    Tensor clone() const;

    std::string shape_str() const;

private:
    std::vector<int64_t> shape_;
    int64_t numel_ = 0;
    DType dtype_ = DType::F32;
    std::byte* data_ = nullptr;
    std::shared_ptr<void> owner_;
};

void fill(Tensor& t, float value);
void fill_randn(Tensor& t, uint64_t seed, float stddev = 1.0f);
float max_abs_diff(const Tensor& a, const Tensor& b);
bool same_shape(const Tensor& a, const Tensor& b);

}  // namespace tinyinfer
