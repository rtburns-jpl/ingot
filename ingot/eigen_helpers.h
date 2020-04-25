template<typename T>
using EMat = typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>;

template<typename T>
using EArr = typename Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>;

// Convenience typedefs
template<typename T, int N>
using MatType = typename Eigen::Array<T, N, Eigen::Dynamic, Eigen::RowMajor>;
template<typename T, int N>
using ColVal = typename Eigen::Array<T, N, 1, Eigen::RowMajor>;
template<typename T, int N>
using ColRef = typename Eigen::Block<Eigen::Map<MatType<T, N>, 0>, N, 1>;

template<typename T, int N>
class ColIter :
    public thrust::iterator_facade<
            ColIter<T, N>, ColVal<T, N>, thrust::device_system_tag,
            thrust::forward_traversal_tag, ColRef<T, N>, Eigen::Index> {
public:
    CUDA_HOSTDEV
    ColIter(T* p, int s) : ptr{p}, stride{s} {}

    CUDA_HOSTDEV
    auto& advance(int i) {
        ptr += i;
        return *this;
    }

    CUDA_HOSTDEV
    auto& increment() { return advance(1); }

    // Can only dereference the value in cuda code!
    CUDA_DEV
    auto dereference() const {
        Eigen::Map<MatType<T, N>> map{ptr, N, stride};
        return map.col(0);
    }

    CUDA_HOSTDEV
    bool equal(const ColIter& rhs) const {
        return ptr == rhs.ptr and stride == rhs.stride;
    }

private:
    friend class thrust::iterator_core_access;
    T* ptr;
    int stride;
};

template<typename T, int N>
class DeviceColumnArray {
    using iterator = ColIter<T, N>;
    thrust::device_vector<T> data;
    int width;
public:
    DeviceColumnArray(int w) : width{w} {
        data.resize(N * width);
    }
    auto begin() { return ColIter<T, N>{data.data().get(), width}; }
    auto end() { return begin() + width; }
};
