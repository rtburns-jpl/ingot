// Convenience typedefs
template<typename T, int N>
using MatType = typename Eigen::Array<T, N, Eigen::Dynamic, Eigen::RowMajor>;
template<typename T, int N>
using ColVal = typename Eigen::Array<T, N, 1>;
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
class HostColIter :
    public thrust::iterator_facade<
            HostColIter<T, N>, ColVal<T, N>, thrust::host_system_tag,
            thrust::forward_traversal_tag, ColRef<T, N>, Eigen::Index> {
public:
    CUDA_HOSTDEV
    HostColIter(T* p, int s) : ptr{p}, stride{s} {}

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
    bool equal(const HostColIter& rhs) const {
        return ptr == rhs.ptr and stride == rhs.stride;
    }

private:
    friend class thrust::iterator_core_access;
    T* ptr;
    int stride;
};

template<typename T, int N>
class HostColumnArray;
template<typename T, int N>
class DeviceColumnArray {
    using iterator = ColIter<T, N>;
    thrust::device_vector<T> data;
    int width;

    friend class HostColumnArray<T, N>;
public:
    DeviceColumnArray(int w) : width{w} {
        data.resize(N * width);
    }
    auto begin() { return ColIter<T, N>{data.data().get(), width}; }
    auto end() { return begin() + width; }
};

template<typename T, int N>
class HostColumnArray {
    using iterator = HostColIter<T, N>;
    thrust::host_vector<T> data;
    int width;
public:
    HostColumnArray(int w) : width{w} {
        data.resize(N * width);
    }
    auto begin() { return HostColIter<T, N>{data.data().get(), width}; }
    auto end() { return begin() + width; }

    auto& operator=(DeviceColumnArray<T, N>& other) {
        if (width != other.width) {
            throw std::runtime_error("can only assign equal sized arrays");
        }
        return *this;
    }
};
