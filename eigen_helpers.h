#pragma once

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include <Eigen/Dense>

template<typename T>
using EMat = typename Eigen::Matrix<T,
      Eigen::Dynamic,
      Eigen::Dynamic,
      Eigen::RowMajor>;

// Eigen Dynamic Device Iterator
template<typename T>
class eddi {
    using map_t = typename Eigen::Map<EMat<T>>;
    using block_t = typename Eigen::Block<map_t, Eigen::Dynamic, 1>;
public:
    __host__ __device__
    eddi(map_t m, int i) : map{m}, idx{i} {}
    __host__ __device__
    auto operator+(int offset) const {
        return eddi{map, idx + offset};
    }
    __host__ __device__
    auto operator+=(int offset) {
        idx += offset;
        return *this;
    }
    __host__ __device__
    auto operator++() { return *this += 1; }

    __host__ __device__
    auto operator==(const eddi& other) const {
        return
            //map == other.map && TODO compare pointer
            idx == other.idx;
    }
    __host__ __device__
    auto operator!=(const eddi& other) const {
        return not (*this == other);
    }
    __device__ auto operator[](int i)       { return map.col(idx + i); }
    __device__ auto operator[](int i) const { return map.col(idx + i); }

    using difference_type = int;
    using value_type = block_t;
    using pointer = block_t;
    using reference = block_t;
    using iterator_category = std::forward_iterator_tag;

private:
    map_t map;
    int idx;
};
