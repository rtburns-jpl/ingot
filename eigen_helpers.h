#pragma once

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include <Eigen/Dense>

template<typename T>
using EMat = typename Eigen::Matrix<T,
      Eigen::Dynamic,
      Eigen::Dynamic,
      Eigen::RowMajor>;

template<typename T>
class StatevectorColumns {
    Eigen::Map<EMat<T>> map;
public:
    StatevectorColumns(T* ptr, int stride) : map{ptr, 6, stride} {}

    __device__ auto operator()(int i) {
        return map.col(i);
    }
};
