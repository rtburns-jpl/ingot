#include <thrust/device_vector.h>

#include "eigen_helpers.h"

template<class... Ts>
auto zip_tuple_iters(Ts... ts) {
    return thrust::make_zip_iterator(thrust::make_tuple(ts...));
}

struct test_transform {
    template<typename T>
    __device__ void operator()(T arg) {
        printf("%g %g %g %g %g %g\n",
               arg[0], arg[1], arg[2],
               arg[3], arg[4], arg[5]);
        arg[0] = 0;
        printf("%g %g %g %g %g %g\n",
               arg[0], arg[1], arg[2],
               arg[3], arg[4], arg[5]);
    }
};
struct test_transform_tuple {
    template<typename T>
    __device__ void operator()(T arg) {
        test_transform{}(thrust::get<1>(arg));
        auto& t = thrust::get<0>(arg);
        printf("t = %g\n", t);
        t = 10;
        printf("t = %g\n", t);
    }
};

int main() {
    using T = double;

    // Allocate initial buffer
    const int nparticles = 4;
    thrust::device_vector<T> x{6 * nparticles};

    // Fill with 0...n
    thrust::sequence(x.begin(), x.end());

    ColIter<T, 6> ci{x.data().get(), nparticles};

    thrust::device_vector<T> t{nparticles};
    thrust::fill(t.begin(), t.end(), 0);
    thrust::device_vector<T> h{nparticles};
    thrust::fill(h.begin(), h.end(), std::numeric_limits<T>::epsilon());

    auto zp = zip_tuple_iters(t.begin(), ci);

    thrust::for_each(zp, zp + nparticles, test_transform_tuple{});
    thrust::for_each(zp, zp + nparticles, test_transform_tuple{});
}
