#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>

#include "eigen_helpers.h"

template<class... Ts>
auto zip_tuple_iters(Ts... ts) {
    return thrust::make_zip_iterator(thrust::make_tuple(ts...));
}

struct test_operator {
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

int main() {

    // Allocate initial buffer
    const int nparticles = 10;
    thrust::device_vector<double> x{6 * nparticles};

    // Fill with 0...n
    thrust::sequence(x.begin(), x.end());

    thrust::counting_iterator<int> index{0};
    thrust::transform_iterator<StatevectorColumns<double>, decltype(index)> iter{
        index,
        {x.data().get(), nparticles}
    };

    thrust::for_each(iter, iter + nparticles, test_operator{});
}
