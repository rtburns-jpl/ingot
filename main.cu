#include <thrust/device_vector.h>

#include "eigen_helpers.h"

struct operator_holder {
    template<typename T>
    __device__ void operator()(const T arg) {
        double sum = 0;
        for (int i = 0; i < 6; i++) {
            sum += arg[i];
        }
        printf("sum = %g\n", sum);
    }
};

int main() {
    // Allocate initial buffer
    const int nparticles = 5;
    thrust::device_vector<double> x(6 * nparticles);

    // Fill with 0...n
    thrust::sequence(x.begin(), x.end(), 0);

    // Create iterators
    Eigen::Map<EMat<double>> mapx{x.data().get(), 6, nparticles};
    eddi<double> begin{mapx, 0};
    auto end = begin + nparticles;

    // Test an elementwise function
    thrust::for_each(thrust::device,
                     begin, end,
                     operator_holder{});
}
