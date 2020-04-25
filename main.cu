#include <thrust/device_vector.h>

#include <ingot/ingot.h>
using namespace ingot;

struct integration_step {

    ode::CR3BP cr3bp{0.04};

    template<typename T>
    __device__ void operator()(T arg) {

        auto& t = thrust::get<0>(arg); // time
        auto& h = thrust::get<1>(arg); // timestep
        auto x = thrust::get<2>(arg);  //

        constexpr auto N = decltype(x)::SizeAtCompileTime;
        static_assert(N != Eigen::Dynamic,
                      "ODE state-vector must be statically sized!");

        StackArray<double, 6> x_old = thrust::get<2>(arg);

        // Eigen::Array<double, 6, 1> x_new = RK4{}(cr3bp, t, h, x_old);
        Eigen::Array<double, 6, 1> x_new = RKF78{}(cr3bp, t, h, x_old);

        t += h;
        x = x_new;

        if (threadIdx.x == 0 and blockIdx.x == 0) {
            const auto s = x_new;
            printf("%g %g %g %g %g %g\n", s[0], s[1], s[2], s[3], s[4], s[5]);
        }
    }
};

struct initializer {
    __device__ auto operator()(int i) const {

        const auto vmag = 0.1;
        const auto theta = i * M_PI;

        const auto vx = vmag * cos(theta);
        const auto vy = vmag * sin(theta);

        Eigen::Array<double, 6, 1> ret;
        ret << -.5, 0, 0, vx, vy, 0;

        return ret;
    }
};

int main() {
    using T = double;

    // Allocate initial buffer
    const int nparticles = 1024;
    thrust::device_vector<T> x{nparticles};

    ColIter<T, 6> ci{x.data().get(), nparticles};

    auto idx = thrust::counting_iterator<int>{0};

    thrust::transform(idx, idx + nparticles, ci, initializer{});

    thrust::device_vector<T> t{nparticles};
    thrust::fill(t.begin(), t.end(), 0);
    thrust::device_vector<T> h{nparticles};
    // thrust::fill(h.begin(), h.end(), std::numeric_limits<T>::epsilon());
    thrust::fill(h.begin(), h.end(), .01);

    auto zp = zip_tuple_iters(t.begin(), h.begin(), ci);

    for (int i = 0; i < 500; i++)
        thrust::for_each(zp, zp + nparticles, integration_step{});
}
