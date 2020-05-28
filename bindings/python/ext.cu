#include <ingot/ingot.h>
#include <ingot/ode/CR3BP.h>
#include <ingot/integrator/adaptive.h>
#include <ingot/integrate.h>
#include <ingot/method/RKF78.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
using namespace ingot;

// Did the sign of the y-component change?
template<typename T, int N>
struct YVal {
    CUDA_HOSTDEV
    T operator()(double t, double h, ColVal<T, N> const& u) const {
        return u[1];
    }
};

#define cuCheck(x) cuCheckImpl((x), __PRETTY_FUNCTION__, __LINE__)

void cuCheckImpl(cudaError_t x, std::string func, int line) {
    if (x != cudaSuccess) {
        throw std::runtime_error("In function " + func +
                                 " line " + std::to_string(line));
    }
}

auto integrate_cr3bp_rkf78_steps(
        double mu,
        Eigen::Ref<Eigen::VectorXd> host_t,
        Eigen::Ref<Eigen::VectorXd> host_h,
        Eigen::Ref<Eigen::MatrixXd> host_u
        ) {

    /*
     * Allocate and initialize particles
     */
    const int size = host_t.size();

    if (host_h.size() != size) {
        throw std::invalid_argument("t.size() != h.size()");
    }
    if (host_u.cols() != size) {
        throw std::invalid_argument("t.size() != u.cols()");
    }
    if (host_u.rows() != 6) {
        throw std::invalid_argument("u.rows() != 6");
    }

    Ensemble<double, 6> ensemble{size};

#define HtoD cudaMemcpyHostToDevice

    cuCheck(cudaMemcpy(ensemble.t.data().get(), host_t.data(),
                size * sizeof(double), HtoD));
    cuCheck(cudaMemcpy(ensemble.h.data().get(), host_t.data(),
                size * sizeof(double), HtoD));
    cuCheck(cudaMemcpy2D(ensemble.u.data.data().get(), size * sizeof(double),
                         host_u.data(), size * sizeof(double),
                         size * sizeof(double), 6, HtoD));

    /*
     * Integrate with output function for fixed number of steps
     */
    const auto i = integrator::make_adaptive(method::RKF78{}, 1e-8);
    const auto sols = integrate_steps(i, ode::CR3BP{mu}, ensemble, 10000,
                                      YVal<double, 6>{});

    return sols;
}

PYBIND11_MODULE(PY_EXT_NAME, m) {

    using namespace ingot;

    py::class_<output<double, 6>>(m, "Output3D")
        .def_readonly("t", &output<double, 6>::t)
        .def_readonly("h", &output<double, 6>::h)
        .def_readonly("u", &output<double, 6>::u)
        ;

    m.def("integrate_cr3bp_rkf78_steps", integrate_cr3bp_rkf78_steps);
}
