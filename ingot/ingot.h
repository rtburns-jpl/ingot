#pragma once

namespace ingot {
template<typename T, int N>
class StackArray;
}

#define EIGEN_DENSEBASE_PLUGIN <ingot/densebase_plugin.h>
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include <Eigen/Dense>

#include <thrust/iterator/iterator_facade.h>
#include <thrust/device_vector.h>

#include <vector>

namespace ingot {

#include "defs.h"
#include "StackArray.h"
#include "eigen_helpers.h"


namespace ode {
#include "ode/BaseODE.h"
#include "ode/TwoBody.h"
#include "ode/CR3BP.h"
} // namespace ode

namespace method {
#include "method/Integrator.h"
#include "method/RK4.h"
#include "method/RKF78.h"
} // namespace method

#include "ODEProblem.h"
#include "EnsembleProblem.h"
#include "solve.h"

} // namespace ingot
