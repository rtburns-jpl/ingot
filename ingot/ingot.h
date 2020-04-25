#pragma once

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include <Eigen/Dense>

#include <thrust/iterator/iterator_facade.h>

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

#include "Integrator.h"
#include "methods/RK4.h"
#include "methods/RKF78.h"

#include "ODEProblem.h"
#include "EnsembleProblem.h"
#include "solve.h"

} // namespace ingot
