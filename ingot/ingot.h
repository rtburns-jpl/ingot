#pragma once

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include <Eigen/Dense>

#include <thrust/iterator/iterator_facade.h>

#include "defs.h"
#include "StackArray.h"
#include "eigen_helpers.h"
#include "CR3BP.h"
#include "Integrator.h"
#include "ODEProblem.h"
#include "EnsembleProblem.h"
