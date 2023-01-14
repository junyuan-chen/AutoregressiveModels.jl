module AutoregressiveModels

using LinearAlgebra: cholesky!, lu!, ldiv!, rdiv!, inv!, mul!, diagm, eigen!, eigen, I
using MatrixEquations: lyapd
using Random: randn!
using Roots: find_zero, Brent
using StatsAPI: StatisticalModel, RegressionModel
using StatsBase: sample
using Tables
using Tables: getcolumn

import Base: show
import StatsAPI: coef, vcov, confint, coeftable, modelmatrix, residuals, dof_residual, fit

# Reexport objects from StatsAPI
export coef, stderror, confint, coeftable, modelmatrix, residuals, dof_residual, fit

export VARProcess,
       nvar,
       arorder,
       companionform,
       isstable,
       hasintercept,
       simulate!,
       impulse!,
       impulse,

       OLS,
       coefcorrected,
       residvcov,
       VectorAutoregression,
       biascorrect,

       boot!

include("utils.jl")
include("process.jl")
include("estimation.jl")
include("bootstrap.jl")

end # module AutoregressiveModels
