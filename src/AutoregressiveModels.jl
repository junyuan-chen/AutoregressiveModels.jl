module AutoregressiveModels

using Base: ReshapedArray
using LinearAlgebra: Cholesky, cholesky!, cholesky, UpperTriangular, LowerTriangular,
    lu!, ldiv!, rdiv!, inv!, mul!, diagm, eigen!, eigen, I
using MatrixEquations: lyapd
using Random: randn!
using Roots: find_zero, Brent
using StatsAPI: RegressionModel
using StatsBase: sample
using Tables
using Tables: getcolumn

import Base: show
import StatsAPI: response, modelmatrix, coef, residuals, dof_residual, fit

# Reexport objects from StatsAPI
export response, modelmatrix, coef, residuals, dof_residual, fit

export VARProcess,
       nvar,
       arorder,
       maorder,
       companionform,
       isstable,
       hasintercept,
       simulate!,
       simulate,
       impulse!,
       impulse,
       forecastvar!,
       forecastvar,
       histvar!,
       histvar,

       OLS,
       intercept,
       coefcorrected,
       residvcov,
       residchol,
       VectorAutoregression,
       biascorrect,

       randomindex,
       wilddraw!,
       iidresiddraw!,
       bootstrap!,

       ARMAProcess

include("utils.jl")
include("process.jl")
include("estimation.jl")
include("bootstrap.jl")
include("arma.jl")

end # module AutoregressiveModels
