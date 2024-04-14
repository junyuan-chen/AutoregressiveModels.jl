module AutoregressiveModels

using Base: ReshapedArray
using LinearAlgebra: Cholesky, cholesky!, cholesky, UpperTriangular, LowerTriangular,
    lu!, ldiv!, rdiv!, inv!, mul!, diagm, eigen!, eigen, I,
    BLAS.@blasfunc, BlasInt, chkstride1, libblastrampoline, LAPACK.chklapackerror,
    Algorithm, default_svd_alg, DivideAndConquer, QRIteration
using MatrixEquations: lyapd
using Random: randn!
using Roots: find_zero, Brent
using StatsAPI: RegressionModel
using StatsBase: sample
using Tables
using Tables: getcolumn

import Base: show
import LinearAlgebra: LAPACK.gesdd!, LAPACK.gesvd!, _svd!
import StatsAPI: response, modelmatrix, coef, residuals, dof_residual, fit, fit!, r2

# Reexport objects from StatsAPI
export response, modelmatrix, coef, residuals, dof_residual, fit, fit!, r2

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

       ARMAProcess,

       Factor

include("lapack.jl")
include("utils.jl")
include("process.jl")
include("estimation.jl")
include("bootstrap.jl")
include("arma.jl")
include("factor.jl")

end # module AutoregressiveModels
