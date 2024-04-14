using Test
using AutoregressiveModels

using AutoregressiveModels: SDDcache, SVDcache, svdcache, datafile, coefB, intercept
using CSV
using ConfidenceBands
using DataFrames
using LinearAlgebra: diagm, I, cholesky, _svd!, DivideAndConquer, QRIteration
using LocalProjections: datafile as lpdatafile
using MAT

lpexampledata(name) = CSV.read(lpdatafile(name), DataFrame)

const tests = [
    "lapack",
    "process",
    "estimation",
    "bootstrap",
    "arma",
    "factor"
]

printstyled("Running tests:\n", color=:blue, bold=true)

@time for test in tests
    include("$test.jl")
    println("\033[1m\033[32mPASSED\033[0m: $(test)")
end
