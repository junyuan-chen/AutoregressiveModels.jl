using Test
using AutoregressiveModels

using CSV
using ConfidenceBands
using DataFrames
using LinearAlgebra: diagm, I, cholesky
using LocalProjections: datafile

exampledata(name) = CSV.read(datafile(name), DataFrame)

const tests = [
    "process",
    "estimation",
    "bootstrap"
]

printstyled("Running tests:\n", color=:blue, bold=true)

@time for test in tests
    include("$test.jl")
    println("\033[1m\033[32mPASSED\033[0m: $(test)")
end
