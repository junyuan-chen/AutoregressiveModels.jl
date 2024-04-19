# An alternative to reshape that does not allocate
_reshape(A::AbstractArray, dims::Int...) = ReshapedArray(A, dims, ())

# Check whether the input data is a column table
function checktable(data)
    Tables.istable(data) ||
        throw(ArgumentError("data of type $(typeof(data)) is not `Tables.jl`-compatible"))
    Tables.columnaccess(data) ||
        throw(ArgumentError("data of type $(typeof(data)) is not a column table"))
end

# Indicate rows with finite and nonmissing data
function _esample!(esample::AbstractVector{Bool}, aux::AbstractVector{Bool},
        v::AbstractVector{<:Union{Real, Missing}})
    aux .= isequal.(isfinite.(v), true)
    esample .&= aux
end

# Inner product with a subset of Y ordered backward
function bdot(X::Tuple, Y::AbstractVector, t::Int, N::Int)
    out = 0.0
    for i in 1:N
        out += X[i] * Y[t-i]
    end
    return out
end

function _ols!(Y, X, crossX, coef, resid)
    mul!(coef, X', Y)
    mul!(crossX, X', X)
    ldiv!(cholesky!(crossX), coef)
    copyto!(resid, Y)
    mul!(resid, X, coef, -1.0, 1.0)
end

datafile(name::Union{Symbol,String}) = (@__DIR__)*"/../data/$(name)"
