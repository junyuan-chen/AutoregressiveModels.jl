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
