"""
    VARProcess(B::AbstractMatrix, B0=nothing; copy::Bool=false)

Return a vector autoregressive process ``Y_t`` defined as

```math
Y_t = B_0 + \\sum_{l=1}^p B_l Y_{t-l} + \\ε_t
```

where the autoregressive coefficients ``\\{\\B_l\\}_l``
are collected in matrix `B` by column
and the optional intercept ``B_0`` is in a vector `B0`.
"""
struct VARProcess{TB<:AbstractMatrix, TI<:Union{AbstractVector,Nothing}}
    B::TB
    B0::TI
    function VARProcess(B::AbstractMatrix{TF},
            B0::Union{AbstractVector{TF}, Nothing}=nothing; copy::Bool=false) where TF
        Base.require_one_based_indexing(B)
        m, n = size(B)
        n >= m > 0 || throw(ArgumentError(
            "coefficient matrix of size $(size(B)) is not accepted"))
        if B0 !== nothing
            Base.require_one_based_indexing(B0)
            m == length(B0) || throw(DimensionMismatch(
                "coefficient matrix of size $(size(B)) does not match intercept of length $(length(B0))"))
        end
        n % m == 0 || throw(ArgumentError(
            "coefficient matrix of size $(size(B)) is not accepted"))
        if copy
            B = collect(B)
            B0 === nothing || (B0 = collect(B0))
        end
        return new{typeof(B), typeof(B0)}(B, B0)
    end
end

nvar(var::VARProcess) = size(var.B, 1)
arorder(var::VARProcess) = size(var.B, 2) ÷ size(var.B, 1)
maorder(var::VARProcess) = 0
hasintercept(::VARProcess{TB,TI}) where {TB,TI} = TI !== Nothing

"""
    companionform(var::VARProcess)

Return a coefficient matrix representing the companion form of `var`.
"""
function companionform(var::VARProcess)
    B = var.B
    N, NP = size(B)
    C = diagm(-N=>ones(NP-N))
    copyto!(view(C, 1:N, :), B)
    return C
end

"""
    isstable(var::VARProcess, offset::Real=0)

Return a Boolean value indicating whether `var` is stable,
optionally with the comparison adjusted by `offset` for some tolerance.
"""
isstable(var::VARProcess, offset::Real=0) =
    abs(eigen!(companionform(var), sortby=abs).values[end]) < 1 - offset

# Outcomes are written to ε in-place
# Values in Y are ordered backward in time
@inline function (var::VARProcess)(ε::AbstractVecOrMat, Y::AbstractVecOrMat)
    B = var.B
    NP = size(Y, 1)
    if size(B, 2) == NP
        mul!(ε, B, Y, 1.0, 1.0)
    else
        B2 = view(B, :, 1:NP)
        mul!(ε, B2, Y, 1.0, 1.0)
    end
    if hasintercept(var)
        ε .+= var.B0
    end
    return ε
end

(var::VARProcess)(ε::AbstractVecOrMat) = (ε .+= var.B0)
(var::VARProcess{TB, Nothing} where {TB<:AbstractMatrix})(ε::AbstractVecOrMat) = ε

"""
    simulate!(εs::AbstractArray, var::VARProcess, Y0=nothing; kwargs...)

Simulate the `var` process using the shocks specified in `εs` and initial values `Y0`.
Results are stored by overwriting `εs` in-place.
If `Y0` is `nothing` or does not contain enough lags, zeros are used.
See also [`simulate`](@ref) and [`impulse!`](@ref).

# Keywords
- `nlag::Integer=arorder(var)`: the number of lags from `var` used for simulations.
"""
function simulate!(εs::AbstractMatrix, var::VARProcess,
        Y0::Union{AbstractVecOrMat,Nothing}=nothing; nlag::Integer=arorder(var))
    Base.require_one_based_indexing(εs)
    0 < nlag <= arorder(var) || throw(ArgumentError(
        "nlag must be positive and no greater than $(arorder(r))"))
    N = nvar(var)
    size(εs, 1) == N || throw(DimensionMismatch(
        "εs of size $(size(εs)) does not match var with $N variables"))
    if Y0 === nothing || isempty(Y0)
        var(view(εs, :, 1))
        t0 = 1
    else
        Base.require_one_based_indexing(Y0)
        size(Y0, 1) == N || throw(DimensionMismatch(
            "Y0 of size $(size(Y0)) does not match var with $N variables"))
        t0 = size(Y0, 2)
        copyto!(view(εs, :, 1:t0), Y0)
    end
    for t in t0+1:size(εs, 2)
        rlag = t-1:-1:max(t-nlag, 1)
        var(view(εs, :, t), _reshape(view(εs, :, rlag), N*length(rlag)))
    end
    return εs
end

function simulate!(εs::AbstractArray{T,3}, var::VARProcess,
        Y0::Union{AbstractArray,Nothing}=nothing; nlag::Integer=arorder(var)) where T
    Base.require_one_based_indexing(εs)
    0 < nlag <= arorder(var) || throw(ArgumentError(
        "nlag must be positive and no greater than $(arorder(r))"))
    N = nvar(var)
    size(εs, 1) == N || throw(DimensionMismatch(
        "εs of size $(size(εs)) does not match var with $N variables"))
    if Y0 === nothing || isempty(Y0)
        var(view(εs, :, 1, :))
        t0 = 1
    else
        Base.require_one_based_indexing(Y0)
        size(Y0, 1) == N || throw(DimensionMismatch(
            "Y0 of size $(size(Y0)) does not match var with $N variables"))
        ndims(Y0) == 3 && size(εs, 3) == size(Y0, 3) || throw(DimensionMismatch(
            "Y0 of size $(size(Y0)) does not match εs of size $(size(εs))"))
        t0 = size(Y0, 2)
        copyto!(view(εs, :, 1:t0, :), Y0)
    end
    N3 = size(εs, 3)
    for t in t0+1:size(εs, 2)
        rlag = t-1:-1:max(t-nlag, 1)
        var(view(εs, :, t, :), _reshape(view(εs, :, rlag, :), N*length(rlag), N3))
    end
    return εs
end

"""
    simulate(εs::AbstractArray, var::VARProcess, Y0=nothing; nlag::Integer=arorder(var))

Same as [`simulate!`](@ref), but makes a copy of `εs` for results
and hence does not overwrite `εs`.
"""
simulate(εs::AbstractArray, var::VARProcess, Y0=nothing; kwargs...) =
    simulate!(copy(εs), var, Y0; kwargs...)

"""
    impulse!(out, var::VARProcess, ε0::AbstractVecOrMat; kwargs...)
    impulse!(out, var::VARProcess, ishock::Union{Integer, AbstractRange}; kwargs...)

Compute the responses of the `var` process to the impulse
specified with `ε0` or `ishock` and store the results in an array `out`.
The number of horizons to be computed is determined by the second dimension of `out`.
See also [`impulse`](@ref) and [`simulate!`](@ref).

As a vector, `ε0` specifies the magnitude of the impulse to each variable;
columns in a matrix are interpreted as multiple impulses with results
stored separately along the third dimension of array `out`.
Alternatively, `ishock` specifies the index of a single variable that is affected on impact;
a range of indices is intercepted as multiple impulses.

# Keywords
- `nlag::Integer=arorder(var)`: the number of lags from `var` used for simulations.
"""
function impulse!(out::AbstractArray, var::VARProcess, ε0::AbstractVecOrMat;
        nlag::Integer=arorder(var))
    size(ε0, 1) == size(out, 1) && size(ε0, 2) == size(out, 3) || throw(DimensionMismatch(
        "ε0 of size $(size(ε0)) does not match out of size $(size(out))"))
    fill!(out, zero(eltype(out)))
    # view allocates if array dimension changes
    if ndims(out) == 2
        copyto!(view(out, :, 1), ε0)
    elseif ndims(out) == 3
        copyto!(view(out, :, 1, :), ε0)
    else
        throw(ArgumentError("dimension of out must be 2 or 3"))
    end
    if size(out, 2) > 1
        hasintercept(var) && (var = VARProcess(var.B))
        simulate!(out, var; nlag=nlag)
    end
    return out
end

function impulse!(out::AbstractArray, var::VARProcess, ishock::Union{Integer, AbstractRange};
        nlag::Integer=arorder(var))
    length(ishock) == size(out, 3) || throw(DimensionMismatch(
        "number of shocks $(length(ishock)) does not match out of size $(size(out))"))
    fill!(out, zero(eltype(out)))
    if ndims(out) == 2
        ishock isa Integer || (ishock = ishock[1])
        out[ishock,1] = 1
    elseif ndims(out) == 3
        for (i, s) in enumerate(ishock)
            out[s,1,i] = 1
        end
    else
        throw(ArgumentError("dimension of out must be 2 or 3"))
    end
    if size(out, 2) > 1
        hasintercept(var) && (var = VARProcess(var.B))
        simulate!(out, var; nlag=nlag)
    end
    return out
end

const _impulse_common_docstr = """
    Same as [`impulse!`](@ref), but allocates an array for storing the results.
    The number of horizons needs to be specified explicitly with `nhorz`."""

"""
    impulse(var::VARProcess, ε0::AbstractVecOrMat, nhorz::Integer; kwargs...)
    impulse(var::VARProcess, ishock::Union{Integer, AbstractRange}, nhorz::Integer; kwargs...)

$_impulse_common_docstr
"""
function impulse(var::VARProcess, ε0::AbstractVecOrMat, nhorz::Integer;
        nlag::Integer=arorder(var))
    N = nvar(var)
    size(ε0, 1) == N || throw(DimensionMismatch(
        "ε0 of size $(size(ε0)) does not match var with $N variables"))
    out = zeros(N, nhorz, size(ε0, 2))
    copyto!(view(out, :, 1, :), ε0)
    if nhorz > 1
        hasintercept(var) && (var = VARProcess(var.B))
        simulate!(out, var; nlag=nlag)
    end
    return out
end

function impulse(var::VARProcess, ishock::Union{Integer, AbstractRange}, nhorz::Integer;
        nlag::Integer=arorder(var))
    N = nvar(var)
    out = zeros(N, nhorz, length(ishock))
    for (i, s) in enumerate(ishock)
        out[s,1,i] = 1
    end
    if nhorz > 1
        hasintercept(var) && (var = VARProcess(var.B))
        simulate!(out, var; nlag=nlag)
    end
    return out
end

"""
    forecastvar!(out::AbstractArray{T,3}, irfs::AbstractArray{T,3})

Compute the variance of forecast error over each horizon
given orthogonalized impulse response coefficients specified with `irfs`
and store the results in an array `out`.
The number of horizons computed is determined by the second dimension of `irfs`
and `out` should have the same size as `irfs`.
See also [`forecastvar`](@ref).
"""
function forecastvar!(out::AbstractArray{T,3}, irfs::AbstractArray{T,3}) where T
    size(out) == size(irfs) ||
        throw(DimensionMismatch("out and irfs do not have the same size"))
    out .= irfs.^2
    # cumsum! allocates and is slower
    t2 = first(axes(out, 2)) + 1
    @inbounds for s in axes(out, 3)
        for t in t2:last(axes(out, 2))
            for i in axes(out, 1)
                out[i,t,s] += out[i,t-1,s]
            end
        end
    end
    return out
end

"""
    forecastvar(irfs::AbstractArray)

Same as [`forecastvar!`](@ref), but allocates an array for storing the results.
"""
forecastvar(irfs::AbstractArray) = forecastvar!(similar(irfs), irfs)

"""
    histvar!(out::AbstractArray{TF,3}, irfs::AbstractArray{TF,3}, shocks::AbstractMatrix)

Compute the variance to be used for historical decomposition
given impulse response coefficients specified with `irfs` and historical `shocks`.
Results are stored in an array `out`.
See also [`histvar`](@ref).
"""
function histvar!(out::AbstractArray{TF,3}, irfs::AbstractArray{TF,3},
        shocks::AbstractMatrix) where TF
    N, T, S = size(out)
    N1, nhorz, S1 = size(irfs)
    T2, S2 = size(shocks)
    N == N1 || throw(DimensionMismatch("the first dimension of out is expected to be $N1"))
    T == T2 || throw(DimensionMismatch("the second dimension of out is expected to be $T2"))
    S1 == S2 || throw(DimensionMismatch(
        "the numbers of shocks in irfs and shocks do not match"))
    S == S1 || throw(DimensionMismatch("the thrid dimension of out is expected to be $S"))
    for s in 1:S
        for t in 1:T
            hs = 1:min(nhorz, t)
            bs = t:-1:max(t-nhorz+1, 1)
            mul!(view(out,:,t,s), view(irfs,:,hs,s), view(shocks,bs,s))
        end
    end
    return out
end

"""
    histvar(irfs::AbstractArray, shocks::AbstractMatrix)

Same as [`histvar!`](@ref), but allocates an array for storing the results.
"""
function histvar(irfs::AbstractArray, shocks::AbstractMatrix)
    T, S = size(shocks)
    out = similar(irfs, (size(irfs,1), T, S))
    return histvar!(out, irfs, shocks)
end

show(io::IO, var::VARProcess) =
    print(io, size(var.B,1), '×', size(var.B,2), " ", typeof(var))

function show(io::IO, ::MIME"text/plain", var::VARProcess)
    println(io, var, " with coefficient matrix:")
    Base.print_array(io, var.B)
end
