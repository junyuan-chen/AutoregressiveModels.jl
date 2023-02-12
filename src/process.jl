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
hasintercept(::VARProcess{TB,TI}) where {TB,TI} = TI !== Nothing

function companionform(var::VARProcess)
    B = var.B
    N, NP = size(B)
    C = diagm(-N=>ones(NP-N))
    copyto!(view(C, 1:N, :), B)
    return C
end

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

simulate(εs::AbstractArray, var::VARProcess, Y0=nothing; kwargs...) =
    simulate!(copy(εs), var, Y0; kwargs...)

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
