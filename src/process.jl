struct VARProcess{TB<:AbstractMatrix, TI<:Union{AbstractVector,Nothing}}
    B::TB
    B0::TI
    function VARProcess(B::AbstractMatrix{TF},
            B0::Union{AbstractVector{TF},Nothing}=nothing; copy::Bool=false) where TF
        m, n = size(B)
        n >= m > 0 || throw(ArgumentError(
            "coefficient matrix of size $(size(B)) is not accepted"))
        B0 === nothing || m == length(B0) || throw(DimensionMismatch(
            "coefficient matrix of size $(size(B)) does not match intercept of length $(length(B0))"))
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

function simulate!(εs::AbstractMatrix, var::VARProcess, Y0::AbstractVecOrMat;
        nlag::Integer=arorder(var))
    0 < nlag <= arorder(var) || throw(ArgumentError(
        "nlag must be positive and no greater than $(arorder(r))"))
    N = size(var.B, 1)
    size(εs, 1) == N || throw(DimensionMismatch(
        "εs of size $(size(εs)) does not match var with $N variables"))
    size(Y0, 1) == N || throw(DimensionMismatch(
        "Y0 of size $(size(Y0)) does not match var with $N variables"))
    p0 = size(Y0, 2)
    p0 > nlag && @warn "extra initial values are ignored"
    rlag0 = max(p0-nlag+1, 1):p0
    copyto!(view(εs, :, 1:length(rlag0)), view(Y0, :, rlag0))
    for t in length(rlag0)+1:size(εs, 2)
        X = view(εs, :, t-1:-1:max(t-nlag, 1))
        var(view(εs, :, t), _reshape(X, length(X)))
    end
    return εs
end

function simulate!(εs::AbstractArray{T1,3}, var::VARProcess, Y0::AbstractArray{T2,3};
        nlag::Integer=arorder(var)) where {T1,T2}
    0 < nlag <= arorder(var) || throw(ArgumentError(
        "nlag must be positive and no greater than $(arorder(r))"))
    N = size(var.B, 1)
    size(εs, 1) == N || throw(DimensionMismatch(
        "εs of size $(size(εs)) does not match var with $N variables"))
    size(Y0, 1) == N || throw(DimensionMismatch(
        "Y0 of size $(size(Y0)) does not match var with $N variables"))
    size(εs, 3) == size(Y0, 3) || throw(DimensionMismatch(
        "Y0 of size $(size(Y0)) does not match εs of size $(size(εs))"))
    p0 = size(Y0, 2)
    p0 > nlag && @warn "extra initial values are ignored"
    rlag0 = max(p0-nlag+1, 1):p0
    copyto!(view(εs, :, 1:length(rlag0), :), view(Y0, :, rlag0, :))
    N3 = size(εs, 3)
    for t in length(rlag0)+1:size(εs, 2)
        rlag = t-1:-1:max(t-nlag, 1)
        var(view(εs, :, t, :), _reshape(view(εs, :, rlag, :), N*length(rlag), N3))
    end
    return εs
end

# For some unknown reason, bootstrap! involving impulse! allocates for type inference
function simulate!(εs::AbstractMatrix, var::VARProcess; nlag::Integer=arorder(var))
    0 < nlag <= arorder(var) || throw(ArgumentError(
        "nlag must be positive and no greater than $(arorder(r))"))
    N = size(var.B, 1)
    size(εs, 1) == N || throw(DimensionMismatch(
        "εs of size $(size(εs)) does not match var with $N variables"))
    var(view(εs, :, 1))
    for t in 2:size(εs, 2)
        X = view(εs, :, t-1:-1:max(t-nlag, 1))
        var(view(εs, :, t), _reshape(X, length(X)))
    end
    return εs
end

function simulate!(εs::AbstractArray{T,3}, var::VARProcess;
        nlag::Integer=arorder(var)) where T
    0 < nlag <= arorder(var) || throw(ArgumentError(
        "nlag must be positive and no greater than $(arorder(r))"))
    N = size(var.B, 1)
    size(εs, 1) == N || throw(DimensionMismatch(
        "εs of size $(size(εs)) does not match var with $N variables"))
    var(view(εs, :, 1, :))
    N3 = size(εs, 3)
    for t in 2:size(εs, 2)
        rlag = t-1:-1:max(t-nlag, 1)
        var(view(εs, :, t, :), _reshape(view(εs, :, rlag, :), N*length(rlag), N3))
    end
    return εs
end

function impulse!(out::AbstractMatrix, var::VARProcess, ε0::AbstractVector;
        nlag::Integer=arorder(var))
    size(ε0, 1) == size(out, 1) || throw(DimensionMismatch(
        "ε0 of size $(size(ε0)) does not match out of size $(size(out))"))
    fill!(out, zero(eltype(out)))
    copyto!(view(out, :, 1), ε0)
    if size(out, 2) > 1
        hasintercept(var) && (var = VARProcess(var.B))
        simulate!(out, var, nlag=nlag)
    end
    return out
end

function impulse!(out::AbstractArray{T,3}, var::VARProcess, ε0::AbstractMatrix;
        nlag::Integer=arorder(var)) where T
    size(ε0, 1) == size(out, 1) && size(ε0, 2) == size(out, 3) || throw(DimensionMismatch(
        "ε0 of size $(size(ε0)) does not match out of size $(size(out))"))
    fill!(out, zero(eltype(out)))
    copyto!(view(out, :, 1, :), ε0)
    if size(out, 2) > 1
        hasintercept(var) && (var = VARProcess(var.B))
        simulate!(out, var, nlag=nlag)
    end
    return out
end

function impulse(var::VARProcess, ε0::AbstractVecOrMat, nhorz::Integer;
        nlag::Integer=arorder(var))
    N = size(var.B, 1)
    size(ε0, 1) == N || throw(DimensionMismatch(
        "ε0 of size $(size(ε0)) does not match var with $N variables"))
    out = zeros(N, nhorz+1, size(ε0, 2))
    copyto!(view(out, :, 1, :), ε0)
    if nhorz > 0
        hasintercept(var) && (var = VARProcess(var.B))
        simulate!(out, var, nlag=nlag)
    end
    return out
end

