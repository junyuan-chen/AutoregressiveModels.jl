"""
    AbstractDetrend

Types for specifying the methods for
transforming the data before and after factor estimation.

Methods of `nskip`, `detrend!` and `invdetrend!` should be defined
for each concrete subtype of `AbstractDetrend`.
"""
abstract type AbstractDetrend end

"""
    NoTrend <: AbstractDetrend

Do not transform the data for factor estimation.
"""
struct NoTrend <: AbstractDetrend end

nskip(::NoTrend) = 0
detrend!(::NoTrend, Y0::AbstractMatrix, Y::AbstractMatrix) = copyto!(Y0, Y)
invdetrend!(::NoTrend, Y::AbstractMatrix, Y0::AbstractMatrix) = copyto!(Y, Y0)

"""
    FirstDiff <: AbstractDetrend

Take the first difference, possibly over multiple horizons, for factor estimation.
"""
struct FirstDiff <: AbstractDetrend
    h::Int
    FirstDiff(h::Integer=1) = new(h)
end

nskip(tr::FirstDiff) = tr.h

function detrend!(tr::FirstDiff, Y0::AbstractMatrix, Y::AbstractMatrix)
    h = tr.h
    @inbounds for j in axes(Y, 2)
        for i in h+1:size(Y, 1)
            Y0[i-h,j] = Y[i,j] - Y[i-h,j]
        end
    end
    return Y0
end

function invdetrend!(tr::FirstDiff, Y::AbstractMatrix, Y0::AbstractMatrix)
    h = tr.h
    @inbounds for j in axes(Y, 2)
        for i in 1:h
            Y[i,j] = Y0[i,j]
        end
        for i in h+1:size(Y, 1)
            Y[i,j] = Y[i-h,j] + Y0[i,j]
        end
    end
    return Y
end

"""
    DynamicFactor

Results and cache from estimating a dynamic factor model.
This object holds all arrays that are necessary for
estimating the model without additional memory allocations.
See also [`fit`](@ref) and [`fit!`](@ref).

# Fields
- `Y::Matrix{TF}`: data matrix where each column corresponds to a time series.
- `Y0::Matrix{TF}`: transformed `Y` that is used for factor estimation.
- `trans::AbstractDetrend`: transformation conducted before and after factor estimation.
- `facobs::Union{Matrix{TF}, Nothing}`: data matrix for observed factors.
- `facobs0::Union{Matrix{TF}, Nothing}`: transformed `facobs` that is used for factor estimation.
- `f::Factor{TF}`: results and cache from estimating the factors.
- `facX::Matrix{TF}`: factors transformed back from `f` and possibly additional variables.
- `crossfacX::Matrix{TF}`: cache for holding `facX'facX` and factorization.
- `Λ::Matrix{TF}`: least-squares coefficient estimates associated with `facX`.
- `u::Matrix{TF}`: residuals from regressing `Y` on `facX`.
- `facproc`: model for the evolution of factors.
- `lagca::Matrix{TF}`: cache for the lags of an idiosyncratic component.
- `crosslagca::Matrix{TF}`: cache for holding `lagca'lagca` and factorization.
- `arcoef::Matrix{TF}`: autoregressive coefficient estimates for the idiosyncratic components.
- `resid::Matrix{TF}`: residuals after removing the autoregressive terms from the idiosyncratic components.
- `σ::Vector{TF}`: residual standard errors associated with `resid`.
- `nfaclag::Int`: number of lags for estimating `facproc`.
- `narlag::Int`: number of lags for fitting the autoregressive model with the idiosyncratic components.
"""
struct DynamicFactor{TF, TR<:AbstractDetrend, FO<:Union{Matrix{TF}, Nothing}, F<:Factor{TF}, FM}
    Y::Matrix{TF}
    Y0::Matrix{TF}
    trans::TR
    facobs::FO
    facobs0::FO
    f::F
    facX::Matrix{TF}
    crossfacX::Matrix{TF}
    Λ::Matrix{TF}
    u::Matrix{TF}
    facproc::FM
    lagca::Matrix{TF}
    crosslagca::Matrix{TF}
    arcoef::Matrix{TF}
    resid::Matrix{TF}
    σ::Vector{TF}
    nfaclag::Int
    narlag::Int
end

nfactor(f::DynamicFactor) = nfactor(f.f)

function _dfm_est!(Y, facX, crossfacX, Λ, u, lagca, crosslagca, arcoef, resid, σ, narlag, arexclude)
    Tfull, N = size(Y)
    # Estimate coefficients for common components
    Y1 = view(Y, Tfull-size(facX,1)+1:Tfull, :)
    _ols!(Y1, facX, crossfacX, Λ, u)
    T = size(lagca, 1)
    T1 = size(u, 1)
    t1 = T1 - T + 1
    dofr = T - narlag
    # Estimate coefficients for idiosyncratic components
    for n in 1:N
        coefn = view(arcoef, :, n)
        un = view(u, t1:T1, n)
        residn = view(resid, :, n)
        if arexclude !== nothing && n in arexclude
            fill!(coefn, zero(eltype(coefn)))
            copyto!(residn, un)
            σ[n] = sqrt(sum(abs2, residn) / T)
        else
            for l in 1:narlag
                lagca[:,l] .= view(u, t1-l:T1-l, n)
            end
            _ols!(un, lagca, crosslagca, coefn, residn)
            σ[n] = sqrt(sum(abs2, residn) / dofr)
        end
    end
end

function DynamicFactor(Y::AbstractMatrix, facobs::Union{AbstractVecOrMat, Nothing},
        trans::AbstractDetrend, nfac::Union{Integer, AbstractNFactorCriterion},
        FacProc::Union{Type, Nothing}, nfaclag::Integer, narlag::Integer,
        X::Union{AbstractVecOrMat, Nothing}=nothing;
        svdalg::Algorithm=default_svd_alg(Y), maxiter::Integer=10000, tol::Real=1e-8,
        arexclude=nothing, facprockwargs::NamedTuple=NamedTuple())
    Y = convert(Matrix, Y)
    Tfull, N = size(Y)
    nsk = nskip(trans)
    T = Tfull - narlag - nsk
    nX = X === nothing ? 0 : size(X, 2)
    Tfull - nsk > nX && T > narlag+1 ||
        throw(ArgumentError("number of observations is too low"))
    Y0 = similar(Y, Tfull-nsk, N)
    detrend!(trans, Y0, Y)
    if facobs === nothing
        facobs0 = nothing
    else
        size(facobs, 1) == Tfull || throw(DimensionMismatch(
            "facobs and Y do not have the same number of rows"))
        facobs = convert(Matrix, facobs)
        facobs0 = similar(facobs, Tfull-nsk, size(facobs,2))
        detrend!(trans, facobs0, facobs)
    end
    f = Factor(Y0, facobs0, nfac; svdalg=svdalg, maxiter=maxiter, tol=tol)
    # nfac could be AbstractNFactorCriterion
    nfac = nfactor(f)
    # Number of valid rows should remain unchanged with invdetrend!
    facX = similar(Y, Tfull-nsk, nfac+nX+1)
    invdetrend!(trans, view(facX,:,1:nfac), f.fac)
    if X !== nothing
        size(X, 1) == Tfull || throw(DimensionMismatch(
            "X and Y do not have the same number of rows"))
        copyto!(view(facX,:,nfac+1:nfac+nX), view(X,nsk+1:Tfull,:))
    end
    # Always add a constant term at the end for common components
    facX[:,end] .= 1.0
    nfacX = size(facX, 2)
    crossfacX = similar(Y, nfacX, nfacX)
    Λ = similar(Y, nfacX, N)
    u = similar(Y, Tfull-nsk, N)
    # No need to add constant term for idiosyncratic components
    lagca = similar(Y, T, narlag)
    crosslagca = similar(Y, narlag, narlag)
    arcoef = similar(Y, narlag, N)
    resid = similar(Y, T, N)
    σ = similar(Y, N)
    _dfm_est!(Y, facX, crossfacX, Λ, u, lagca, crosslagca, arcoef, resid, σ,
        narlag, arexclude)
    if FacProc !== nothing
        facproc = fit(FacProc, view(facX,:,1:nfac), nfaclag; facprockwargs...)
    else
        facproc = nothing
    end
    return DynamicFactor(Y, Y0, trans, facobs, facobs0, f, facX, crossfacX,
        Λ, u, facproc, lagca, crosslagca, arcoef, resid, σ, nfaclag, narlag)
end

function fit!(f::DynamicFactor; maxiter::Integer=10000, tol::Real=1e-8,
        arexclude=nothing, facprockwargs::NamedTuple=NamedTuple())
    detrend!(f.trans, f.Y0, f.Y)
    if f.facobs !== nothing
        detrend!(f.trans, f.facobs0, f.facobs)
        copyto!(view(f.f.fac,:,1:f.f.nfaco), f.facobs0)
    end
    fit!(f.f; maxiter=maxiter, tol=tol)
    nfac = size(f.f.fac, 2)
    invdetrend!(f.trans, view(f.facX,:,1:nfac), f.f.fac)
    _dfm_est!(f.Y, f.facX, f.crossfacX, f.Λ, f.u, f.lagca, f.crosslagca, f.arcoef, f.resid,
        f.σ, f.narlag, arexclude)
    f.facproc === nothing ||
        fit!(f.facproc, view(f.facX,:,1:nfac); facprockwargs...)
    return f
end

"""
    fit(::Type{<:DynamicFactor}, data, names, fonames, trans::AbstractDetrend,
        nfac::Union{Integer, AbstractNFactorCriterion},
        FacProc::Union{Type, Nothing}, nfaclag::Integer, narlag::Integer,
        X::Union{AbstractVecOrMat, Nothing}=nothing; kwargs...)

Fit a dynamic factor model with `nfac` factors using variables indexed by `names`
from a `Tables.jl`-compatible `data` table.
Factors that are observed are specified with `fonames`
as indices of columns from `data`.
When only unobserved factors are involved,
the number of factors may be selected based on a defined criterion.
The type of the model for the evolution of factors
may be specified with `FacProc`.
See also [`DynamicFactor`](@ref) and [`fit!`](@ref).

The meaning of the remaining arguments matches
the corresponding fields in [`DynamicFactor`](@ref).
All keyword arguments for fitting [`Factor`](@ref) are accepted.
The following are the additional keywords accepted.

# Keywords
- `arexclude=nothing`: set of indices for variables to be excluded from estimating
the autoregressive coefficients of the idiosyncratic components.
- `facprockwargs::NamedTuple=NamedTuple()`: keyword argument for `FacProc`.

# Reference
**Stock, James H. and Mark W. Watson.** 2016.
"Chapter 8---Dynamic Factor Models, Factor-Augmented Vector Autoregressions, and Structural Vector Autoregressions in Macroeconomics."
In *Handbook of Macroeconomics*, Vol. 2A,
edited by John B. Taylor and Harald Uhlig, 415--525. Amsterdam: Elsevier.
"""
function fit(::Type{<:DynamicFactor}, data, names, fonames, trans::AbstractDetrend,
        nfac::Union{Integer, AbstractNFactorCriterion},
        FacProc::Union{Type, Nothing}, nfaclag::Integer, narlag::Integer,
        X::Union{AbstractVecOrMat, Nothing}=nothing;
        subset::Union{BitVector, Nothing}=nothing, TF::Type=Float64,
        svdalg::Algorithm=DivideAndConquer(), maxiter::Integer=10000, tol::Real=1e-8,
        arexclude=nothing, facprockwargs::NamedTuple=NamedTuple())
    Y, faco = _factor_tabletomat(data, names, fonames, subset, TF)
    X !== nothing && subset !== nothing && (X = view(X, subset, :))
    return DynamicFactor(Y, faco, trans, nfac, FacProc, nfaclag, narlag, X;
        svdalg=svdalg, maxiter=maxiter, tol=tol,
        arexclude=arexclude, facprockwargs=facprockwargs)
end

show(io::IO, f::DynamicFactor) =
    print(io, size(f.f.fac,1), '×', size(f.f.fac,2), " ", typeof(f))

function show(io::IO, ::MIME"text/plain", f::DynamicFactor)
    nfac = nfactor(f)
    nfo = f.f.nfaco
    nfu = nfac - nfo
    print(io, f, " with ", nfu, " unobserved factor")
    nfu > 1 && print(io, "s")
    print(io, " and ", nfo, " observed factor")
    nfo > 1 && print(io, "s")
    println(io, ":")
    M, N = displaysize(io)
    NF = min(max(f.narlag+4, 5), 10, floor(Int, M/2))
    Base.print_matrix(IOContext(io, :compact=>true, :limit=>true,
        :displaysize=>(M-4-NF+5, N)), view(f.facX,:,1:nfac), "  ")
    print(io, "\n ", " Idiosyncratic AR coefficients for ", f.narlag, " lag")
    println(io, f.narlag>1 ? "s:" : ":")
    Base.print_matrix(IOContext(io, :compact=>true, :limit=>true,
        :displaysize=>(NF, N)), f.arcoef, "  ")
    println(io, "\n Evolution of factors:")
    print(io, "  ", f.facproc === nothing ? "Not estimated" : f.facproc)
end
