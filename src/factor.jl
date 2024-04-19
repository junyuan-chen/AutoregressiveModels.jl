"""
    AbstractNFactorCriterion

Types for specifying the method for determining the number of unobserved factors.
"""
abstract type AbstractNFactorCriterion end

ICp2penalty(N::Integer, T::Integer) = (N + T) / (N * T) * log(min(N, T))

"""
    BaiNg{P} <: AbstractNFactorCriterion

Criterion of the form defined in Bai and Ng (2002) with penalty function of type `P`.
The default is to use the ICp2 criterion with `ICp2penalty`.

# Reference
**Bai, Jushan and Serena Ng.** 2002.
"Determining the Number of Factors in Approximate Factor Models."
*Econometrica* 70 (1): 191--221.
"""
struct BaiNg{P} <: AbstractNFactorCriterion
    nfacmax::Int
    p::P
    BaiNg(nfacmax, p=ICp2penalty) = new{typeof(p)}(nfacmax, p)
end

criterion(c::BaiNg, nfac::Integer, s::AbstractFloat, N::Integer, T::Integer) =
    log(1 - s) + nfac * c.p(N, T)

function nfactor(c::BaiNg, S::AbstractVector, N::Integer, T::Integer)
    # Assume S contains valid singular values but still check the order
    issorted(S, rev=true) || throw(ArgumentError(
        "singular values in S must be sorted in reverse order"))
    Ssum = sum(abs2, S)
    cmin = Inf
    nfac = 0
    s = 0.0
    for n in 0:min(c.nfacmax, length(S))
        n > 0 && (s += (S[n])^2)
        cn = criterion(c, n, s/Ssum, N, T)
        if cn < cmin
            cmin = cn
            nfac = n
        end
    end
    return nfac
end

"""
    Factor{TF, S} <: RegressionModel

Results and cache from estimating a factor model.
This object holds all arrays that are necessary for
estimating the model without additional memory allocations.
See also [`fit`](@ref) and [`fit!`](@ref).

# Fields
- `Y::Matrix{TF}`: data matrix where each column corresponds to a time series.
- `Ystd::Matrix{TF}`: standardized data with zero mean and unit standard deviation across the columns.
- `Ysd::Vector{TF}`: standard deviation of each column in `Y`.
- `Yca::Matrix{TF}`: cache of the same size as `Y`.
- `fac::Matrix{TF}`: factors of the model; observed factors are placed in the beginning columns.
- `crossfac::Matrix{TF}`: cache for holding `fac'fac` and factorization.
- `Λ::Matrix{TF}`: loading matrix of the model; each row corresponds to a factor in `fac`.
- `crossΛu::Union{Matrix{TF}, Nothing}`: cache needed when factors and loading matrix need to be solved iteratively.
- `svdca::Union{SDDcache{TF}, SVDcache{TF}, Nothing}`: cache for singular value decomposition if any unobserved factor is involved.
- `nfaco::Int`: number of observed factors.
- `resid::Matrix{TF}`: residuals from the standardized data `Ystd` and estimated `fac` before scaling the loading matrix `Λ` for `Y`.
- `tss::Vector{TF}`: total sum of squares for each columns of `Ystd`.
- `rss::Vector{TF}`: residual sum of squares for each columns of `Ystd`.
- `r2::Vector{TF}`: r-squared for each columns of `Ystd`.
"""
struct Factor{TF<:AbstractFloat, S<:Union{SDDcache{TF}, SVDcache{TF}, Nothing}} <: RegressionModel
    Y::Matrix{TF}
    Ystd::Matrix{TF}
    Ysd::Vector{TF}
    Yca::Matrix{TF}
    fac::Matrix{TF}
    crossfac::Matrix{TF}
    Λ::Matrix{TF}
    crossΛu::Union{Matrix{TF}, Nothing}
    svdca::S
    nfaco::Int
    resid::Matrix{TF}
    tss::Vector{TF}
    rss::Vector{TF}
    r2::Vector{TF}
end

response(f::Factor) = f.Y
modelmatrix(f::Factor) = f.fac
coef(f::Factor) = f.Λ
residuals(f::Factor) = f.resid
r2(f::Factor) = f.r2

nfactor(f::Factor) = size(f.fac, 2)

# Assume balanced panel (no missing/nan)
function _standardize!(out::AbstractMatrix, sd::AbstractVector, Y::AbstractMatrix)
    T = size(Y, 1)
    @inbounds for (j, y) in enumerate(eachcol(Y))
        m = sum(y) / T
        std = sqrt(sum(x -> (x - m)^2, y) / T)
        out[:,j] .= (y .- m) ./ std
        sd[j] = std
    end
    return out, sd
end

function _factor!(Y, Ystd, Ysd, Yca, fac, crossfac, Λ, crossΛu, svdca, nfo, resid,
        tss, rss, r2, computestdsvd; maxiter::Integer=10000, tol::Real=1e-8)
    nfac = size(fac, 2)
    nfu = nfac - nfo
    rfu = nfo+1:nfac
    computestdsvd && _standardize!(Ystd, Ysd, Y)
    if nfu > 0
        if computestdsvd
            # SVD always overwrites the matrix
            copyto!(Yca, Ystd)
            # Throw error if Ystd contains NaN
            _svd!(svdca, Yca)
        end
        fac[:,rfu] .= view(svdca.U,:,1:nfu) .* view(svdca.S,1:nfu)'
    end
    # The simple case with balanced panel that does not require a loop
    if nfo == 0 || nfu == 0
        mul!(crossfac, fac', fac)
        # Must use Ystd instead of Y when all factors are observed
        # Y vs Ystd affects r2 but not Λ if all factors are unobserved
        mul!(Λ, fac', Ystd)
        ldiv!(cholesky!(crossfac), Λ)
        # Use Ystd instead of Y for residuals before rescaling Λ
        copyto!(resid, Ystd)
        mul!(resid, fac, Λ, -1.0, 1.0)
        # Rescale Λ after the residuals are computed
        Λ .*= Ysd'
    else
        ssrold = 0.0
        Yu = Yca
        for iter in 1:maxiter
            # Compute loading matrix given factors
            mul!(crossfac, fac', fac)
            mul!(Λ, fac', Ystd)
            ldiv!(cholesky!(crossfac), Λ)
            # Compute factors given loading matrix
            copyto!(Yu, Ystd)
            mul!(Yu, view(fac,:,1:nfo), view(Λ,1:nfo,:), -1.0, 1.0)
            Λu = view(Λ, rfu, :)
            mul!(crossΛu, Λu, Λu')
            facu = view(fac,:,rfu)
            mul!(facu, Yu, Λu')
            # No such a method on Julia v1.6
            ldiv!(cholesky!(crossΛu), facu')
            copyto!(resid, Ystd)
            resid = mul!(resid, fac, Λ, -1.0, 1.0)
            ssr = sum(abs2, resid)
            abs(ssr - ssrold) < tol*length(Y) && break
            ssrold = ssr
        end
        # Rescale Λ after residuals are computed
        Λ .*= Ysd'
    end
    for i in 1:size(Ystd, 2)
        # ! Use Ystd instead of Y for r2
        tss[i] = t = sum(abs2, view(Ystd,:,i))
        rss[i] = r = sum(abs2, view(resid,:,i))
        r2[i] = 1.0 - r / t
    end
    return nothing
end

function Factor(Y::AbstractMatrix, facobs::Union{AbstractVecOrMat, Nothing},
        nfac::Union{Integer, AbstractNFactorCriterion};
        svdalg::Algorithm=default_svd_alg(Y), maxiter::Integer=10000, tol::Real=1e-8)
    # ! To Do: Allow unbalanced panel and restrictions on loading coefficients
    Y = convert(Matrix, Y)
    T, N = size(Y)
    Ystd = similar(Y)
    Ysd = similar(Y, N)
    Yca = similar(Ystd)
    if nfac isa Integer
        nfac > 0 || throw(ArgumentError("nfac must be positive"))
        computestdsvd = true
    else
        facobs === nothing || throw(ArgumentError(
            "selection of factor number is not supported with observed factors"))
        _standardize!(Ystd, Ysd, Y)
        copyto!(Yca, Ystd)
        svdca = svdcache(svdalg, Ystd)
        U, S, VT = _svd!(svdca, Yca)
        nfac = nfactor(nfac, S, N, T)
        nfac > 0 || error(
            "there is no factor structure based on the selection criterion")
        computestdsvd = false
    end
    nfac > min(T, N) && throw(ArgumentError(
        "number of factors cannot be greater than $(min(T, N)) given the size of Y"))
    if facobs === nothing
        nfo = 0
        nfu = nfac
    else
        nfo = size(facobs, 2)
        nfu = nfac - nfo
        nfu < 0 && throw(ArgumentError(
            "number of observed factors ($nfo) is greater than nfac ($nfac)"))
        any(isnan, facobs) && throw(ArgumentError(
            "NaN is not allowed for observed factors"))
        size(facobs, 1) == T || throw(DimensionMismatch(
            "facobs and Y do not have the same number of rows"))
    end
    fac = similar(Y, T, nfac)
    if nfo > 0
        fac[:,1:nfo] .= facobs
    end
    crossfac = similar(Y, nfac, nfac)
    Λ = similar(Y, nfac, N)
    resid = similar(Y)
    tss = similar(Y, N)
    rss = similar(Y, N)
    r2 = similar(Y, N)
    crossΛu = nfo == 0 || nfu == 0 ? nothing : similar(Y, nfu, nfu)
    if computestdsvd
        svdca = nfu > 0 ? svdcache(svdalg, Ystd) : nothing
    end
    _factor!(Y, Ystd, Ysd, Yca, fac, crossfac, Λ, crossΛu, svdca, nfo, resid, tss, rss, r2,
        computestdsvd; maxiter=maxiter, tol=tol)
    return Factor(Y, Ystd, Ysd, Yca, fac, crossfac, Λ, crossΛu, svdca, nfo, resid,
        tss, rss, r2)
end

Factor(Y::AbstractMatrix, nfac::Union{Integer, AbstractNFactorCriterion}; kwargs...) =
        Factor(Y, nothing, nfac; kwargs...)

function _factor_tabletomat(data, names, fonames, subset, TF)
    checktable(data)
    names isa Symbol && (names = (names,))
    N = length(names)
    _checknames(names) || throw(
        ArgumentError("invalid names; must be integers or `Symbol`s"))
    if fonames !== nothing
        fonames isa Symbol && (fonames = (fonames,))
        nfo = length(fonames)
        if nfo == 0
            fonames = nothing
        else
            _checknames(fonames) || throw(
                ArgumentError("invalid fonames; must be integers or `Symbol`s"))
        end
    end
    Tfull = Tables.rowcount(data)
    subset === nothing || length(subset) == Tfull ||
        throw(ArgumentError("length of subset ($(length(subset))) does not match the number of rows in data ($Tfull)"))
    T = subset === nothing ? Tfull : sum(subset)
    subset === nothing && (subset = :)
    # ! missing, nan and inf are not handled
    Y = Matrix{TF}(undef, T, N)
    for i in 1:N
        Y[:,i] .= view(getcolumn(data, names[i]), subset)
    end
    if fonames === nothing
        faco = nothing
    else
        faco = Matrix{TF}(undef, T, nfo)
        for i in 1:nfo
            faco[:,i] .= view(getcolumn(data, fonames[i]), subset)
        end
    end
    return Y, faco
end

"""
    fit(::Type{<:Factor}, data, names, fonames,
        nfac::Union{Integer, AbstractNFactorCriterion}; kwargs...)

Fit a factor model with `nfac` factors using variables indexed by `names`
from a `Tables.jl`-compatible `data` table.
Factors that are observed are specified with `fonames`
as indices of columns from `data`.
When only unobserved factors are involved,
the number of factors may be selected based on a defined criterion.
R-squared for each variable is computed based on standardized data
with zero mean and unit standard deviation.
See also [`Factor`](@ref) and [`fit!`](@ref).

# Keywords
- `subset::Union{BitVector, Nothing}=nothing`: subset of `data` to be used for estimation.
- `TF::Type=Float64`: numeric type used for estimation.
- `svdalg::Algorithm=DivideAndConquer()`: algorithm for singular value decomposition.
- `maxiter::Integer=10000`: maximum number of iterations; only relevant when the model involves both observed and unobserved factors.
- `tol::Real=1e-8`: tolerance level for convergence; only relevant when the model involves both observed and unobserved factors.

# Reference
**Stock, James H. and Mark W. Watson.** 2016.
"Chapter 8---Dynamic Factor Models, Factor-Augmented Vector Autoregressions, and Structural Vector Autoregressions in Macroeconomics."
In *Handbook of Macroeconomics*, Vol. 2A,
edited by John B. Taylor and Harald Uhlig, 415--525. Amsterdam: Elsevier.
"""
function fit(::Type{<:Factor}, data, names, fonames,
        nfac::Union{Integer, AbstractNFactorCriterion};
        subset::Union{BitVector, Nothing}=nothing, TF::Type=Float64,
        svdalg::Algorithm=DivideAndConquer(), maxiter::Integer=10000, tol::Real=1e-8)
    Y, faco = _factor_tabletomat(data, names, fonames, subset, TF)
    return Factor(Y, faco, nfac; svdalg=svdalg, maxiter=maxiter, tol=tol)
end

"""
    fit!(f::Factor; maxiter::Integer=10000, tol::Real=1e-8)

Reestimate the factor model with data contained in `f`.
This method assumes that the content in `f` remains valid after any modification
and allows non-allocating estimation.
See also [`Factor`](@ref) and [`fit`](@ref).
"""
function fit!(f::Factor; maxiter::Integer=10000, tol::Real=1e-8)
    _factor!(f.Y, f.Ystd, f.Ysd, f.Yca, f.fac, f.crossfac, f.Λ, f.crossΛu, f.svdca,
        f.nfaco, f.resid, f.tss, f.rss, f.r2, true; maxiter=maxiter, tol=tol)
    return f
end

show(io::IO, f::Factor) =
    print(io, size(f.fac,1), '×', size(f.fac,2), " ", typeof(f))

function show(io::IO, ::MIME"text/plain", f::Factor)
    nfac = nfactor(f)
    nfo = f.nfaco
    nfu = nfac - nfo
    print(io, f, " with ", nfu, " unobserved factor")
    nfu > 1 && print(io, "s")
    print(io, " and ", nfo, " observed factor")
    nfo > 1 && print(io, "s")
    println(io, ":")
    M, N = displaysize(io)
    NF = min(max(nfac+4, 5), 10, floor(Int, M/2))
    Base.print_matrix(IOContext(io, :compact=>true, :limit=>true,
        :displaysize=>(M-4-NF+5, N)), f.fac, "  ")
    println(io, "\n ", size(f.Λ,1), '×', size(f.Λ,2), " Loading matirx:")
    Base.print_matrix(IOContext(io, :compact=>true, :limit=>true,
        :displaysize=>(NF, N)), f.Λ, "  ")
    println(io, "\n R-squared by column:")
    Base.print_matrix(IOContext(io, :compact=>true, :limit=>true), f.r2', "  ")
end
