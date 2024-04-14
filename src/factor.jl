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
- `esample::BitVector`: indicators for rows involved in estimation from original data table when constructing `Y`; irrelevant to estimation once `Y` is given.
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
    esample::BitVector
end

response(f::Factor) = f.Y
modelmatrix(f::Factor) = f.fac
coef(f::Factor) = f.Λ
residuals(f::Factor) = f.resid
r2(f::Factor) = f.r2

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
        tss, rss, r2; maxiter::Integer=10000, tol::Real=1e-8)
    nfac = size(fac, 2)
    nfu = nfac - nfo
    rfu = nfo+1:nfac
    _standardize!(Ystd, Ysd, Y)
    if nfu > 0
        # SVD always overwrites the matrix
        copyto!(Yca, Ystd)
        # Throw error if Ystd contains NaN
        U, S, VT = _svd!(svdca, Yca)
        fac[:,rfu] .= view(U,:,1:nfu) .* view(S,1:nfu)'
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

function Factor(Y::AbstractMatrix, facobs::Union{AbstractMatrix, Nothing},
        nfac::Integer, esample::BitVector;
        svdalg::Algorithm=default_svd_alg(Y), maxiter::Integer=10000, tol::Real=1e-8)
    # ! To Do: Allow unbalanced panel and restrictions on loading coefficients
    nfac > 0 || throw(ArgumentError("nfac must be positive"))
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
    end
    Y = convert(Matrix, Y)
    T, N = size(Y)
    Ystd = similar(Y)
    Ysd = similar(Y, N)
    Yca = similar(Ystd)
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
    svdca = nfu > 0 ? svdcache(svdalg, Ystd) : nothing
    _factor!(Y, Ystd, Ysd, Yca, fac, crossfac, Λ, crossΛu, svdca, nfo, resid, tss, rss, r2;
        maxiter=maxiter, tol=tol)
    return Factor(Y, Ystd, Ysd, Yca, fac, crossfac, Λ, crossΛu, svdca, nfo, resid,
        tss, rss, r2, esample)
end

Factor(Y::AbstractMatrix, nfac::Integer, esample::BitVector; kwargs...) =
    Factor(Y, nothing, nfac, esample; kwargs...)

"""
    fit(::Type{<:Factor}, data, names, fonames, nfac::Integer; kwargs...)

Fit a factor model with `nfac` factors using variables indexed by `names`
from a `Tables.jl`-compatible `data` table.
Factors that are observed are specified with `fonames`
as indices of columns from `data`.
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
function fit(::Type{<:Factor}, data, names, fonames, nfac::Integer;
        subset::Union{BitVector, Nothing}=nothing, TF::Type=Float64,
        svdalg::Algorithm=DivideAndConquer(), maxiter::Integer=10000, tol::Real=1e-8)
    checktable(data)
    names isa Symbol && (names = (names,))
    N = length(names)
    N >= nfac || throw(ArgumentError("number of columns must be at least $nfac"))
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
    return Factor(Y, faco, nfac, subset; svdalg=svdalg, maxiter=maxiter, tol=tol)
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
        f.nfaco, f.resid, f.tss, f.rss, f.r2; maxiter=maxiter, tol=tol)
    return f
end

show(io::IO, f::Factor) =
    print(io, size(f.fac,1), '×', size(f.fac,2), " ", typeof(f))

function show(io::IO, ::MIME"text/plain", f::Factor)
    nfo = f.nfaco
    nfu = size(f.fac,2) - nfo
    print(io, f, " with ", nfu, " unobserved factor")
    nfu > 1 && print(io, "s")
    print(io, " and ", nfo, " observed factor")
    nfo > 1 && print(io, "s")
    println(io, ":")
    M, N = displaysize(io)
    nfac = size(f.fac, 2)
    NF = min(max(nfac+4, 5), 10, floor(Int, M/2))
    Base.print_matrix(IOContext(io, :compact=>true, :limit=>true,
        :displaysize=>(M-4-NF+5, N)), f.fac, "  ")
    println(io, "\n ", size(f.Λ,1), '×', size(f.Λ,2), " Loading matirx:")
    Base.print_matrix(IOContext(io, :compact=>true, :limit=>true,
        :displaysize=>(NF, N)), f.Λ, "  ")
    println(io, "\n R-squared by column:")
    Base.print_matrix(IOContext(io, :compact=>true, :limit=>true), f.r2', "  ")
end
