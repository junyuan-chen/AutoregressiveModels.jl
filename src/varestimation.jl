"""
    VAROLS <: RegressionModel

Data from an ordinary least squares regression for vector autoregression estimation.
"""
struct VAROLS{TF<:AbstractFloat, TI<:Union{Vector{TF}, Nothing},
        TC<:Union{Matrix{TF}, Nothing},
        CF<:Union{Cholesky{TF, Matrix{TF}}, Nothing},
        CL<:Union{Matrix{TF}, Nothing}} <: RegressionModel
    Y::Matrix{TF}
    X::Matrix{TF}
    crossXcache::Matrix{TF}
    coef::Matrix{TF}
    coefB::Matrix{TF}
    intercept::TI
    coefcorrected::TC
    resid::Matrix{TF}
    residvcov::Matrix{TF}
    residchol::CF
    residcholL::CL
    esample::Union{BitVector, Nothing}
    dofr::Int
end

response(m::VAROLS) = m.Y
modelmatrix(m::VAROLS) = m.X
coef(m::VAROLS) = m.coef
residuals(m::VAROLS) = m.resid
dof_residual(m::VAROLS) = m.dofr

coefB(m::VAROLS) = m.coefB
intercept(m::VAROLS) = m.intercept
coefcorrected(m::VAROLS) = m.coefcorrected
residvcov(m::VAROLS) = m.residvcov
# Return the copied lower-triangular matrix instead of factorization
residchol(m::VAROLS) = m.residcholL

nvar(m::VAROLS) = size(response(m), 2)
arorder(m::VAROLS) = size(coefB(m), 2) ÷ size(coefB(m), 1)
hasintercept(::VAROLS{TF, Vector{TF}}) where TF = true
hasintercept(::VAROLS{TF, Nothing}) where TF = false

function VAROLS(Y::AbstractMatrix, X::AbstractMatrix, esample::Union{BitVector, Nothing},
        dofr::Int, nocons::Bool, cholresid::Bool)
    Y = convert(Matrix, Y)
    X = convert(Matrix, X)
    crossX = X'X
    coef = ldiv!(cholesky!(crossX), X'Y)
    # Make copies for VARProcess
    if nocons
        coefB = collect(coef')
        intercept = nothing
    else
        coefB = coef'[:,2:end]
        intercept = coef[1,:]
    end
    resid = copy(Y)
    resid = mul!(resid, X, coef, -1.0, 1.0)
    dofr = max(1, dofr)
    residvcov = rdiv!(resid'resid, dofr)
    if cholresid
        residchol = cholesky(residvcov)
        Cfactors = getfield(residchol, :factors)
        Cuplo = getfield(residchol, :uplo)
        residcholL = collect(Cuplo === 'U' ?
            UpperTriangular(Cfactors)' : LowerTriangular(Cfactors))
    else
        residchol = nothing
        residcholL = nothing
    end
    return VAROLS(Y, X, crossX, coef, coefB, intercept, nothing, resid, residvcov, residchol,
        residcholL, esample, dofr)
end

function fit!(m::VAROLS{TF}; Yinresid::Bool=false) where TF
    X, coef, resid = m.X, m.coef, m.resid
    Y = Yinresid ? m.resid : m.Y
    mul!(coef, X', Y)
    mul!(m.crossXcache, X', X)
    ldiv!(cholesky!(m.crossXcache), coef)
    if m.intercept === nothing
        copyto!(m.coefB, coef')
    else
        copyto!(m.coefB, view(coef, 2:size(coef,1), :)')
        copyto!(m.intercept, view(coef, 1, :))
    end
    Yinresid || copyto!(resid, Y)
    mul!(resid, X, coef, -1.0, 1.0)
    mul!(m.residvcov, resid', resid)
    rdiv!(m.residvcov, m.dofr)
    residchol = m.residchol
    residcholL = m.residcholL
    if residchol !== nothing
        Cfactors = getfield(residchol, :factors)
        Cuplo = getfield(residchol, :uplo)
        copyto!(Cfactors, m.residvcov)
        # residchol could involve changes in immutable objects but it is not used
        residchol = cholesky!(Cfactors)
        N = size(m.residvcov, 1)
        @inbounds for j in 1:N
            for i in 1:j-1
                residcholL[i,j] = zero(TF)
            end
            for i in j:N
                residcholL[i,j] = Cuplo === 'U' ? Cfactors[j,i] : Cfactors[i,j]
            end
        end
    end
    return nothing
end

# Assume no missing/NaN in Yfull
function _fillYX!(Y, X, Yfull::AbstractMatrix, nlag::Integer, nocons::Bool)
    j0 = nocons ? 0 : 1
    Tfull, N = size(Yfull)
    for j in 1:N
        Y[:,j] .= view(Yfull, nlag+1:Tfull, j)
        for l in 1:nlag
            # Variables with the same lag are put together
            # The first column in X may be the intercept
            X[:,j0+(l-1)*N+j] .= view(Yfull, nlag+1-l:Tfull-l, j)
        end
    end
end

function fit(::Type{<:Union{<:VARProcess, <:VAROLS}}, data::AbstractMatrix, nlag::Integer;
        choleskyresid::Bool=false, adjust_dofr::Bool=true,
        nocons::Bool=false, TF::Type=Float64)
    Tfull, N = size(data)
    T = Tfull - nlag
    Y = Matrix{TF}(undef, T, N)
    X = Matrix{TF}(undef, T, nocons ? N*nlag : 1+N*nlag)
    nocons || fill!(view(X, :, 1), one(TF))
    _fillYX!(Y, X, data, nlag, nocons)
    dofr = adjust_dofr ? T - size(X,2) : T
    m = VAROLS(Y, X, nothing, dofr, nocons, choleskyresid)
    return m
end

"""
    fit!(m::VAROLS, [data::AbstractMatrix]; kwargs...)

Reestimate vector autoregression of the same specification for `m`
with ordinary least squares and store results in `m` in-place.
Modified data for variables may be provided as columns of `data` matrix
where all data are valid for estimation (no `missing`, `NaN`, etc.)
and have the number of observations that matches `m`.
If `data` is not specified,
existing values stored in `m` are used.
See also [`fit`](@ref).

# Keywords
- `Yinresid::Bool=false`: store data in `residual(m)` for estimation; this avoids copying data from `response(m)` to `residual(m)`.
"""
function fit!(m::VAROLS, data::AbstractMatrix; Yinresid::Bool=false)
    nlag = arorder(m)
    Tfull, N = size(data)
    Tfull == size(response(m), 1) + nlag && N == nvar(m) ||
        throw(DimensionMismatch("size of data does not match m"))
    i0 = hasintercept(m) ? 1 : 0
    # if Yinresid=true, Y in VAROLS is left untouched but use resid in-place
    # This avoids the need of copying Y to resid for computing residuals
    Y = Yinresid ? residuals(m) : response(m)
    X = modelmatrix(m)
    # esample is not used here
    copyto!(Y, view(data, nlag+1:Tfull, :))
    # Assume the column for constant term remains untouched
    for l in 1:nlag
        copyto!(view(X, :, i0+(l-1)*N+1:i0+l*N), view(data, nlag+1-l:Tfull-l, :))
    end
    fit!(m; Yinresid=Yinresid)
end

"""
    VARProcess(m::VAROLS; copy::Bool=false)

Construct a `VARProcess` based on the coefficient estimates from `m`.
"""
function VARProcess(m::VAROLS; copy::Bool=false)
    if hasintercept(m)
        return VARProcess(coefB(m), intercept(m), copy=copy)
    else
        return VARProcess(coefB(m), copy=copy)
    end
end

function show(io::IO, ols::VAROLS)
    M, N = size(modelmatrix(ols))
    nv = nvar(ols)
    nlag = arorder(ols)
    print(io, M, '×', N, " OLS regression for VAR with ", nv, " variable")
    nv > 1 && print(io, "s")
    print(io, " and ", nlag, " lag")
    nlag > 1 && print(io, "s")
end

"""
    VectorAutoregression{TE, HasIntercept} <: RegressionModel

Results from vector autoregression estimation.
"""
struct VectorAutoregression{TE, HasIntercept} <: RegressionModel
    est::TE
    names::Vector{Symbol}
    lookup::Dict{Symbol,Int}
    VectorAutoregression(est, names::Vector{Symbol}, lookup::Dict{Symbol,Int},
        nocons::Bool) = new{typeof(est), !nocons}(est, names, lookup)
end

response(r::VectorAutoregression) = response(r.est)
modelmatrix(r::VectorAutoregression) = modelmatrix(r.est)
coef(r::VectorAutoregression) = coef(r.est)
residuals(r::VectorAutoregression) = residuals(r.est)
intercept(r::VectorAutoregression) = intercept(r.est)

"""
    coefcorrected(r::VectorAutoregression)

Return the coefficient estimates after bias correction.
See also [`biascorrect`](@ref).
"""
coefcorrected(r::VectorAutoregression) = coefcorrected(r.est)

"""
    residvcov(r::VectorAutoregression)

Return the variance-covariance matrix of the residuals.
"""
residvcov(r::VectorAutoregression) = residvcov(r.est)

"""
    residchol(r::VectorAutoregression)

Return the lower-triangular matrix from
Cholesky factorization of the residual variance-covariance matrix.
"""
residchol(r::VectorAutoregression) = residchol(r.est)

nvar(r::VectorAutoregression) = nvar(r.est)
arorder(r::VectorAutoregression) = arorder(r.est)
maorder(r::VectorAutoregression) = 0
hasintercept(::VectorAutoregression{TE, HI}) where {TE, HI} = HI

_varindex(r::VectorAutoregression, name::Symbol) = r.lookup[name]
_varindex(::VectorAutoregression, id::Integer) = Int(id)

"""
    coef(r::VectorAutoregression, yid, xid, lag::Integer=1)

Return the coefficient estimate of `r` for variable `yid` with respect to
variable `xid` with lag `lag`.
`yid` and `xid` can be variable names of type `Symbol` or integer indices.
Unless `var.nocons` is `true`,
the intercept term is always named `:constant` and has an index of `1`.
"""
Base.@propagate_inbounds function coef(r::VectorAutoregression, yid, xid, lag::Integer=1)
    j = _varindex(r, yid)
    if xid == :constant || xid == 1
        return coef(r)[1,j]
    else
        i0 = hasintercept(r) ? 1 : 0
        i = i0 + _varindex(r, xid) + (lag-1)*length(r.names)
        return coef(r)[i,j]
    end
end

"""
    residvcov(r::VectorAutoregression, id1, id2=id1)

Return the variance-covariance estimate between the reduced-form errors from equation `id1` and `id2`.
The equations are indexed by the outcome variables, which can be variable names of type `Symbol` or integer indices.
"""
Base.@propagate_inbounds residvcov(r::VectorAutoregression, id1, id2=id1) =
    residvcov(r)[_varindex(r, id1), _varindex(r, id2)]

"""
    VARProcess(r::VectorAutoregression; copy::Bool=false)

Construct a `VARProcess` based on the coefficient estimates from `r`.
"""
VARProcess(r::VectorAutoregression; copy::Bool=false) =
    VARProcess(r.est; copy=copy)

_checknames(names) = all(n isa Union{Integer, Symbol} for n in names)

_toname(data, name::Symbol) = name
_toname(data, i::Integer) = Tables.columnnames(data)[i]

function _fillYX!(Y, X, esampleT, aux, tb, idx, nlag, subset, nocons)
    i0 = nocons ? 0 : 1
    nx = length(idx)
    for j in 1:nx
        col = getcolumn(tb, idx[j])
        Tfull = length(col)
        v = view(col, nlag+1:Tfull)
        subset === nothing || j > 1 || (esampleT .&= view(subset, nlag+1:Tfull))
        _esample!(esampleT, aux, v)
        copyto!(view(Y, esampleT, j), view(v, esampleT))
        for l in 1:nlag
            v = view(col, nlag+1-l:Tfull-l)
            subset === nothing || j > 1 ||
                (esampleT .&= view(subset, nlag+1-l:Tfull-l))
            _esample!(esampleT, aux, v)
            # Variables with the same lag are put together
            # The first column in X may be the intercept
            copyto!(view(X, esampleT, i0+(l-1)*nx+j), view(v, esampleT))
        end
    end
    return esampleT
end

"""
    fit(::Type{<:VARProcess}, data, names, nlag::Integer; kwargs...)
    fit(::Type{<:Union{<:VARProcess, <:VAROLS}}, data::AbstractMatrix, nlag::Integer; kwargs...)

Estimate vector autoregression with ordinary least squares
using `nlag` lags of variables indexed by `names`
from a `Tables.jl`-compatible `data` table.
Alternatively, if all data are valid for estimation (no `missing`, `NaN`, etc.),
`data` can be a matrix only containing the relevant columns (no `names`).
See also [`fit!`](@ref).

# Keywords
- `subset::Union{BitVector, Nothing}=nothing`: subset of `data` to be used for estimation.
- `choleskyresid::Bool=false`: conduct Cholesky factorization for the residual variance-covariance matrix.
- `adjust_dofr::Bool=true`: adjust the degrees of freedom when computing the residual variance-covariance matrix.
- `nocons::Bool=false`: do not include an intercept term in the estimation.
- `TF::Type=Float64`: numeric type used for estimation.
"""
function fit(::Type{<:VARProcess}, data, names, nlag::Integer;
        subset::Union{BitVector, Nothing}=nothing,
        choleskyresid::Bool=false, adjust_dofr::Bool=true,
        nocons::Bool=false, TF::Type=Float64)
    checktable(data)
    names isa Symbol && (names = (names,))
    N = length(names)
    N > 0 || throw(ArgumentError("names cannot be empty"))
    _checknames(names) || throw(
        ArgumentError("invalid names; must be integers or `Symbol`s"))
    names = Symbol[_toname(data, n) for n in names]

    Tfull = Tables.rowcount(data)
    subset === nothing || length(subset) == Tfull ||
        throw(ArgumentError("length of subset ($(length(subset))) does not match the number of rows in data ($Tfull)"))
    T = Tfull - nlag
    Y = Matrix{TF}(undef, T, N)
    X = Matrix{TF}(undef, T, nocons ? N*nlag : 1+N*nlag)
    nocons || fill!(view(X, :, 1), one(TF))
    # Indicators for valid observations within the T rows
    esampleT = trues(T)
    # A cache for indicators
    aux = BitVector(undef, T)
    _fillYX!(Y, X, esampleT, aux, data, names, nlag, subset, nocons)
    T1 = sum(esampleT)
    if T1 < T
        T1 > size(X, 2) || throw(ArgumentError(
            "not enough observations ($T1) for estimation"))
        Y = Y[esampleT, :]
        X = X[esampleT, :]
    end
    T = T1
    dofr = adjust_dofr ? T - size(X,2) : T
    m = VAROLS(Y, X, esampleT, dofr, nocons, choleskyresid)
    return VectorAutoregression(m, names,
        Dict{Symbol,Int}(n=>i for (i,n) in enumerate(names)), nocons)
end

"""
    simulate!(εs::AbstractArray, r::VectorAutoregression, Y0=nothing; kwargs...)

Simulate the [`VARProcess`](@ref) with the coefficient estimates in `r`
using the shocks specified in `εs` and initial values `Y0`.
Results are stored by overwriting `εs` in-place.
If `Y0` is `nothing` or does not contain enough lags, zeros are used.
See also [`simulate`](@ref) and [`impulse!`](@ref).

# Keywords
- `nlag::Integer=arorder(var)`: the number of lags from `var` used for simulations.
"""
simulate!(εs::AbstractArray, r::VectorAutoregression, Y0=nothing; kwargs...) =
    simulate!(εs, VARProcess(r), Y0; kwargs...)

"""
    simulate(εs::AbstractArray, r::VectorAutoregression, Y0=nothing; kwargs...)

Same as [`simulate!`](@ref), but makes a copy of `εs` for results
and hence does not overwrite `εs`.
"""
simulate(εs::AbstractArray, r::VectorAutoregression, Y0=nothing; kwargs...) =
    simulate(εs, VARProcess(r), Y0; kwargs...)

"""
    impulse!(out, r::VectorAutoregression, ε0::AbstractVecOrMat; kwargs...)
    impulse!(out, r::VectorAutoregression, ishock::Union{Integer, AbstractRange}; kwargs...)

Compute impulse responses to shocks specified with `ε0` or `ishock`
based on the estimation result in `r` and store the results in an array `out`.
The number of horizons to be computed is determined by the second dimension of `out`.
Responses to structural shocks identified based on temporal (short-run) restrictions
can be computed if Cholesky factorization has been conducted for `r`.
See also [`impulse`](@ref) and [`simulate!`](@ref).

As a vector, `ε0` specifies the magnitude of the impulse to each variable;
columns in a matrix are interpreted as multiple impulses with results
stored separately along the third dimension of array `out`.
Alternatively, `ishock` specifies the index of a single variable that is affected on impact;
a range of indices is intercepted as multiple impulses.
With the keyword `choleskyshock` being `true`,
any shock index specified is interpreted as referring to the structural shock
instead of the reduced-form shock.

# Keywords
- `nlag::Integer=arorder(var)`: the number of lags from `var` used for simulations.
- `choleskyshock::Bool=false`: whether any shock index refers to a structural shock.
"""
impulse!(out::AbstractArray, r::VectorAutoregression, ε0::AbstractVecOrMat;
    nlag::Integer=arorder(r)) = impulse!(out, VARProcess(coefB(r.est)), ε0, nlag=nlag)

function impulse!(out::AbstractArray, r::VectorAutoregression,
        ishock::Union{Integer, AbstractRange};
        nlag::Integer=arorder(r), choleskyshock::Bool=false)
    if choleskyshock
        chol = residchol(r)
        chol === nothing && throw(ArgumentError(
            "Cholesky factorization is not taken for r; see the choleskyresid option of fit"))
        # view allocates if array dimension changes
        ishock isa Integer && (ishock = ishock:ishock)
        ε0 = view(residchol(r), :, ishock)
        impulse!(out, VARProcess(coefB(r.est)), ε0; nlag=nlag)
    else
        impulse!(out, VARProcess(coefB(r.est)), ishock; nlag=nlag)
    end
end

"""
    impulse(r::VectorAutoregression, ε0::AbstractVecOrMat, nhorz::Integer; kwargs...)
    impulse(r::VectorAutoregression, ishock::Union{Integer, AbstractRange}, nhorz::Integer; kwargs...)

$_impulse_common_docstr
"""
impulse(r::VectorAutoregression, ε0::AbstractVecOrMat, nhorz::Integer;
    nlag::Integer=arorder(r)) = impulse(VARProcess(coefB(r.est)), ε0, nhorz, nlag=nlag)

function impulse(r::VectorAutoregression, ishock::Union{Integer, AbstractRange},
        nhorz::Integer; nlag::Integer=arorder(r), choleskyshock::Bool=false)
    if choleskyshock
        chol = residchol(r)
        chol === nothing && throw(ArgumentError(
            "Cholesky factorization is not taken for r; see the choleskyresid option of fit"))
        ishock isa Integer && (ishock = ishock:ishock)
        ε0 = view(residchol(r), :, ishock)
        impulse(VARProcess(coefB(r.est)), ε0, nhorz; nlag=nlag)
    else
        impulse(VARProcess(coefB(r.est)), ishock, nhorz; nlag=nlag)
    end
end

_isstable!(C::AbstractMatrix, offset::Real) =
    abs(eigen!(C, sortby=abs).values[end]) < 1 - offset

function _biasrelax(δ, B, C, b, T, Blast, offset)
    B .= C .+ δ .* b ./ T
    copyto!(Blast, B)
    return abs(eigen!(Blast, sortby=abs).values[end]) - 1 + offset
end

"""
    biascorrect(r::VectorAutoregression; kwargs...)

Correct the bias of the least-squares coefficient estimates in `r`
with Pope's formula.
The corrected estimates can be retrieved from the returned object
with [`coefcorrected`](@ref).

# Reference
**Pope, Alun L.** 1990.
"Biases of Estimators in Multivariate Non-Gaussian Autoregressions."
*Journal of Time Series Analysis* 11 (3): 249--258.
"""
function biascorrect(r::VectorAutoregression;
        offset::Real=1e-4, factor::Union{Real, Nothing}=nothing,
        x0=(0, 0.999), solver=Brent(), xatol=1e-9,
        warnunstable::Bool=true, kwargs...)
    var = VARProcess(r)
    if !isstable(var, offset)
        warnunstable && @warn "VAR process is not stable with offset $(offset); bias correction is not conducted"
        return r, nothing
    end
    C = companionform(var)
    N, NP = size(var.B)
    G = zeros(NP, NP)
    copyto!(view(G, 1:N, 1:N), residvcov(r))
    Γ0 = lyapd(C, G)
    aux = zeros(ComplexF64, NP, NP)
    ldiv!(aux, lu!(I - C*C), C)
    aux .+= inv!(lu!(I - C))
    F = eigen(C)
    for λ in F.values
        mul!(aux, λ, inv!(lu!(I - λ * C)), 1.0, 1.0)
    end
    auxr = Matrix{Float64}(undef, NP, NP)
    copyto!(auxr, aux')
    b = G * rdiv!(auxr, lu!(Γ0))
    T = size(residuals(r), 1)
    B = C .+ b ./ T
    if factor === nothing
        Blast = copy(B)
        if _isstable!(Blast, offset)
            δ = 1.0
        else
            δ = find_zero(x->_biasrelax(x, B, C, b, T, Blast, offset), x0, solver;
                xatol=xatol, kwargs...)
        end
    else
        δ = factor
    end
    B .= C .+ δ .* b ./ T
    coefc = Matrix{Float64}(undef, NP, N)
    copyto!(coefc, view(B, 1:N, :)')
    m = r.est
    olsc = VAROLS(m.Y, m.X, m.crossXcache, m.coef, m.coefB, m.intercept, coefc, m.resid,
        m.residvcov, m.residchol, m.residcholL, m.esample, m.dofr)
    return VectorAutoregression(olsc, r.names, r.lookup, !hasintercept(r)), δ
end

show(io::IO, r::VectorAutoregression) =
    print(io, size(coef(r),2), '×', size(coef(r),1), " ", typeof(r))

function show(io::IO, ::MIME"text/plain", r::VectorAutoregression)
    println(io, r, " with coefficient matrix:")
    M, N = displaysize(io)
    Base.print_matrix(IOContext(io, :compact=>true, :limit=>true,
        :displaysize=>(M-1, N)), coef(r)')
end
