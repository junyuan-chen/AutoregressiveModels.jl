"""
    OLS{TF<:AbstractFloat} <: RegressionModel

Data from an ordinary least squares regression.
"""
struct OLS{TF<:AbstractFloat, TC<:Union{Matrix{TF},Nothing}} <: RegressionModel
    X::Matrix{TF}
    crossXcache::Matrix{TF}
    coef::Matrix{TF}
    coefcorrected::TC
    resid::Matrix{TF}
    residvcov::Matrix{TF}
    esample::BitVector
    dofr::Int
end

function OLS(Y::AbstractMatrix, X::AbstractMatrix, esample::BitVector, dofr::Int)
    X = convert(Matrix, X)
    crossX = X'X
    coef = ldiv!(cholesky!(crossX), X'Y)
    resid = mul!(Y, X, coef, -1.0, 1.0)
    dofr = max(1, dofr)
    residvcov = rdiv!(resid'resid, dofr)
    return OLS(X, crossX, coef, nothing, resid, residvcov, esample, dofr)
end

modelmatrix(m::OLS) = m.X
coef(m::OLS) = m.coef
residuals(m::OLS) = m.resid
dof_residual(m::OLS) = m.dofr

coefcorrected(m::OLS) = m.coefcorrected
residvcov(m::OLS) = m.residvcov

show(io::IO, ols::OLS) = print(io, "OLS regression ", size(residuals(ols)))

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
    VectorAutoregression <: StatisticalModel

Results from vector autoregression estimation.
"""
struct VectorAutoregression{TE} <: StatisticalModel
    est::TE
    names::Vector{Symbol}
    lookup::Dict{Symbol,Int}
    nocons::Bool
end

modelmatrix(r::VectorAutoregression) = modelmatrix(r.est)
coef(r::VectorAutoregression) = coef(r.est)
coefcorrected(r::VectorAutoregression) = coefcorrected(r.est)
residvcov(r::VectorAutoregression) = residvcov(r.est)
residuals(r::VectorAutoregression) = residuals(r.est)

nvar(r::VectorAutoregression) = length(r.names)
arorder(r::VectorAutoregression) = size(coef(r), 1) ÷ size(coef(r), 2)
hasintercept(r::VectorAutoregression) = !r.nocons

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
        i0 = r.nocons ? 0 : 1
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

function VARProcess(r::VectorAutoregression; copy::Bool=false)
    B = coef(r)'
    if size(B, 2) % size(B, 1) == 1
        return VARProcess(view(B,:,2:size(B,2)), view(B,:,1), copy=copy)
    else
        return VARProcess(B, copy=copy)
    end
end

_checknames(names) = all(n isa Union{Integer, Symbol} for n in names)

_toname(data, name::Symbol) = name
_toname(data, i::Integer) = Tables.columnnames(data)[i]

function fit(::Type{<:VARProcess}, data, names, nlag::Integer;
        subset::Union{BitVector, Nothing}=nothing,
        nocons::Bool=false, TF::Type=Float64)
    checktable(data)
    names isa Symbol && (names = (names,))
    N = length(names)
    N > 0 || throw(ArgumentError("names cannot be empty"))
    _checknames(names) || throw(
        ArgumentError("invalid names; must be integers or `Symbol`s"))
    names = Symbol[_toname(data, n) for n in names]

    Tfull = Tables.rowcount(data)
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
    dofr = T - size(X,2)
    m = OLS(Y, X, esampleT, dofr)
    return VectorAutoregression(m, names,
        Dict{Symbol,Int}(n=>i for (i,n) in enumerate(names)), nocons)
end

simulate!(εs::AbstractArray, r::VectorAutoregression, Y0::AbstractArray;
    nlag::Integer=arorder(r)) = simulate!(εs, VARProcess(r), Y0, nlag=nlag)

simulate!(εs::AbstractArray, r::VectorAutoregression; nlag::Integer=arorder(r)) =
    simulate!(εs, VARProcess(r), nlag=nlag)

impulse!(out::AbstractArray, r::VectorAutoregression, ε0::AbstractVecOrMat;
    nlag::Integer=arorder(r)) = impulse!(out, VARProcess(r), ε0, nlag=nlag)

impulse(r::VectorAutoregression, ε0::AbstractVecOrMat, nhorz::Integer;
    nlag::Integer=arorder(r)) = impulse(VARProcess(r), ε0, nhorz, nlag=nlag)

_isstable!(C::AbstractMatrix, offset::Real) =
    abs(eigen!(C, sortby=abs).values[end]) < 1 - offset

function _biasrelax(δ, B, C, b, T, Blast, offset)
    B .= C .+ δ .* b ./ T
    copyto!(Blast, B)
    return abs(eigen!(Blast, sortby=abs).values[end]) - 1 + offset
end

function biascorrect(r::VectorAutoregression;
        offset::Real=1e-4, factor::Union{Real,Nothing}=nothing,
        x0=(0,0.999), solver=Brent(), xatol=1e-9, kwargs...)
    var = VARProcess(r)
    isstable(var, offset) || return r
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
        if !_isstable!(Blast, offset)
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
    olsc = OLS(m.X, m.crossXcache, m.coef, coefc, m.resid, m.residvcov, m.esample, m.dofr)
    return VectorAutoregression(olsc, r.names, r.lookup, r.nocons), δ
end

