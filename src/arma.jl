"""
    ARMAProcess(ars, mas, intercept=nothing)

Return an autoregressive–moving-average process ``y_t`` defined as

```math
\\left(1-\\sum_{i=1}^p \\rho_i L^i\\right) = c + \\left(1-\\sum_{i=1}^q \\varphi_i L^i\\right) \\varepsilon_t
```
where ``\\{\\rho_i\\}_i``, ``\\{\\varphi_i\\}_i`` and ``c``
are specified with `mas`, `ars` and `intercept` respectively;
``L`` is the lag operator.
Notice the *negative* sign associated with the moving-average terms.
"""
struct ARMAProcess{TA<:NTuple, TM<:NTuple, TF<:Union{Real, Nothing}}
    ars::TA
    mas::TM
    intercept::TF
    function ARMAProcess(ars::Union{Real, Tuple, AbstractVector, Nothing},
            mas::Union{Real, Tuple, AbstractVector, Nothing},
            intercept::Union{Real, Nothing}=nothing)
        ars1 = ars === nothing ? () : (ars...,)
        mas1 = mas === nothing ? () : (mas...,)
        return new{typeof(ars1), typeof(mas1), typeof(intercept)}(ars1, mas1, intercept)
    end
end

nvar(arma::ARMAProcess) = 1
arorder(::ARMAProcess{<:NTuple{N}}) where N = N
maorder(::ARMAProcess{<:Any,<:NTuple{N}}) where N = N
hasintercept(::ARMAProcess{<:Any,<:Any,<:Real}) = true
hasintercept(::ARMAProcess{<:Any,<:Any,Nothing}) = false

"""
    simulate!(out::AbstractArray, εs::AbstractArray, arma::ARMAProcess, y0=nothing)

Simulate the `arma` process using the shocks specified in `εs` and initial values `y0`.
Results are stored in `out`.
If `y0` is `nothing` or does not contain enough lags, zeros are used.
See also [`simulate`](@ref) and [`impulse!`](@ref).
"""
function simulate!(out::AbstractArray, εs::AbstractArray, arma::ARMAProcess,
        y0::Union{Real,AbstractArray,Nothing}=nothing)
    Base.require_one_based_indexing(out, εs)
    nar = arorder(arma)
    nma = maorder(arma)
    noy0 = y0 === nothing || length(y0) == 0
    δ = Int(noy0)
    if noy0
        out[1] = hasintercept(arma) ? εs[1]+arma.intercept : εs[1]
        t0 = 1
    else
        Base.require_one_based_indexing(y0)
        copyto!(out, y0)
        t0 = lastindex(y0)
    end
    for t in t0+1:min(lastindex(out), lastindex(εs))
        arsum = bdot(arma.ars, out, t, min(t-1, nar))
        if noy0 || t > t0+1 # The initial εs are not used if there is y0
            arsum -= bdot(arma.mas, εs, t, min(t-t0+δ-1, nma))
        end
        out[t] = hasintercept(arma) ? εs[t] + arsum + arma.intercept : εs[t] + arsum
    end
    return out
end

"""
    simulate(εs::AbstractArray, arma::ARMAProcess, y0=nothing)

Same as [`simulate!`](@ref), but allocates an array for results.
"""
simulate(εs::AbstractArray, arma::ARMAProcess, y0=nothing) =
    simulate!(similar(εs), εs, arma, y0)

"""
    impulse!(out::AbstractArray, arma::ARMAProcess, ε0::Real=1.0)

Compute the responses of the `arma` process to the impulse
specified with `ε0` and store the results in `out`.
The number of horizons to be computed is determined by the length of `out`.
See also [`impulse`](@ref) and [`simulate!`](@ref).
"""
function impulse!(out::AbstractArray, arma::ARMAProcess, ε0::Real=1.0)
    Base.require_one_based_indexing(out)
    out[1] = ε0
    nar = arorder(arma)
    nma = maorder(arma)
    for t in 2:lastindex(out)
        arsum = bdot(arma.ars, out, t, min(t-1, nar))
        out[t] = t <= nma + 1 ? arsum - arma.mas[t-1] * ε0 : arsum
    end
    return out
end

"""
    impulse(arma::ARMAProcess, nhorz::Integer, ε0::Real=1.0)

$_impulse_common_docstr
"""
impulse(arma::ARMAProcess, nhorz::Integer, ε0::Real=1.0) =
    impulse!(Vector{Float64}(undef, nhorz), arma, ε0)
