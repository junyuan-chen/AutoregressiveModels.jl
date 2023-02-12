function _convert_armacoefs(coefs::AbstractVector, copy::Bool)
    N = length(coefs)
    if N == 0
        return nothing
    elseif N == 1
        return first(coefs)
    elseif copy || Base.has_offset_axes(coefs)
        return collect(coefs)
    else
        return coefs
    end
end

_convert_armacoefs(coefs::Any, ::Bool) = coefs

struct ARMAProcess{TA<:Union{Real, AbstractVector, Nothing},
        TM<:Union{Real, AbstractVector, Nothing}, TF<:Union{Real, Nothing}}
    ars::TA
    mas::TM
    intercept::TF
    function ARMAProcess(ars::Union{Real, AbstractVector, Nothing},
            mas::Union{Real, AbstractVector, Nothing},
            intercept::Union{Real, Nothing}=nothing; copy::Bool=false)
        ars1 = _convert_armacoefs(ars, copy)
        mas1 = _convert_armacoefs(mas, copy)
        return new{typeof(ars1), typeof(mas1), typeof(intercept)}(ars1, mas1, intercept)
    end
end

nvar(arma::ARMAProcess) = 1
arorder(arma::ARMAProcess{<:AbstractVector}) = length(arma.ars)
arorder(arma::ARMAProcess{<:Real}) = 1
arorder(arma::ARMAProcess{Nothing}) = 0
maorder(arma::ARMAProcess{<:Any,<:AbstractVector}) = length(arma.mas)
maorder(arma::ARMAProcess{<:Any,<:Real}) = 1
maorder(arma::ARMAProcess{<:Any,Nothing}) = 0
hasintercept(::ARMAProcess{<:Any,<:Any,<:Any}) = true
hasintercept(::ARMAProcess{<:Any,<:Any,Nothing}) = false

function simulate!(out::AbstractArray, εs::AbstractArray, arma::ARMAProcess,
        y0::Union{Real,AbstractArray,Nothing}=nothing)
    Base.require_one_based_indexing(out, εs)
    nar = arorder(arma)
    nma = maorder(arma)
    noy0 = y0 === nothing || length(y0) == 0
    if noy0
        out[1] = hasintercept(arma) ? εs[1]+arma.intercept : εs[1]
        t0 = 1
    else
        Base.require_one_based_indexing(y0)
        copyto!(out, y0)
        t0 = lastindex(y0)
    end
    for t in t0+1:min(lastindex(out), lastindex(εs))
        if nar == 0
            arsum = 0.0
        elseif nar == 1
            arsum = arma.ars * out[t-1]
        else
            if t <= nar
                arsum = dot(view(arma.ars, 1:t-1), view(out, t-1:-1:1))
            else
                arsum = dot(arma.ars, view(out, t-1:-1:t-nar))
            end
        end
        if noy0 || t > t0+1 # The initial εs are not used if there is y0
            if nma == 1
                arsum -= arma.mas * εs[t-1]
            elseif nma > 1
                δ = noy0 ? 1 : 0
                if t - t0 + δ <= nma
                    arsum -= dot(view(arma.mas, 1:t-t0-(1-δ)), view(εs, t-1:-1:t0-δ+1))
                else
                    arsum -= dot(arma.mas, view(εs, t-1:-1:t-nma))
                end
            end
        end
        out[t] = hasintercept(arma) ? εs[t] + arsum + arma.intercept : εs[t] + arsum
    end
    return out
end

simulate(εs::AbstractArray, arma::ARMAProcess, y0=nothing) =
    simulate!(similar(εs), εs, arma, y0)

function impulse!(out::AbstractArray, arma::ARMAProcess, ε0::Real=1.0)
    Base.require_one_based_indexing(out)
    out[1] = ε0
    nar = arorder(arma)
    nma = maorder(arma)
    @inbounds for t in 2:lastindex(out)
        if nar == 0
            arsum = 0.0
        elseif nar == 1
            arsum = arma.ars * out[t-1]
        else
            if t <= nar
                arsum = dot(view(arma.ars, 1:t-1), view(out, t-1:-1:1))
            else
                arsum = dot(arma.ars, view(out, t-1:-1:t-nar))
            end
        end
        if t <= nma + 1
            if nma == 1
                out[t] = arsum - arma.mas * ε0
            else
                out[t] = arsum - arma.mas[t-1] * ε0
            end
        else
            out[t] = arsum
        end
    end
    return out
end

impulse(arma::ARMAProcess, nhorz::Integer, ε0::Real=1.0) =
    impulse!(Vector{Float64}(undef, nhorz), arma, ε0)
