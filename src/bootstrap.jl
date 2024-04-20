randomindex(r::VectorAutoregression) = sample(1:size(residuals(r),1))

function wilddraw!(out, r::VectorAutoregression)
    randn!(view(out, :, 1))
    K = size(out, 2)
    if K > 1
        for j in 2:K
            copyto!(view(out,:,j), view(out,:,1))
        end
    end
    out .*= residuals(r)
    return out
end

function iidresiddraw!(out, r::VectorAutoregression)
    T = size(out,1)
    for i in 1:T
        idraw = sample(1:T)
        copyto!(view(out, i, :), view(residuals(r), idraw, :))
    end
    return out
end

function _bootstrap!(ks, stats, r::VectorAutoregression, var, initialindex, drawresid,
        ::Val{estimatevar}, allbootdata) where estimatevar
    keepbootdata = allbootdata isa Vector
    T, N = size(residuals(r))
    nlag = arorder(r)
    NP = N * nlag
    X0 = modelmatrix(r)
    bootdata = Matrix{Float64}(undef, T+nlag, N)
    if estimatevar
        m = deepcopy(r.est)
        rk = VectorAutoregression(m, r.names, r.lookup, !hasintercept(r))
    else
        rk = nothing
    end
    X1s = _reshape(view(X0, :, hasintercept(r) ? (2:NP+1) : 1:NP), T, N, nlag)
    Y0s = view(X1s, :, :, nlag:-1:1)
    for k in ks
        i0 = initialindex(r)
        Y0 = view(Y0s, i0, :, :)
        drawresid(view(bootdata, nlag+1:nlag+T, :), r)
        copyto!(view(bootdata, 1:nlag, :), Y0')
        for t in nlag+1:nlag+T
            var(view(bootdata, t, :), _reshape(view(bootdata, t-1:-1:t-nlag, :)', N*nlag))
        end
        keepbootdata && (allbootdata[k] = copy(bootdata))
        estimatevar && fit!(m, bootdata; Yinresid=true)
        for stat in stats
            nt = (out=selectdim(stat[1], ndims(stat[1]), k), data=bootdata, r=rk)
            stat[2](nt)
        end
    end
end

"""
    bootstrap!(stats, r::VectorAutoregression; kwargs...)

Conduct autoregressive bootstrap for statistics specified with `stats`
while assuming the data generating process can be approximated by
an estimated vector autoregressive model `r`.

Each statistic to be computed needs to be paired with an array for storing the output.
The number of bootstrap iterations is determined by the last dimension of the array;
while the other dimensions should match the size of the statistic.
For example, if the statistic is a scalar, the output array should be a vector;
it should be a matrix if the statistic is a vector.
The statistic is specified as a function accepting a `NamedTuple` as the single argument,
which contains a field `out` that points to a view of the array
indexed by the last dimension,
a field `data` that provides the current sample of bootstrap data,
and a field `r` for the results of VAR estimation based on `data`
if the keyword `estimatevar` is `true` (`nothing` otherwise).
The provided function is called in each bootstrap iteration
with such a `NamedTuple` containing the updated values
and the computed statistics are expected to be saved in-place to `out`.
For the sake of performance, unnecessary allocations should be avoided
when defining such a function.
The argument `stats` can be either a `Pair` of
the pre-allocated output array and the statistic function
or an iterable of `Pairs` if more than one statistic is needed
for the same bootstrap procedure.

# Keywords
- `initialindex::Union{Function,Integer}=randomindex`: the row index of a data columns from `r` for selecting the initial lags to be used for the bootstrap data.
- `drawresid::Function=wilddraw!`: a function that specifies how residuals are generated for the bootstrap data given the initial lags.
- `correctbias=false`: conduct bias correction with [`biascorrect`](@ref).
- `estimatevar::Bool=true`: re-estimate VAR model specified in `r` with each sample of the bootstrap data.
- `ntasks::Integer=Threads.nthreads()`: number of tasks spawned to conduct bootstrap iterations in concurrent chunks with multiple threads.
- `keepbootdata::Bool=false`: collect and return each sample of bootstrap data generated (with possibly very large memory usage).
"""
function bootstrap!(stats, r::VectorAutoregression;
        initialindex::Union{Function,Integer}=randomindex,
        drawresid::Function=wilddraw!,
        correctbias=false, estimatevar::Bool=true,
        ntasks::Integer=Threads.nthreads(), keepbootdata::Bool=false)
    stats isa Pair && (stats = (stats,))
    out1 = stats[1][1]
    nsample = size(out1, ndims(out1))
    nstat = length(stats)
    if nstat > 1
        for k in 1:nstat
            outk = stats[k][1]
            size(outk, ndims(outk)) == nsample || throw(ArgumentError(
                "all arrays for statistics must have the same length in the last dimesion"))
        end
    end
    if initialindex isa Integer
        initindex = r -> initialindex
    else
        initindex = initialindex
    end
    if correctbias != false
        if coefcorrected(r) === nothing
            if correctbias == true
                r, δ = biascorrect(r)
            else
                r, δ = biascorrect(r; correctbias...)
            end
        end
        # biascorrect is not applied if VAR is not stable
        if δ === nothing
            var = VARProcess(r)
        elseif hasintercept(r)
            # Slightly faster simulate! with copied var
            var = VARProcess(coefcorrected(r)', view(coef(r),1,:), copy=true)
        else
            var = VARProcess(coefcorrected(r)', copy=true)
        end
    else
        var = VARProcess(r)
    end
    allbootdata = keepbootdata ? Vector{Matrix}(undef, nsample) : nothing
    ntasks = min(ntasks, nsample)
    if ntasks > 1
        npertask = nsample ÷ ntasks
        # Assign sample IDs across tasks
        ks = Vector{UnitRange{Int64}}(undef, ntasks)
        k0 = npertask + nsample%ntasks
        ks[1] = 1:k0
        for i in 2:ntasks
            ks[i] = k0+1:k0+npertask
            k0 += npertask
        end
        @sync for itask in 1:ntasks
            Threads.@spawn begin
                _bootstrap!(ks[itask], stats, r, var, initindex, drawresid, Val(estimatevar),
                    allbootdata)
            end
        end
    else
        _bootstrap!(1:nsample, stats, r, var, initindex, drawresid, Val(estimatevar), allbootdata)
    end
    return keepbootdata ? allbootdata : nothing
end
