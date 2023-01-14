# An unsafe in-place version of OLS for bootstrap
function _fit!(m::OLS)
    Y = m.resid
    X = m.X
    coef = m.coef
    mul!(coef, X', Y)
    mul!(m.crossXcache, X', X)
    ldiv!(cholesky!(m.crossXcache), coef)
    mul!(Y, X, coef, -1.0, 1.0)
    mul!(m.residvcov, Y', Y)
    rdiv!(m.residvcov, m.dofr)
    return m
end

# An unsafe version for bootstrap
function _fitvar!(data::Matrix, m::OLS, nlag, nocons)
    i0 = nocons ? 0 : 1
    Y = residuals(m)
    X = modelmatrix(m)
    N = size(Y, 2)
    # esample is not used here
    for j in 1:N
        col = view(data, :, j)
        Tfull = length(col)
        copyto!(view(Y, :, j), view(col, nlag+1:Tfull))
        for l in 1:nlag
            copyto!(view(X, :, i0+(l-1)*N+j), view(col, nlag+1-l:Tfull-l))
        end
    end
    _fit!(m)
end

function _default_fillresids!(out, r::VectorAutoregression)
    randn!(view(out, :, 1))
    K = size(out, 2)
    if K > 1
        for j in 2:K
            copyto!(view(out,:,j), view(out,:,1))
        end
    end
    out .*= residuals(r)
end

_default_initialindex(r::VectorAutoregression) = sample(1:size(residuals(r),1))

function _boot!(ks, stats, r::VectorAutoregression, var, initialindex, fillresids!,
        estimatevar, allbootdata)
    keepbootdata = allbootdata isa Vector
    T, N = size(residuals(r))
    nlag = arorder(r)
    NP = N * nlag
    X0 = modelmatrix(r)
    bootdata = Matrix{Float64}(undef, T+nlag, N)
    if estimatevar
        m = deepcopy(r.est)
        rk = VectorAutoregression(m, r.names, r.lookup, r.nocons)
        _fr(x, k) = x[2]((out=view(x[1], :, k), data=bootdata, r=rk))
        _f = _fr
    else
        _fnor(x, k) = x[2]((out=view(x[1], :, k), data=bootdata))
        _f = _fnor
    end
    X1s = reshape(view(X0, :, hasintercept(r) ? (2:NP+1) : 1:NP), T, N, nlag)
    Y0s = view(X1s, :, :, nlag:-1:1)
    for k in ks
        Y0 = view(Y0s, initialindex(r), :, :)
        fillresids!(view(bootdata, nlag+1:nlag+T, :), r)
        εs = bootdata'
        copyto!(view(εs, :, 1:nlag), view(Y0, :, 1:nlag))
        for t in nlag+1:nlag+T
            rlag = t-1:-1:t-nlag
            var(view(εs, :, t), reshape(view(εs, :, rlag), :))
        end
        keepbootdata && (allbootdata[k] = copy(bootdata))
        estimatevar && _fitvar!(bootdata, m, nlag, r.nocons)
        for stat in stats
            _f(stat, k)
        end
    end
end

function boot!(stats, r::VectorAutoregression;
        initialindex::Function=_default_initialindex,
        fillresids!::Function=_default_fillresids!,
        correctbias::Bool=true, estimatevar::Bool=true,
        ntasks::Integer=Threads.nthreads(), keepbootdata::Bool=false)
    stats isa Pair && (stats = (stats,))
    nsample = size(stats[1][1], 2)
    nstat = length(stats)
    if nstat > 1
        for s in 1:nstat
            size(stats[s][1], 2) == nsample || throw(ArgumentError(
                "all matrices for statistics must have the same number of columns"))
        end
    end
    if correctbias
        if coefcorrected(r) === nothing
            r, _ = biascorrect(r)
        end
        # Slightly faster simulate! with copied var
        if hasintercept(r)
            var = VARProcess(coefcorrected(r)', view(coef(r),1,:), copy=true)
        else
            var = VARProcess(coefcorrected(r)', copy=true)
        end
    else
        var = VARProcess(r, copy=true)
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
                _boot!(ks[itask], stats, r, var, initialindex, fillresids!, estimatevar,
                    allbootdata)
            end
        end
    else
        _boot!(1:nsample, stats, r, var, initialindex, fillresids!, estimatevar, allbootdata)
    end
    return keepbootdata ? allbootdata : nothing
end

