@testset "bootstrap VAR" begin
    df = exampledata(:gk)
    ns = (:logcpi, :logip, :ff, :ebp, :ff4_tc)
    r = fit(VARProcess, df, ns, 12)
    stat1 = zeros(1, 2)
    _length(x) = x.out[1] = length(x.data)
    alldata = bootstrap!(stat1=>_length, r, keepbootdata=true)
    @test stat1 == [1350 1350]
    @test length(alldata) == 2
    @test all(x->size(x)==(270,5), alldata)
    stat2 = zeros(1, 2)
    bootstrap!((stat1=>_length, stat2=>_length), r, correctbias=false)
    @test stat1 == [1350 1350]
    @test stat2 == [1350 1350]
    stat2 = zeros(1, 3)
    @test_throws ArgumentError bootstrap!((stat1=>_length, stat2=>_length), r)

    ε05 = zeros(5)
    ε05[5] = 1
    fillirf!(x, ε0) = impulse!(reshape(x.out, 5, 10), x.r, ε0)
    ndraw = 1000
    f5!(x) = fillirf!(x, ε05)
    irf5 = Array{Float64}(undef, 50, ndraw)
    bootstrap!(irf5=>f5!, r)
    mirf5 = sum(irf5, dims=2)./ndraw
    @test mirf5[5] == 1
    @test mirf5[9] ≈ 0.39 atol=5e-2
    bootstrap!(irf5=>f5!, r, ntasks=1)
    mirf5 = sum(irf5, dims=2)./ndraw
    @test mirf5[5] == 1
    @test mirf5[9] ≈ 0.39 atol = 5e-2

    ns = (3, 4, 6, 11)
    r1 = fit(VARProcess, df, ns, 12, adjust_dofr=false)
    Σ = residvcov(r1)
    fillsirf!(x) = impulse!(reshape(x.out, 4, 37), x.r, view(cholesky(residvcov(x.r)).L,:,3))
    ndraw = 10000
    sirf = Array{Float64}(undef, 4*37, ndraw)
    bootstrap!(sirf=>fillsirf!, r1, initialindex=1, drawresid=iidresiddraw!,
        correctbias=false)
    sirf2 = view(sirf, 2:4:2+4*36, :)
    lb, ub, ptlevel = confint(SuptQuantileBootBand(), sirf2, level=0.68)
    @test lb[1] == 0
    @test lb[2] ≈ 0.035 atol = 5e-3
    @test lb[20] ≈ -0.47 atol = 5e-2
    @test ub[2] ≈ 0.132 atol = 5e-2
    @test ub[20] ≈ 0.058 atol = 1e-2
end
