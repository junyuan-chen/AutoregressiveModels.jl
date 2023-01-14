@testset "bootstrap VAR" begin
    df = exampledata(:gk)
    ns = (:logcpi, :logip, :ff, :ebp, :ff4_tc)
    r = fit(VARProcess, df, ns, 12)
    stat1 = zeros(1, 2)
    _length(x) = x.out[1] = length(x.data)
    alldata = boot!(stat1=>_length, r, keepbootdata=true)
    @test stat1 == [1350 1350]
    @test length(alldata) == 2
    @test all(x->size(x)==(270,5), alldata)
    stat2 = zeros(1, 2)
    boot!((stat1=>_length, stat2=>_length), r, correctbias=false)
    @test stat1 == [1350 1350]
    @test stat2 == [1350 1350]
    stat2 = zeros(1, 3)
    @test_throws ArgumentError boot!((stat1=>_length, stat2=>_length), r)

    ε05 = zeros(5)
    ε05[5] = 1
    fillirf(x, ε0) = impulse!(reshape(x.out, 5, 10), VARProcess(x.r), ε0)
    ndraw = 1000
    f5(x) = fillirf(x, ε05)
    irf5 = Array{Float64}(undef, 50, ndraw)
    boot!(irf5=>f5, r)
    mirf5 = sum(irf5, dims=2)./ndraw
    @test mirf5[5] == 1
    @test mirf5[9] ≈ 0.39 atol=5e-2
    boot!(irf5=>f5, r, ntasks=1)
    mirf5 = sum(irf5, dims=2)./ndraw
    @test mirf5[5] == 1
    @test mirf5[9] ≈ 0.39 atol=5e-2
end
