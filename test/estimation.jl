@testset "OLS VAR" begin
    df = exampledata(:gk)
    ns = (:logcpi, :logip, :ff, :ebp, :ff4_tc)
    r = fit(VARProcess, df, ns, 12)
    ols = r.est
    @test size(modelmatrix(ols)) == (258, 61)
    @test size(coef(ols)) == (61, 5)
    @test size(residuals(ols)) == (258, 5)
    @test dof_residual(ols) == 197
    @test coefcorrected(ols) === nothing
    @test size(residvcov(ols)) == (5, 5)
    @test sprint(show, ols) == "OLS regression (258, 5)"

    @test coef(r) == coef(ols)
    @test coef(r, 2, 2) == coef(ols)[3,2]
    @test coef(r, :ff, :logcpi, 3) == coef(ols)[12,3]
    @test coef(r, 2, :constant) == coef(ols)[1,2]
    @test residvcov(r) == residvcov(ols)
    @test residvcov(r, 1) == residvcov(ols)[1]
    @test residvcov(r, :logip, 3) == residvcov(ols)[2,3]

    var = VARProcess(r)
    # Compare results with Matlab (QR vs Cholesky)
    @test var.B[:,1:6] ≈ [1.400747125045873 0.031042491155815 0.168907819652938 -0.202252774142558 -0.595974154264276 -0.516057707685298;
        0.118208051049184 0.899415175581331 -0.010727535303623 -0.306437461449999 1.321493801405006 -0.341819209078565;
        -0.094756205453230 0.066681937843200 1.358126743062167 -0.085723012758911 -0.269970488835939 0.132546391866932;
        0.156916459934180 -0.060094199554040 0.030250242691529 0.592483007475474 0.394494467440027 -0.247149015642490;
        -0.012737867887599 0.001180703950030 0.037482883610356 0.000190645817543 0.196292432107634 0.017359562324033] atol = 1e-7
    @test var.B0 ≈ [0.193904475973759, 2.689522849660043, 1.104783075149450,
        -2.945365821418703, -0.105443126437674] atol = 1e-7
    irf = zeros(5, 10, 5)
    impulse!(irf, r, I(5))
    @test irf[1,10,:] ≈ [0.963812354485012, 0.142430057040239, 0.158895931720420,
        -0.160259976752492, -2.400207853745114] atol = 1e-7
    @test impulse(r, I(5), 9) ≈ irf

    rc, δ = biascorrect(r, factor=0.1)
    @test δ == 0.1
    coefc = coefcorrected(rc)
    # Compare results with Matlab
    @test coefc[1:6,1] ≈ [1.402783743801523, 0.031005981131695, 0.168886290491873,
        -0.202210673789537, -0.595697165375395, -0.516601400819960] atol=1e-7
    @test coefc[1:6,4] ≈ [0.157100526002647, -0.060149336363554, 0.030205995300717,
        0.594232338137092, 0.393669749431599, -0.247703449017035] atol=1e-7
    rc, δ = biascorrect(r)
    @test δ ≈ 0.09791089490627655 atol = 1e-7

    r1 = fit(VARProcess, df, ns, 12, nocons=true)
    @test !hasintercept(r1)
    var1 = VARProcess(r1)
    @test !hasintercept(var1)
end
