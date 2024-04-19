@testset "transformation" begin
    Y0 = zeros(10, 2)
    Y = randn(10, 2)
    tr = NoTrend()
    @test nskip(tr) == 0
    detrend!(tr, Y0, Y)
    @test Y0 == Y
    fill!(Y0, 0)
    invdetrend!(tr, Y, Y0)
    @test all(==(0), Y)

    Y0 = zeros(9, 2)
    Y = randn(10, 2)
    tr = FirstDiff()
    @test nskip(tr) == 1
    detrend!(tr, Y0, Y)
    @test Y0 ≈ diff(Y, dims=1)
    Y1 = zeros(9, 2)
    invdetrend!(tr, Y1, Y0)
    @test Y1 ≈ cumsum(Y0, dims=1)

    Y0 = zeros(8, 2)
    Y = randn(10, 2)
    tr = FirstDiff(2)
    @test nskip(tr) == 2
    detrend!(tr, Y0, Y)
    @test Y0 ≈ view(Y,3:10,:) .- view(Y,1:8,:)
    Y1 = zeros(8, 2)
    invdetrend!(tr, Y1, Y0)
    @test Y1[1:2,:] ≈ Y0[1:2,:]
    @test Y1[3:end,:] - Y1[1:end-2,:] ≈ Y0[3:8,:]
end

@testset "DynamicFactor" begin
    data = matread(datafile("lpw_data.mat"))["dataout"]
    tb = Tables.table(data)
    ibal = all(!isnan, data[3:end,:], dims=1)[:]
    ibal = (1:length(ibal))[ibal]
    xbal = data[2:end,1:8]
    f1 = DynamicFactor(xbal, nothing, FirstDiff(), 3, nothing, 4, 4, 1:223)
    @test size(f1.facX) == (222, 5)
    # Compare results with Matlab code from Li, Plagborg-Møller and Wolf (2024)
    # Make modifications in data/src/lpw_savedata.m to get Matlab results
    @test abs.(f1.facX[1,1:3]) ≈
        abs.([0.669750255362180, -1.415042540425748, -0.279535312367630]) atol=1e-10
    @test abs.(f1.facX[10,1:3]) ≈
        abs.([-2.267063090431706, -1.434815860575840, 0.454588283183868]) atol=1e-10
    @test abs.(f1.facX[20,1:3]) ≈
        abs.([7.323931107653098, -4.727910038603926, 1.825862900774614]) atol=1e-10
    @test abs.(f1.facX[221,1:3]) ≈
        abs.([-0.300027230626971, 0.168284777077292, 1.054070197848707]) atol=1e-10
    @test abs.(f1.Λ'[:,1:3]) ≈
        abs.([0.816933524667466   0.877661701684616   0.673038907345591;
        0.211586118251876  -0.797016010137497  -0.195121578082652;
        0.294391325215926   0.084195578852317   0.212593894215132;
        1.388030195340448   1.495730233977966   0.015497277005997;
        1.075675875094696   0.991704176968184   0.151939966515270;
        1.371864827828503  -0.513779766786925  -0.154190698776157;
        0.000282808827547   0.000684102622842   0.000234552180429;
       -0.128279701643663  -0.168769489168454   1.763765804567798]) atol=1e-10
    @test f1.u[1:8,end] ≈
        [-0.975415737426374, -1.412518101895273, -1.994916116989259, -1.048416744367444,
         -1.093584435914863, -0.217932197392372, -0.927498314077866, -1.269794438833060
        ] atol=1e-10
    @test f1.arcoef' ≈
        [0.736964838253279   0.216046718510706   0.031400816622726  -0.055512886991835;
         0.754170105533859   0.128560903554997   0.094147365408704  -0.071278429940820;
         0.914728014043238  -0.139294004432957   0.104859589673747   0.048585759489537;
         0.623955768161770   0.120327956139262   0.033709249938343  -0.059589528011958;
         1.075841476634743  -0.006232388714530  -0.054906986294867  -0.110350444768725;
         1.264562793369892  -0.138543195394575  -0.175856436931966  -0.016666483863437;
         0.502103927024850   0.100605789022884   0.060796568132528  -0.058813198639289;
         0.998861622995029  -0.043270565241107   0.067670952611482  -0.109591180672500
        ] atol=1e-10
    @test f1.σ ≈ [2.551851358100894, 0.740766400455260, 0.566134049898459,
        1.206309600482594, 1.031873591284054, 3.113219352996401, 0.003014397428756,
        0.528899210498389] atol=1e-10

    f11 = deepcopy(f1)
    fill!(f11.f.fac, 0)
    fill!(f11.Λ, 0)
    fill!(f11.arcoef, 0)
    fit!(f11)
    @test f11.f.fac ≈ f1.f.fac
    @test f11.Λ ≈ f1.Λ
    @test f11.arcoef ≈ f1.arcoef

    @test sprint(show, f1) ==
        "222×3 DynamicFactor{Float64, FirstDiff, Nothing, Factor{Float64, SDDcache{Float64}}, Nothing}"
    if VERSION >= v"1.7"
    @test sprint(show, MIME("text/plain"), f1, context=:displaysize=>(10,120)) == """
        222×3 DynamicFactor{Float64, FirstDiff, Nothing, Factor{Float64, SDDcache{Float64}}, Nothing} with 3 unobserved factors and 0 observed factor:
          -0.66975  1.41504  0.279535
           ⋮                 
          Idiosyncratic AR coefficients for 4 lags:
          ⋮      ⋱  
         Evolution of factors:
          Not estimated"""
    end

    f2 = fit(DynamicFactor, tb, ibal, nothing, FirstDiff(), BaiNg(30),
        VAROLS, 4, 4, 1:224; subset=(1:224).>1)
    @test abs.(f2.facX[1,1:3]) ≈
        abs.([2.633579135057327, -4.515438104711660, 0.967695227734307]) atol=1e-10
    @test abs.(f2.facX[20,1:3]) ≈
        abs.([-11.213179331809721, -10.755437451370934, -9.740548002249279]) atol=1e-10
    # Li, Plagborg-Møller and Wolf (2024) has an initial row of zeros
    # for lagmatrix generated from cumsum_nan
    # Removing the initial row yields the same estimates here
    @test abs.(coef(f2.facproc)) ≈
        abs.([0.673856090754789  -0.081037456424403  -0.490680042824428;
         1.616377319397290   0.296011178122282  -0.131965176312504;
        -0.244663060538245   1.369015986230120  -0.098455471141018;
         0.283991836982538  -0.195631889289281   0.932725936459795;
        -0.732635002010108  -0.277190250747602   0.179818440183486;
         0.364379235288969  -0.499477715914352  -0.079759252139077;
        -0.055844609138274   0.032057475641071  -0.021067035288564;
         0.259088832362754   0.121301115717096   0.056402897635998;
        -0.072960531361710   0.273201943623642   0.210890285510650;
         0.112900500553000  -0.017828891944840   0.191598679035541;
        -0.215963192051662  -0.118458036521081  -0.102091302758074;
         0.026271389380724  -0.163505737633263  -0.041529088862164;
        -0.179952139267754   0.145787466313025  -0.151287681694109]) atol=1e-10
    @test sprint(show, f2) ==
        "222×3 DynamicFactor{Float64, FirstDiff, Nothing, Factor{Float64, SDDcache{Float64}}, VAROLS{Float64, Vector{Float64}, Nothing, Nothing, Nothing}}"
    if VERSION >= v"1.7"
    @test sprint(show, MIME("text/plain"), f2, context=:displaysize=>(10,120)) == """
        222×3 DynamicFactor{Float64, FirstDiff, Nothing, Factor{Float64, SDDcache{Float64}}, VAROLS{Float64, Vector{Float64}, Nothing, Nothing, Nothing}} with 3 unobserved factors and 0 observed factor:
          2.63358  4.51544  -0.967695
          ⋮                 
          Idiosyncratic AR coefficients for 4 lags:
          ⋮      ⋱  
         Evolution of factors:
          218×13 OLS regression for VAR with 3 variables and 4 lags"""
    end
end
