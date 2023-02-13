@testset "ARMAProcess" begin
    arma = ARMAProcess(nothing, nothing)
    @test nvar(arma) == 1
    @test arorder(arma) == 0
    @test maorder(arma) == 0
    @test !hasintercept(arma)
    εs = [0.1, -0.2, 0.3, -0.1]
    @test simulate(εs, arma) == εs
    @test impulse(arma, 2) == [1, 0]

    arma = ARMAProcess(0.8, [0.1])
    @test arorder(arma) == 1
    @test maorder(arma) == 1
    @test !hasintercept(arma)
    @test arma.mas == (0.1,)
    @test simulate(εs, arma) ≈ [0.1, -0.13, 0.216, 0.0428]
    @test impulse(arma, 4) ≈ [1, 0.7, 0.56, 0.448]

    arma = ARMAProcess([0.8, 0.1], nothing, 1)
    @test arorder(arma) == 2
    @test maorder(arma) == 0
    @test hasintercept(arma)
    @test simulate(εs, arma) ≈ [1.1, 1.68, 2.754, 3.2712]
    @test simulate(εs, arma, 0.1) ≈ [0.1, 0.88, 2.014, 2.5992]
    @test simulate!(zeros(7), εs, arma, 0.1) ≈ [0.1, 0.88, 2.014, 2.5992, 0, 0, 0]
    @test impulse(arma, 4) ≈ [1, 0.8, 0.74, 0.672]

    mas = (0.1, -0.2)
    arma = ARMAProcess((), mas, 1.0)
    @test arorder(arma) == 0
    @test maorder(arma) == 2
    @test hasintercept(arma)
    @test arma.ars == ()
    @test simulate(εs, arma) ≈ [1.1, 0.79, 1.34, 0.83]
    @test simulate(εs, arma, [0.1, 0.3]) ≈ [0.1, 0.3, 1.3, 0.87]
    simulate!(zeros(5), εs, arma, [0.1, 0.3]) ≈ [0.1, 0.3, 1.3, 0.87, 0]
    @test impulse(arma, 4) ≈ [1, -0.1, 0.2, 0]
end
