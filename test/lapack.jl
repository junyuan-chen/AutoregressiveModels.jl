@testset "SVD" begin
    m, n = 20, 10
    for alg in (DivideAndConquer(), QRIteration())
        for TF in (Float64, Float32)
            A = randn(TF, m, n)
            # A is always overwritten
            @test _svd!(copy(A), false, alg) == _svd!(svdcache(alg, A), copy(A))
        end
    end
end
