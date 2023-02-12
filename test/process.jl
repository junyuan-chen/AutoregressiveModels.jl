@testset "VARProcess" begin
    B = [1.003532758367871 -0.314370089912221 1.662878301208329 -0.008227769152681 -0.200502990878739 -0.320479368332797;
        -0.076183071753115 0.682326947353711 -0.129459582384621 0.077372099814356 0.199130255799557 -0.082127775860029;
        0.005778927375100 -0.008683217084575  0.290444453858049 -0.005184505682256 -0.004531906575996 -0.172695800820669]
    B0 = [2.252034900907038, -0.517314466513827, -0.275340954423760]
    var1 = VARProcess(B, B0)
    @test nvar(var1) == 3
    @test arorder(var1) == 2
    @test hasintercept(var1)
    C1 = companionform(var1)
    @test C1[1:3,:] == B
    @test C1[4:6,:] == diagm(3, 6, 0=>ones(3))
    @test isstable(var1)
    @test !isstable(var1, 0.5)

    εs = fill(0.1, 3)
    var1(εs)
    @test εs == B0 .+ 0.1

    εs = fill(0.1, 3, 2)
    simulate!(εs, var1, ones(3, 2))
    @test all(isone, εs)
    εs = fill(0.1, 3, 4)
    simulate!(εs, var1, ones(3, 2))
    @test εs[:,3] ≈ sum(B,dims=2) .+ B0 .+ 0.1
    @test εs[:,4] ≈ B * view(view(εs,:,3:-1:2),:) .+ B0 .+ 0.1

    εs1 = fill(0.1, 3, 2)
    simulate!(εs1, var1, ones(3, 2), nlag=1)
    @test εs1 == ones(3, 2)
    εs2 = fill(0.1, 3, 1)
    simulate!(εs2, var1, ones(3))
    @test εs1[:,2:2] == εs2
    εs1 = fill(0.1, 3, 4)
    simulate!(εs1, var1, zeros(3, 2))
    εs2 = fill(0.1, 3, 2)
    @test simulate!(εs2, var1) ≈ εs1[:,3:4]
    εs1 = fill(0.1, 3, 4, 2)
    simulate!(εs1, var1, zeros(3, 2, 2))
    @test εs1[:,:,1] ≈ εs1[:,:,2]
    εs2 = fill(0.1, 3, 2, 2)
    @test simulate!(εs2, var1) ≈ εs1[:,3:4,:]
    εs1 = fill(0.1, 3, 4, 2)
    @test_throws DimensionMismatch simulate!(εs1, var1, zeros(3, 2, 3))

    irf = zeros(3, 5)
    impulse!(irf, var1, [1,0,0])
    @test irf[:,1] == [1,0,0]
    @test irf[:,2] ≈ B[:,1]
    irf2 = zeros(3, 5)
    impulse!(irf2, var1, 1)
    @test irf2 ≈ irf
    impulse!(irf2, var1, 1:1)
    @test irf2 ≈ irf
    @test_throws DimensionMismatch impulse!(irf2, var1, 1:2)
    irf2 = zeros(3, 5, 1)
    impulse!(irf2, var1, 1:1)
    @test reshape(irf2, 3, 5) ≈ irf

    @test_throws DimensionMismatch impulse!(irf, var1, I(3))
    irf = zeros(3, 5, 3)
    impulse!(irf, var1, I(3))
    @test irf[:,1,:] == I(3)
    # Compare with Matlab results
    @test irf[:,5,:] ≈ [1.085951501118779 -1.477729061474453 1.834001023603404;
        -0.047546376340552 0.605943409707762 -0.234294401065066;
        0.001378593197746 -0.012981998312923 -0.001548705260207] atol = 1e-10
    irf2 = zeros(3, 5, 3)
    impulse!(irf2, var1, 1:3)
    @test irf2 ≈ irf
    irf2 = impulse(var1, I(3), 5)
    @test irf2 ≈ irf
    irf3 = zeros(3, 5, 2)
    impulse!(irf3, var1, view(I(3),:,3:-1:2))
    @test irf3[:,5,:] ≈ irf[:,5,3:-1:2]
    irf4 = zeros(3, 5, 2)
    impulse!(irf4, var1, 3:-1:2)
    @test irf4 ≈ irf3

    irf = zeros(3, 5)
    impulse!(irf, var1, [0,1,0], nlag=1)
    @test irf[:,1] == [0,1,0]
    @test irf[:,5] ≈ [-0.856166672115337, 0.313653860320074, -0.010442931823260] atol = 1e-10
    irf2 = impulse(var1, [0,1,0], 5, nlag=1)
    @test irf2 ≈ irf
    irf3 = impulse(var1, 2, 5, nlag=1)
    @test irf3 ≈ irf2

    irf = impulse(var1, view(I(3),:,3:-1:2), 5, nlag=1)
    irf2 = zeros(3, 5, 2)
    impulse!(irf2, var1, view(I(3),:,3:-1:2), nlag=1)
    @test irf ≈ irf2
    irf3 = impulse(var1, 3:-1:2, 5, nlag=1)
    @test irf3 ≈ irf

    B1 = [1 0; 0 0.1]
    var2 = VARProcess(B1)
    @test nvar(var2) == 2
    @test arorder(var2) == 1
    @test !hasintercept(var2)
    C2 = companionform(var2)
    @test C2[1:2,:] == B1
    @test !isstable(var2)
    @test isstable(var2, -0.01)

    εs = fill(0.1, 3)
    var2(εs)
    @test all(εs .== 0.1)
end
