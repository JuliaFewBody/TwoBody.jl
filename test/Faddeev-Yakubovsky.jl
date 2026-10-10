import LinearAlgebra

@testset "Faddeev-Yakubovsky.jl" begin
  H = Hamiltonian(Kinetic(1, 1), Coulomb(-1))
  BS = BasisSet(
    SimpleGaussianBasis(13.00773),
    SimpleGaussianBasis(1.962079),
    SimpleGaussianBasis(0.444529),
    SimpleGaussianBasis(0.1219492),
  )
  method = FaddeevYakubovsky(BS)
  res = solve(H, method; perturbation=Hamiltonian(Constant(0.2)), info=1)
  rr = solve(H, BS)

  @test res.E[1] ≈ rr.E[1] atol=1e-9
  @test res.E[1] ≈ -0.499278 atol=1e-6
  @test res.λ ≈ 1 atol=1e-9
  @test 0 < res.iterations <= method.maxiters
  @test res.nₘₐₓ == 1
  @test res.C' * res.S * res.C ≈ ones(1, 1) atol=1e-12
  @test LinearAlgebra.norm(res.H * res.C - res.S * res.C * res.E[1]) < 1e-8
  # Check the original resolvent equation independently of the symmetric kernel.
  @test (res.E[1] * res.S - res.T) \ (res.V * res.C) ≈ res.C atol=1e-8
  @test res.expectation[:S][1] ≈ 1
  @test res.expectation[:H][1] ≈ res.E[1] atol=1e-9
  @test res.expectation[:perturbation][1] ≈ 0.2
  @test 4π * quadgk(r -> r^2 * abs2(TwoBody.ψ(res, r)), 0, Inf)[1] ≈ 1 atol=1e-9
  for r in (0.0, 0.5, 1.0, 2.0)
    @test abs(TwoBody.ψ(res, r)) ≈ abs(TwoBody.ψ(rr, r)) atol=1e-8
  end

  @testset "basis and interaction choices" begin
    geometric = GeometricBasisSet(SimpleGaussianBasis, 0.1, 80.0, 20)
    hydrogen = solve(H, FaddeevYakubovsky(geometric); info=0)
    @test hydrogen.E[1] ≈ -0.5 atol=2e-5
    @test hydrogen.E[1] ≈ solve(H, geometric).E[1] atol=1e-9

    # Short-range attraction with a repulsive core; both terms form one pair.
    potential = Hamiltonian(Kinetic(), Gaussian(-8, 1), Gaussian(10, 4))
    bs = GeometricBasisSet(SimpleGaussianBasis, 0.2, 8.0, 12)
    fy = solve(potential, FaddeevYakubovsky(bs; Eₘᵢₙ=-10); info=0)
    @test fy.E[1] ≈ solve(potential, bs).E[1] atol=1e-8

    # Rest energies shift the free spectrum and the search interval together.
    shifted = Hamiltonian(H.terms..., RestEnergy(c=1, m=2))
    shift = solve(shifted, FaddeevYakubovsky(BS; Eₘᵢₙ=1, Eₘₐₓ=2); info=0)
    @test shift.E[1] ≈ res.E[1] + 2 atol=1e-9

    single = PowerSlaterBasis(n=0, a=1.0)
    exact = solve(H, FaddeevYakubovsky(single); info=0)
    @test exact.E[1] ≈ -0.5 atol=1e-10
    for (Eₘᵢₙ, Eₘₐₓ) in ((-0.5, 0.0), (-1.0, -0.5))
      endpoint = solve(H, FaddeevYakubovsky(single; Eₘᵢₙ, Eₘₐₓ); info=0)
      @test endpoint.E[1] ≈ -0.5 atol=1e-10
      @test endpoint.iterations == 0
    end
    contracted = ContractedBasis((1.0, 0.5), (BS[1], BS[2]))
    bs = BasisSet(contracted, BS[3], BS[4])
    @test solve(H, FaddeevYakubovsky(bs); info=0).E[1] ≈ solve(H, bs).E[1] atol=1e-9
  end

  @testset "diagnostics" begin
    quiet = solve(H, method; info=0)
    minimal = solve(H, method; info=-1)
    @test isempty(quiet.expectation)
    @test minimal.E == quiet.E == res.E
    @test !haskey(minimal, :C)
    @test occursin("FaddeevYakubovsky", sprint(show, quiet))
    @test occursin("E:", sprint(show, minimal))
  end

  @testset "invalid input and convergence" begin
    @test_throws ArgumentError FaddeevYakubovsky(BasisSet())
    @test_throws ArgumentError FaddeevYakubovsky(BS; Eₘᵢₙ=0, Eₘₐₓ=-1)
    @test_throws ArgumentError FaddeevYakubovsky(BS; Eₘᵢₙ=-Inf)
    @test_throws ArgumentError FaddeevYakubovsky(BS; Eₘₐₓ=NaN)
    @test_throws ArgumentError FaddeevYakubovsky(BS; tolerance=0)
    @test_throws ArgumentError FaddeevYakubovsky(BS; tolerance=Inf)
    @test_throws ArgumentError FaddeevYakubovsky(BS; maxiters=0)
    @test_throws ArgumentError solve(Hamiltonian(Coulomb(-1)), method; info=0)
    @test_throws ArgumentError solve(Hamiltonian(Kinetic()), method; info=0)
    @test_throws ArgumentError solve(Hamiltonian(Kinetic(), Coulomb(1)), method; info=0)
    @test_throws ArgumentError solve(H, FaddeevYakubovsky(BS; Eₘᵢₙ=-0.1); info=0)
    @test_throws ArgumentError solve(H, FaddeevYakubovsky(BS; Eₘₐₓ=-0.9); info=0)
    @test_throws ArgumentError solve(H, FaddeevYakubovsky(BS; Eₘₐₓ=100); info=0)
    @test_throws ArgumentError solve(H, FaddeevYakubovsky(BasisSet(BS[1], BS[1])); info=0)
    @test_throws ErrorException solve(H, FaddeevYakubovsky(BS; maxiters=1); info=0)
  end
end
