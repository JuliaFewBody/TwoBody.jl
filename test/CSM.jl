@testset "CSM.jl" begin
  basisset = GeometricBasisSet(GaussianBasis, 0.1, 20.0, 12)
  free = Hamiltonian(Kinetic(hbar=1, m=1))
  θ = 0.2

  unscaled = solve(free, ComplexScalingMethod(basisset))
  scaled = solve(free, ComplexScalingMethod(basisset; θ=θ))
  @test scaled.E ≈ cis(-2θ) .* unscaled.E rtol=1e-12
  @test scaled.S ≈ unscaled.S

  coulomb = Hamiltonian(Kinetic(hbar=1, m=1), Coulomb(coefficient=-1))
  rayleigh_ritz = solve(coulomb, basisset, info=-1)
  complex_scaling = solve(coulomb, ComplexScalingMethod(basisset))
  @test complex_scaling.E ≈ rayleigh_ritz.E rtol=1e-11

  gaussian = Gaussian(coefficient=-2, exponent=0.7)
  simple = SimpleGaussianBasis(0.4)
  method = ComplexScalingMethod(simple; θ=θ)
  expected = -2 * (pi / (2simple.a + gaussian.exponent * cis(2θ)))^(3/2)
  @test TwoBody.matrix(gaussian, method)[1,1] ≈ expected
  @test TwoBody.ψ(scaled, 1.0) isa Complex
  @test_throws ArgumentError ComplexScalingMethod(basisset; θ=-0.1)
  @test_throws ArgumentError TwoBody.matrix(Tabulated(0.0:0.1:1.0, ones(11)), method)
end
