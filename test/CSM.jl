# An operator type without an analytic continuation rule.
struct CSMUnsupportedOperator <: TwoBody.Operator end

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

  # the bound state is stable under the rotation while the continuum rotates
  # into the lower half-plane
  rotated = solve(coulomb, ComplexScalingMethod(basisset; θ=θ))
  @test abs(rotated.E[1] - rayleigh_ritz.E[1]) < 1e-2
  @test abs(imag(rotated.E[1])) < 1e-2
  @test minimum(imag, rotated.E) < -0.1

  gaussian = Gaussian(coefficient=-2, exponent=0.7)
  simple = SimpleGaussianBasis(0.4)
  method = ComplexScalingMethod(simple; θ=θ)
  expected = -2 * (pi / (2simple.a + gaussian.exponent * cis(2θ)))^(3/2)
  @test TwoBody.matrix(gaussian, method)[1,1] ≈ expected
  @test TwoBody.ψ(scaled, 1.0) isa Complex
  @test_throws ArgumentError ComplexScalingMethod(basisset; θ=-0.1)
  @test_throws ArgumentError TwoBody.matrix(Tabulated(0.0:0.1:1.0, ones(11)), method)

  # every supported operator is continued analytically; θ = 0 recovers the
  # real Rayleigh-Ritz matrix and the scaled matrix stays complex symmetric
  gem = GeometricBasisSet(GaussianBasis, 0.1, 20.0, 12)
  real_basisset = BasisSet(gem.basis...)
  for operator in (
    RestEnergy(c=1, m=1),
    Laplacian(coefficient=-0.5),
    Kinetic(hbar=1, m=1),
    RelativisticCorrection(c=10.0, m=1, n=2),
    Constant(constant=0.3),
    Linear(coefficient=0.2),
    Coulomb(coefficient=-1),
    PowerLaw(coefficient=0.7, exponent=2.5),
    Gaussian(coefficient=-0.3, exponent=0.8),
    Exponential(coefficient=0.2, exponent=0.5),
    Yukawa(coefficient=-0.4, exponent=0.3),
    Delta(coefficient=0.1),
    Custom(r -> -exp(-r)),
  )
    unrotated = TwoBody.matrix(operator, ComplexScalingMethod(gem))
    @test unrotated ≈ TwoBody.matrix(operator, real_basisset)
    rotated = TwoBody.matrix(operator, ComplexScalingMethod(gem; θ=θ))
    @test all(isfinite, rotated)
    @test rotated == transpose(rotated)
  end
  @test_throws ArgumentError TwoBody.matrix(RelativisticKinetic(c=1, m=1), ComplexScalingMethod(gem; θ=θ))
  @test_throws ArgumentError TwoBody.matrix(CSMUnsupportedOperator(), ComplexScalingMethod(gem; θ=θ))

  # display
  @test occursin("θ=$(θ)", sprint(show, ComplexScalingMethod(gem; θ=θ)))
  @test occursin("# eigenvalue", sprint(show, scaled))
  @test occursin("E₁", string(scaled))
end
