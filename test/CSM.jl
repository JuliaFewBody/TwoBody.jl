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
  @test eltype(scaled.E) <: Complex
  @test eltype(unscaled.E) <: Complex
  @test haskey(scaled, :C) && !haskey(scaled, :info)
  for n in axes(scaled.C, 2)
    @test transpose(scaled.C[:,n]) * scaled.S * scaled.C[:,n] ≈ 1
  end
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

  # the analytic continuation of each potential agrees with the numerically
  # continued potential V(r e^{iθ}) integrated against the real basis functions
  for potential in (
    Constant(constant=0.3),
    Linear(coefficient=0.2),
    Coulomb(coefficient=-1),
    PowerLaw(coefficient=0.7, exponent=2.5),
    Gaussian(coefficient=-0.3, exponent=0.8),
    Exponential(coefficient=0.2, exponent=0.5),
    Yukawa(coefficient=-0.4, exponent=0.3),
  )
    analytic = TwoBody.matrix(potential, ComplexScalingMethod(gem; θ=θ))
    numerical = [
      first(quadgk(
        r -> r^2 * TwoBody.φ(real_basisset[i], r) * TwoBody.V(potential, r * cis(θ)) * TwoBody.φ(real_basisset[j], r),
        0, Inf; rtol=1e-9,
      ))
      for i in 1:length(real_basisset), j in 1:length(real_basisset)
    ]
    @test analytic ≈ numerical rtol=1e-6
  end

  # complex-range Gaussians keep the imaginary part of a continued Gaussian potential
  complex_range = ComplexGaussianBasisSet(0.1, 20.0, 4)
  continued = Gaussian(coefficient=-0.3, exponent=0.8 * cis(2θ))
  for i in 1:length(complex_range), j in i:length(complex_range)
    b1, b2 = complex_range[i], complex_range[j]
    reference = -0.3 * first(quadgk(
      r -> r^2 * TwoBody.φ(b1, r) * exp(-0.8 * cis(2θ) * r^2) * TwoBody.φ(b2, r), 0, Inf; rtol=1e-9,
    ))
    @test TwoBody.element(continued, b1, b2) ≈ reference rtol=1e-6
  end
  @test any(x -> imag(x) != 0, TwoBody.matrix(Gaussian(coefficient=-0.3, exponent=0.8), ComplexScalingMethod(complex_range; θ=θ)))
  @test TwoBody.element(Gaussian(coefficient=-0.3, exponent=0.8), complex_range[1], complex_range[2]) isa Real
  @test_throws ArgumentError TwoBody.matrix(CSMUnsupportedOperator(), ComplexScalingMethod(gem; θ=θ))

  # display
  @test occursin("θ=$(θ)", sprint(show, ComplexScalingMethod(gem; θ=θ)))
  @test occursin("# eigenvalue", sprint(show, scaled))
  @test occursin("E₁", string(scaled))
end
