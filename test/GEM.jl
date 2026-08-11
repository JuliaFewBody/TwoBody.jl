@testset "GEM.jl" begin
  @testset "normalized Gaussian basis" begin
    for l in 0:3
      basis = GaussianBasis(0.7, l, 0)
      position_norm = quadgk(r -> r^2 * abs2(TwoBody.φ(basis, r)), 0, Inf, rtol=1e-11)[1]
      momentum_norm = quadgk(p -> p^2 * abs2(φp(basis, p)), 0, Inf, rtol=1e-11)[1]
      @test position_norm ≈ 1 atol=1e-10
      @test momentum_norm ≈ 1 atol=1e-10
    end

    basis = GaussianBasis(0.7, 0, 0)
    @test TwoBody.φ(basis, 0.4, 0.3, 0.2) ≈ TwoBody.φ(basis, 0.4) / sqrt(4π)
    @test TwoBody._replace_exponent(GaussianBasis(0.7, 2, -1), 1.2) == GaussianBasis(1.2, 2, -1)
    @test_throws ArgumentError TwoBody.φ(GaussianBasis(-1.0), 0.0)
    @test_throws ArgumentError TwoBody.φ(GaussianBasis(1.0, 1, 2), 0.0)
  end

  @testset "matrix elements" begin
    b1 = GaussianBasis(0.6, 1, 0)
    b2 = GaussianBasis(1.1, 1, 0)
    overlap = quadgk(r -> r^2 * TwoBody.φ(b1, r) * TwoBody.φ(b2, r), 0, Inf, rtol=1e-11)[1]
    @test TwoBody.element(b1, b2) ≈ overlap rtol=1e-10
    @test TwoBody.element(b1, GaussianBasis(1.1, 0, 0)) == 0

    for operator in (
      Constant(constant=0.3),
      Linear(coefficient=0.2),
      Coulomb(coefficient=-0.4),
      PowerLaw(coefficient=0.7, exponent=2.5),
      Gaussian(coefficient=-0.3, exponent=0.8),
      Exponential(coefficient=0.2, exponent=0.5),
      Yukawa(coefficient=-0.4, exponent=0.3),
    )
      numerical = quadgk(
        r -> r^2 * TwoBody.φ(b1, r) * TwoBody.V(operator, r) * TwoBody.φ(b2, r),
        0,
        Inf,
        rtol=1e-10,
      )[1]
      @test TwoBody.element(operator, b1, b2) ≈ numerical rtol=2e-7 atol=1e-10
    end

    kinetic = Kinetic(hbar=1.3, m=0.8)
    radial_laplacian(r) = ForwardDiff.derivative(
      x -> x^2 * ForwardDiff.derivative(y -> TwoBody.φ(b2, y), x),
      r,
    ) / r^2 - b2.l * (b2.l + 1) * TwoBody.φ(b2, r) / r^2
    numerical_kinetic = quadgk(
      r -> r^2 * TwoBody.φ(b1, r) * (-kinetic.hbar^2 / (2 * kinetic.m)) * radial_laplacian(r),
      0,
      Inf,
      rtol=1e-9,
    )[1]
    @test TwoBody.element(kinetic, b1, b2) ≈ numerical_kinetic rtol=2e-8

    correction = RelativisticCorrection(c=2.0, m=1.4, n=2)
    numerical_correction = quadgk(
      p -> p^2 * conj(φp(b1, p)) * (-p^4 / (8 * correction.m^3 * correction.c^2)) * φp(b2, p),
      0,
      Inf,
      rtol=1e-10,
    )[1]
    @test TwoBody.element(correction, b1, b2) ≈ real(numerical_correction) rtol=2e-9

    relativistic = RelativisticKinetic(c=1.0, m=1.4)
    numerical_relativistic = quadgk(
      p -> p^2 * conj(φp(b1, p)) * (sqrt(p^2 + relativistic.m^2) - relativistic.m) * φp(b2, p),
      0,
      Inf,
      rtol=1e-10,
    )[1]
    @test TwoBody.element(relativistic, b1, b2) ≈ real(numerical_relativistic) rtol=2e-7
  end

  @testset "Rayleigh–Ritz integration" begin
    hydrogen = Hamiltonian(Kinetic(hbar=1, m=1), Coulomb(coefficient=-1))
    basisset = BasisSet(
      GaussianBasis(13.00773),
      GaussianBasis(1.962079),
      GaussianBasis(0.444529),
      GaussianBasis(0.1219492),
    )
    result = solve(hydrogen, basisset)
    @test result.E[1] ≈ -0.499278 rtol=1e-5
    @test result.C[:, 1]' * result.S * result.C[:, 1] ≈ 1 atol=1e-10
    @test isfinite(ψp(result, 0.5))

    geometric_basis = GeometricBasisSet(GaussianBasis, 0.1, 10.0, 5)
    @test all(b -> b isa GaussianBasis && b.l == 0 && b.m == 0, geometric_basis.basis)
  end

  @testset "Hiyama complex-range hydrogen benchmark" begin
    hydrogen = Hamiltonian(Kinetic(hbar=1, m=1), Coulomb(coefficient=-1))
    basisset = ComplexGaussianBasisSet(0.015, 2000.0, 80; ω=1.5)
    @test length(basisset) == 160

    for basis in basisset.basis[1:2]
      norm = quadgk(r -> r^2 * abs2(TwoBody.φ(basis, r)), 0, Inf, rtol=1e-10)[1]
      @test norm ≈ 1 atol=2e-9
    end

    result = solve(hydrogen, basisset)
    levels = [1, 3, 10, 26, 30, 36, 40]
    hiyama_table_a5 = [
      -4.999999845e-1,
      -5.555555494e-2,
      -4.999999983e-3,
      -7.396449686e-4,
      -5.555555323e-4,
      -3.856834714e-4,
      -3.106429115e-4,
    ]
    @test maximum(abs.(result.E[levels] .- hiyama_table_a5)) ≤ 5e-11
  end

  @testset "Arifi et al. (2024) Hamiltonian" begin
    ν = 0.2443
    masses = (1.6324, 1.6324)
    a = -0.4235
    b = 0.1655
    αs = 0.4410
    Λ = 0.9639
    spin = -3/4
    reduced_mass = inv(inv(masses[1]) + inv(masses[2]))
    λ = Λ * sqrt(reduced_mass)
    hyperfine = 32π * αs * (λ / sqrt(π))^3 / (9 * masses[1] * masses[2]) * spin

    hamiltonian = Hamiltonian(
      RestEnergy(c=1, m=masses[1]),
      RelativisticKinetic(c=1, m=masses[1]),
      RestEnergy(c=1, m=masses[2]),
      RelativisticKinetic(c=1, m=masses[2]),
      Constant(constant=a),
      Linear(coefficient=b),
      Coulomb(coefficient=-4αs/3),
      Gaussian(coefficient=hyperfine, exponent=λ^2),
    )
    result = solve(hamiltonian, GaussianBasis(ν))
    # Regression value from the attached original Arifi2024 implementation.
    @test result.E[1] ≈ 3.013183414 atol=5e-10
  end
end
