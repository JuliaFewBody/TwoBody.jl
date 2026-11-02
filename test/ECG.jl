@testset "ECG.jl" begin
  exponents = 10.0 .^ range(-3, 2, length=10)
  kinetic = ECGKinetic([0.5;;])
  coulomb = ECGCoulomb(-1.0, [1.0])
  ecg_hamiltonian = Hamiltonian(kinetic, coulomb)
  gem_hamiltonian = Hamiltonian(Kinetic(hbar=1, m=1), Coulomb(coefficient=-1))
  prefactors = (
    (),
    ([0.0, 0.0, 1.0],),
    ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
  )

  for rank in 0:2
    ecg = BasisSet((ECGBasis([exponent;;]; prefactors=prefactors[rank + 1]) for exponent in exponents)...)
    gem = BasisSet((GaussianBasis(a=exponent, l=rank, m=0) for exponent in exponents)...)
    ecg_result = solve(ecg_hamiltonian, ecg, info=-1)
    gem_result = solve(gem_hamiltonian, gem, info=-1)
    @test ecg_result.E ≈ gem_result.E rtol=2e-9
    @test ecg_result.E[1] ≈ -1 / (2(rank + 1)^2) atol=2e-4
  end

  A = [1.0 0.2; 0.2 0.8]
  basis = ECGBasis(A)
  coordinates = [1.0 0.0 0.0; 0.0 1.0 0.0]
  @test TwoBody.φ(basis, coordinates) ≈ exp(-1.8)
  @test TwoBody.element(basis, basis) ≈ (pi^2 / TwoBody.LinearAlgebra.det(2A))^(3/2)

  other = ECGBasis([0.7 -0.1; -0.1 1.2];
    prefactors=([1.0, 0.0, 0.0, 0.0, 1.0, 0.0],))
  vector = ([1.0, 0.0, 0.0, 0.0, 1.0, 0.0],)
  basis = ECGBasis(A; prefactors=vector)
  kinetic = ECGKinetic([0.5 0.1; 0.1 0.7])
  coulomb = ECGCoulomb(-1.0, [1.0, -1.0])
  @test TwoBody.element(basis, other) ≈ TwoBody.element(other, basis)
  @test TwoBody.element(kinetic, basis, other) ≈ TwoBody.element(kinetic, other, basis)
  @test TwoBody.element(coulomb, basis, other) ≈ TwoBody.element(coulomb, other, basis)

  # two coordinates with a diagonal correlation matrix separate into products
  # of one-coordinate elements, which checks the n = 2 kinetic and Coulomb paths
  # against the one-coordinate values validated above
  α, β = 0.7, 1.3
  single(a; prefactors=()) = ECGBasis([a;;]; prefactors=prefactors)
  pair(; prefactors=()) = ECGBasis([α 0.0; 0.0 β]; prefactors=prefactors)
  S₁ = TwoBody.element(single(α), single(α))
  S₂ = TwoBody.element(single(β), single(β))
  T₁ = TwoBody.element(ECGKinetic([0.5;;]), single(α), single(α))
  T₂ = TwoBody.element(ECGKinetic([0.5;;]), single(β), single(β))
  V₁ = TwoBody.element(ECGCoulomb(-1.0, [1.0]), single(α), single(α))
  @test TwoBody.element(pair(), pair()) ≈ S₁ * S₂
  @test TwoBody.element(ECGKinetic([0.5 0.0; 0.0 0.5]), pair(), pair()) ≈ T₁ * S₂ + S₁ * T₂
  @test TwoBody.element(ECGCoulomb(-1.0, [1.0, 0.0]), pair(), pair()) ≈ V₁ * S₂
  @test TwoBody.element(ECGCoulomb(-1.0, [1.0, 0.0]), pair(), pair()) ≈ -(2π / 2α) * (π / 2β)^(3/2)
  z₁ = [0.0, 0.0, 1.0]
  z₁₂ = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
  S₁ᵖ = TwoBody.element(single(α; prefactors=(z₁,)), single(α; prefactors=(z₁,)))
  T₁ᵖ = TwoBody.element(ECGKinetic([0.5;;]), single(α; prefactors=(z₁,)), single(α; prefactors=(z₁,)))
  V₁ᵖ = TwoBody.element(ECGCoulomb(-1.0, [1.0]), single(α; prefactors=(z₁,)), single(α; prefactors=(z₁,)))
  @test TwoBody.element(pair(; prefactors=(z₁₂,)), pair(; prefactors=(z₁₂,))) ≈ S₁ᵖ * S₂
  @test TwoBody.element(ECGKinetic([0.5 0.0; 0.0 0.5]), pair(; prefactors=(z₁₂,)), pair(; prefactors=(z₁₂,))) ≈ T₁ᵖ * S₂ + S₁ᵖ * T₂
  @test TwoBody.element(ECGCoulomb(-1.0, [1.0, 0.0]), pair(; prefactors=(z₁₂,)), pair(; prefactors=(z₁₂,))) ≈ V₁ᵖ * S₂ rtol=1e-8

  # parallel prefactors pair with each other: ⟨z⁴⟩ = 3σ⁴ with σ² = 1/(4a)
  zz = ECGBasis([0.9;;]; prefactors=([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]))
  @test TwoBody.element(zz, zz) ≈ 3 / (16 * 0.9^2) * (π / 1.8)^(3/2)

  # the display shows the prefactors
  @test occursin("prefactors=", string(zz))

  @test_throws ArgumentError ECGBasis([1.0 2.0; 0.0 1.0])
  @test_throws DimensionMismatch ECGBasis(A; prefactors=([1.0, 0.0, 0.0],))
end
