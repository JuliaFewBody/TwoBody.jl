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

  @test_throws ArgumentError ECGBasis([1.0 2.0; 0.0 1.0])
  @test_throws DimensionMismatch ECGBasis(A; prefactors=([1.0, 0.0, 0.0],))
end
