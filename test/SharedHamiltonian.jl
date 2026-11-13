import FewBodyHamiltonians as FBH

@testset "shared Hamiltonian construction" begin
  @testset "potential-only conversion" begin
    shared = FBH.Hamiltonian(FBH.Coulomb(1, 2, -1))
    native = Hamiltonian(shared)
    @test TwoBody.V(native[1], 2.0) == -0.5
  end

  @testset "relative kinetic energy and units" begin
    basis = BasisSet(GaussianBasis(0.5), GaussianBasis(1.2))
    for units in (:atomic, :nuclear)
      shared = FBH.Kinetic([2.0, 3.0]; units) + FBH.Coulomb(1, 2; coefficient=-0.7)
      native = Hamiltonian(Kinetic(hbar=1, m=1.2), Coulomb(coefficient=-0.7))
      @test TwoBody.matrix(shared, basis) ≈ TwoBody.matrix(native, basis)
      @test solve(shared, basis; info=0).E ≈ solve(native, basis; info=0).E
      @test TwoBody.element(shared, basis[1], basis[2]) ≈ TwoBody.element(native, basis[1], basis[2])
    end
    positronium = FBH.Kinetic([1.0, 1.0]) + FBH.Coulomb([1, -1])
    hydrogen_basis = GeometricBasisSet(GaussianBasis, 0.1, 80.0, 20)
    @test solve(positronium, hydrogen_basis; info=0).E[1] ≈ -0.25 rtol=1e-5
    fdm = FiniteDifferenceMethod(Δr=0.2, rₘₐₓ=4.0)
    @test TwoBody.matrix(positronium, fdm) ≈ TwoBody.matrix(Hamiltonian(Kinetic(m=0.5), Coulomb(-1)), fdm)
  end

  @testset "central potentials and perturbations" begin
    basis = BasisSet(GaussianBasis(0.5), GaussianBasis(1.2))
    shared = FBH.Kinetic([2.0, 2.0]) + FBH.Oscillator(1, 2; coefficient=0.5)
    native = Hamiltonian(Kinetic(), PowerLaw(coefficient=0.5, exponent=2))
    ψ(x) = exp(-sum(abs2, x) / 2)
    @test local_energy(shared, ψ, [0.2, -0.3, 0.4]) ≈ 1.5 atol=1e-12
    @test solve(shared, ψ, VariationalMonteCarlo(n_steps=5, burn_in=0)).E ≈ 1.5 atol=1e-12
    for potential in (
      -2 * FBH.Gaussian(1, 2; exponent=0.3) * FBH.Gaussian(2, 1; exponent=0.4),
      FBH.ManyBodyGaussian([0.7 -0.7; -0.7 0.7]; coefficient=-2),
    )
      @test TwoBody.matrix(FBH.Hamiltonian(potential), basis) ≈ TwoBody.matrix(Hamiltonian(Gaussian(-2, 0.7)), basis)
    end
    custom = 3 * FBH.Custom(r -> exp(-0.7r^2), 1, 2; rtol=1e-6, maxevals=100)
    @test TwoBody.matrix(FBH.Hamiltonian(custom), basis) ≈ TwoBody.matrix(Hamiltonian(Gaussian(3, 0.7)), basis) rtol=1e-9
    perturbation = FBH.Hamiltonian(FBH.Coulomb(1, 2; coefficient=0.2))
    a = solve(shared, basis; perturbation, info=1)
    b = solve(native, basis; perturbation=Hamiltonian(Coulomb(0.2)), info=1)
    @test a.expectation[:perturbation] ≈ b.expectation[:perturbation]
    @test solve(native, basis; perturbation, info=1).expectation[:perturbation] ≈ b.expectation[:perturbation]
    @test solve(shared, basis; perturbation=shared, info=1).expectation[:perturbation][1] ≈ b.E[1]
    method = BVM(max_basis=2, pool_size=2, tuple_size=2, initial_samples=1, batch_size=1, rounds=1, search_size=1)
    @test solve(shared, basis, method; perturbation, info=1).E ≈ b.E
    @test solve(native, basis, method; perturbation, info=1).expectation[:perturbation] ≈ b.expectation[:perturbation]
    @test optimize(shared, GaussianBasis(0.5); info=-1).E[1] ≈ 1.5 atol=1e-8
    coulomb = FBH.Kinetic([2.0, 2.0]) + FBH.Coulomb([1, -1])
    @test length(FC(coulomb, PowerSlaterBasis(0, 1.5)).basis) == 2
  end

  @testset "unsupported physical problems fail before solving" begin
    @test_throws ArgumentError solve(FBH.Hamiltonian(FBH.Coulomb(1, 2)), GaussianBasis(0.5))
    @test_throws ArgumentError Hamiltonian(FBH.Kinetic([1, 1, 1]) + FBH.Coulomb([1, -1, 0]))
    @test_throws ArgumentError Hamiltonian(FBH.Kinetic([1, 1]) + FBH.Coulomb(1, 3))
    @test_throws ArgumentError Hamiltonian(FBH.Hamiltonian(FBH.Gaussian(1, 3; exponent=1)))
    @test_throws ArgumentError Hamiltonian(FBH.Hamiltonian(FBH.ManyBodyGaussian([1 -1 0; -1 1 0; 0 0 0])))
    @test_throws ArgumentError Hamiltonian(FBH.Kinetic([1, 1]; units=:nuclear) + FBH.Coulomb([1, -1]))
    nuclear = FBH.Kinetic([1, 1]; units=:nuclear) + FBH.Coulomb(1, 2; coefficient=-1.44)
    @test_throws ArgumentError solve(nuclear, GaussianBasis(0.5); perturbation=FBH.Coulomb([1, -1]))
    atomic_perturbation = FBH.Kinetic([2, 2]) + FBH.Coulomb([1, -1])
    @test_throws ArgumentError solve(nuclear, GaussianBasis(0.5); perturbation=atomic_perturbation, info=1)
    for term in (FBH.Tensor(1, 2; exponent=1), FBH.SpinOrbit(1, 2; exponent=1))
      @test_throws ArgumentError Hamiltonian(FBH.Kinetic([1, 1]) + term)
    end
  end

  @testset "optional backend dispatch" begin
    shared = FBH.Kinetic([2.0, 2.0]) + FBH.Oscillator(1, 2; coefficient=0.5)
    native = Hamiltonian(Kinetic(), PowerLaw(0.5, 2))
    qtt = QuanticsTensorTrainMethod(quantics=2, rₘₐₓ=4.0)
    if isnothing(Base.get_extension(TwoBody, :TwoBodyQTTExt))
      @test_throws ArgumentError TwoBody.matrix(shared, qtt)
      @test_throws ArgumentError solve(shared, qtt)
    else
      a, b = TwoBody.matrix(shared, qtt), TwoBody.matrix(native, qtt)
      @test qttvalue(a, 2, 2) ≈ qttvalue(b, 2, 2)
      @test qttvalue(a, 2, 3) ≈ qttvalue(b, 2, 3)
    end
    vnn = VNN(Δr=0.5, rₘₐₓ=3.0, maxiters=0)
    if isnothing(Base.get_extension(TwoBody, :TwoBodyVNNExt))
      @test_throws ArgumentError solve(shared, vnn)
    else
      @test solve(shared, vnn).E ≈ solve(native, vnn).E
    end
    @test isempty(Test.detect_ambiguities(TwoBody; recursive=true))
  end
end
