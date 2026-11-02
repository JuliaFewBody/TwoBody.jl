using Test
using TwoBody

@testset "SolverMethod" begin
  @test BVM(max_basis=4) isa TwoBody.SolverMethod
  @test SVM(max_basis=4) isa TwoBody.SolverMethod
  @test GVM(max_basis=4) isa TwoBody.SolverMethod

  method = SVM(max_basis=4, candidates=3)
  @test sprint(show, method) == string(method)
  @test startswith(string(method), "StochasticVariationalMethod(max_basis=4, candidates=3, ")
  @test occursin("overlap_tol=$(method.overlap_tol))", string(method))
  @test startswith(string(BVM(max_basis=4)), "BayesianVariationalMethod(max_basis=4, ")
end

@testset "variational energy" begin
  hydrogen = Hamiltonian(Kinetic(hbar=1, m=1), Coulomb(coefficient=-1))
  basisset = BasisSet(
    SimpleGaussianBasis(0.2),
    SimpleGaussianBasis(0.9),
    SimpleGaussianBasis(4.0),
  )

  evaluation = TwoBody._variational_energy(hydrogen, basisset)
  reference = solve(hydrogen, basisset, info=-1)
  @test evaluation.valid
  @test evaluation.reason === nothing
  @test evaluation.message == ""
  @test evaluation.energy ≈ reference.E[1] atol=1e-13

  overlap = TwoBody.matrix(basisset)
  hamiltonian_matrix = TwoBody.matrix(hydrogen, basisset)
  @test evaluation.coefficients' * overlap * evaluation.coefficients ≈ 1 atol=1e-12
  @test evaluation.coefficients' * hamiltonian_matrix * evaluation.coefficients ≈
        evaluation.energy atol=1e-12

  duplicate = TwoBody._variational_energy(
    hydrogen, BasisSet(SimpleGaussianBasis(1.0), SimpleGaussianBasis(1.0)),
  )
  @test !duplicate.valid
  @test duplicate.energy == Inf
  @test duplicate.coefficients == Float64[]
  @test duplicate.reason == :ill_conditioned_overlap
  @test occursin("is below 1.0e-10", duplicate.message)

  @test TwoBody._variational_energy(
    hydrogen, BasisSet(SimpleGaussianBasis(NaN), SimpleGaussianBasis(1.0)),
  ).reason == :nonfinite_matrix

  failure = TwoBody._variational_energy(
    hydrogen, BasisSet(SimpleGaussianBasis(-1.0)),
  )
  @test failure.reason == :numerical_failure
  @test occursin("DomainError", failure.message)

  increase = TwoBody._variational_energy(hydrogen, basisset; reference=-1.0)
  @test !increase.valid
  @test increase.reason == :energy_increase
  @test TwoBody._variational_energy(
    hydrogen, basisset; reference=evaluation.energy,
  ).valid
end
