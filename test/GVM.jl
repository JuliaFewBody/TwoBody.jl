using Test
using TwoBody
import Optim
import Random

@testset "GradientVariationalMethod" begin
  method = GVM()
  @test method isa GradientVariationalMethod
  @test method.max_basis === nothing
  @test method.amin == 1e-4
  @test method.amax == 1e+4
  @test method.optimizer isa Optim.LBFGS
  @test method.gradtol == 1e-8
  @test method.maxiter == 1000
  @test method.overlap_tol == 1e-10
  @test sprint(show, method) == string(method)
  @test occursin("optimizer=L-BFGS", string(method))

  @test_throws ArgumentError GVM(max_basis=0)
  @test_throws ArgumentError GVM(amin=0)
  @test_throws ArgumentError GVM(amin=1.0, amax=1.0)
  @test_throws ArgumentError GVM(gradtol=0)
  @test_throws ArgumentError GVM(maxiter=0)
  @test_throws ArgumentError GVM(overlap_tol=0)

  exponents = TwoBody._gvm_initial_exponents(GVM(amin=1e-2, amax=1e+2), 5)
  @test length(exponents) == 5
  @test exponents[1] ≈ 1e-2
  @test exponents[end] ≈ 1e+2
  @test all(diff(log.(exponents)) .≈ log(exponents[2] / exponents[1]))
  @test only(TwoBody._gvm_initial_exponents(GVM(amin=1e-2, amax=1e+2), 1)) ≈ 1.0
end

@testset "GVM gradient" begin
  hydrogen = Hamiltonian(Kinetic(hbar=1, m=1), Coulomb(coefficient=-1))

  function energy(basis, x)
    basisset = BasisSet((
      TwoBody._replace_exponent(basis[index], exp(x[index]))
      for index in eachindex(basis)
    )...)
    return TwoBody._variational_energy(hydrogen, basisset).energy
  end

  for basis in (
    TwoBody.Basis[
      SimpleGaussianBasis(0.3), SimpleGaussianBasis(1.7), SimpleGaussianBasis(9.0),
    ],
    TwoBody.Basis[GaussianBasis(a=0.4, l=1, m=0), GaussianBasis(a=2.1, l=1, m=0)],
    TwoBody.Basis[PowerSlaterBasis(0, 0.8), PowerSlaterBasis(1, 1.9)],
  )
    exponents = [TwoBody._exponent(b) for b in basis]
    x = log.(exponents)
    evaluation = TwoBody._variational_energy(
      hydrogen, BasisSet(basis...),
    )
    gradient = zeros(length(basis))
    TwoBody._gvm_gradient!(
      gradient,
      hydrogen,
      basis,
      exponents,
      evaluation.energy,
      evaluation.coefficients,
    )

    # central differences of the same energy, in the log parameters
    step = 1e-6
    for k in eachindex(x)
      shift = [index == k ? step : 0.0 for index in eachindex(x)]
      reference = (energy(basis, x + shift) - energy(basis, x - shift)) / 2step
      @test gradient[k] ≈ reference rtol=1e-5
    end
  end
end

@testset "GVM warm start" begin
  hydrogen = Hamiltonian(Kinetic(hbar=1, m=1), Coulomb(coefficient=-1))

  # a single power-Slater function is the exact 1s orbital at a = 1
  slater = solve(hydrogen, BasisSet(PowerSlaterBasis(0, 2.0)), GVM(), info=1)
  @test slater.converged
  @test slater.iterations > 0
  @test slater.basisset[1].a ≈ 1.0 atol=1e-6
  @test slater.E[1] ≈ -0.5 atol=1e-12

  initial = BasisSet(
    SimpleGaussianBasis(13.00773),
    SimpleGaussianBasis(1.962079),
    SimpleGaussianBasis(0.444529),
    SimpleGaussianBasis(0.1219492),
  )
  result = solve(hydrogen, initial, GVM(), info=1)
  @test result.E[1] <= solve(hydrogen, initial, info=-1).E[1]
  @test result.E[1] <= optimize(hydrogen, initial, info=-1).E[1] + 1e-9
  @test result.initialbasisset === initial
  @test length(result.basisset) == length(initial)
  @test result.n_evaluations == length(result.history)
  @test result.history[1].energy ≈ solve(hydrogen, initial, info=-1).E[1] atol=1e-13
  @test minimum(entry.energy for entry in result.history) ≈ result.E[1] atol=1e-12
  @test result.method isa GradientVariationalMethod
  @test result.optimizer === result.method.optimizer
  @test result.expectation[:S][1] ≈ 1.0 atol=1e-12
  @test occursin("optimized basis function", string(result))
  @test occursin("optimization log", string(result))
  @test !occursin("optimization log", string(solve(hydrogen, initial, GVM(), info=1, progress=false)))

  # the optimization is deterministic
  @test solve(hydrogen, initial, GVM(), info=1).E == result.E

  # another gradient algorithm can be selected
  conjugate = solve(
    hydrogen, initial, GVM(optimizer=Optim.ConjugateGradient()), info=1,
  )
  @test conjugate.E[1] ≈ result.E[1] atol=1e-8
end

@testset "GVM cold start" begin
  hydrogen = Hamiltonian(Kinetic(hbar=1, m=1), Coulomb(coefficient=-1))

  result = solve(
    hydrogen,
    SimpleGaussianBasis(1.0),
    GVM(max_basis=6, amin=1e-2, amax=1e+2),
    info=1,
  )
  @test length(result.basisset) == 6
  @test all(basis -> basis isa SimpleGaussianBasis, result.basisset.basis)
  @test [basis.a for basis in result.initialbasisset.basis] ≈
        TwoBody._gvm_initial_exponents(result.method, 6)
  @test result.E[1] < -0.4999
  @test result.E[1] ≈ solve(hydrogen, result.basisset, info=-1).E[1] atol=1e-13

  angular = solve(
    hydrogen,
    GaussianBasis(a=1.0, l=1, m=0),
    GVM(max_basis=5, amin=1e-2, amax=1e+2),
    info=1,
  )
  @test all(basis -> basis.l == 1 && basis.m == 0, angular.basisset.basis)
  @test angular.E[1] ≈ -0.125 atol=1e-4
end

@testset "GVM errors" begin
  hydrogen = Hamiltonian(Kinetic(hbar=1, m=1), Coulomb(coefficient=-1))

  @test_throws ArgumentError solve(hydrogen, SimpleGaussianBasis(1.0), GVM())
  @test_throws ArgumentError solve(
    hydrogen, BasisSet(SimpleGaussianBasis(1.0)), GVM(max_basis=3),
  )
  @test_throws ArgumentError solve(hydrogen, BasisSet(), GVM())

  degenerate = try
    solve(
      hydrogen, BasisSet(SimpleGaussianBasis(1.0), SimpleGaussianBasis(1.0)), GVM(),
    )
    nothing
  catch error
    error
  end
  @test degenerate isa ErrorException
  @test occursin("GVM cannot start from the given BasisSet", sprint(showerror, degenerate))
end

@testset "SVM and GVM" begin
  hydrogen = Hamiltonian(Kinetic(hbar=1, m=1), Coulomb(coefficient=-1))

  stochastic = solve(
    hydrogen,
    SimpleGaussianBasis(1.0),
    SVM(max_basis=6, candidates=8, amin=1e-2, amax=1e+2);
    rng=Random.MersenneTwister(5),
    info=1,
  )
  gradient = solve(hydrogen, stochastic.basisset, GVM(), info=1)
  @test gradient.E[1] <= stochastic.E[1]
  @test -0.5 - 1e-9 <= gradient.E[1] < -0.49
end
