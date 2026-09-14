using Test
using TwoBody
import Random

@testset "StochasticVariationalMethod" begin
  method = SVM(max_basis=12)
  @test method isa StochasticVariationalMethod
  @test method.max_basis == 12
  @test method.candidates == 25
  @test method.amin == 1e-4
  @test method.amax == 1e+4
  @test method.sweeps == 0
  @test method.abstol == 0.0
  @test method.patience == 3
  @test method.overlap_tol == 1e-10
  @test sprint(show, method) == string(method)

  @test_throws ArgumentError SVM(max_basis=0)
  @test_throws ArgumentError SVM(max_basis=4, candidates=0)
  @test_throws ArgumentError SVM(max_basis=4, amin=0)
  @test_throws ArgumentError SVM(max_basis=4, amin=1.0, amax=1.0)
  @test_throws ArgumentError SVM(max_basis=4, sweeps=-1)
  @test_throws ArgumentError SVM(max_basis=4, abstol=-1)
  @test_throws ArgumentError SVM(max_basis=4, patience=0)
  @test_throws ArgumentError SVM(max_basis=4, overlap_tol=0)

  sampler = SVM(max_basis=4, amin=1e-2, amax=1e+2)
  exponents = [
    TwoBody._svm_exponent(Random.MersenneTwister(5), sampler) for _ in 1:8
  ]
  @test all(exponent -> sampler.amin <= exponent <= sampler.amax, exponents)
  @test all(exponent -> exponent == exponents[1], exponents)
  rng = Random.MersenneTwister(5)
  @test TwoBody._svm_exponent(rng, sampler) != TwoBody._svm_exponent(rng, sampler)
end

@testset "SVM growth" begin
  hydrogen = Hamiltonian(Kinetic(hbar=1, m=1), Coulomb(coefficient=-1))
  method = SVM(max_basis=8, candidates=8, amin=1e-2, amax=1e+2)

  result = solve(
    hydrogen, SimpleGaussianBasis(1.0), method;
    rng=Random.MersenneTwister(11), info=1,
  )
  repeated = solve(
    hydrogen, SimpleGaussianBasis(1.0), method;
    rng=Random.MersenneTwister(11), info=1,
  )
  @test result.E == repeated.E
  @test [basis.a for basis in result.basisset.basis] ==
        [basis.a for basis in repeated.basisset.basis]

  default_first = solve(hydrogen, SimpleGaussianBasis(1.0), method; info=1)
  default_second = solve(hydrogen, SimpleGaussianBasis(1.0), method; info=1)
  @test default_first.E == default_second.E

  @test length(result.basisset) == method.max_basis
  @test length(result.history) == method.max_basis
  @test all(step -> step.kind === :growth, result.history)
  @test all(step -> step.accepted, result.history)
  @test result.n_evaluations == method.max_basis * method.candidates
  @test result.n_evaluations ==
        sum(length(step.evaluations) for step in result.history)
  @test result.n_rejected ==
        count(record -> !record.valid, Iterators.flatten(
          (step.evaluations for step in result.history),
        ))
  @test result.history[1].improvement === nothing
  @test all(step -> step.improvement >= -1e-12, result.history[2:end])
  @test all(diff([step.energy for step in result.history]) .<= 1e-12)
  for step in result.history
    valid_records = filter(record -> record.valid, step.evaluations)
    @test step.energy == minimum(record.energy for record in valid_records)
    @test step.exponent ==
          valid_records[argmin(map(record -> record.energy, valid_records))].exponent
    @test result.basisset[step.index].a == step.exponent
  end

  direct = solve(hydrogen, result.basisset, info=-1)
  @test result.E[1] ≈ direct.E[1] atol=1e-13
  @test result.E[1] < -0.49
  @test result.expectation[:S][1] ≈ 1.0 atol=1e-12
  @test isfinite(TwoBody.ψ(result, 0.5))
  @test result.method === method
  @test length(result.initialbasisset) == 0
end

@testset "SVM refinement" begin
  hydrogen = Hamiltonian(Kinetic(hbar=1, m=1), Coulomb(coefficient=-1))
  grown = SVM(max_basis=5, candidates=8, amin=1e-2, amax=1e+2)
  refined = SVM(max_basis=5, candidates=8, amin=1e-2, amax=1e+2, sweeps=2)

  without = solve(
    hydrogen, SimpleGaussianBasis(1.0), grown; rng=Random.MersenneTwister(3), info=1,
  )
  with = solve(
    hydrogen, SimpleGaussianBasis(1.0), refined; rng=Random.MersenneTwister(3), info=1,
  )

  refinement = filter(step -> step.kind === :refinement, with.history)
  @test length(with.history) == refined.max_basis * (1 + refined.sweeps)
  @test length(refinement) == refined.max_basis * refined.sweeps
  @test [step.sweep for step in refinement] ==
        vcat(fill(1, refined.max_basis), fill(2, refined.max_basis))
  @test [step.index for step in refinement] == repeat(1:refined.max_basis, 2)
  @test all(step -> step.improvement >= 0, refinement)
  @test all(diff([step.energy for step in with.history]) .<= 1e-12)
  @test with.E[1] <= without.E[1]
  for index in 1:refined.max_basis
    accepted = filter(
      step -> step.kind === :refinement && step.index == index && step.accepted,
      with.history,
    )
    isempty(accepted) && continue
    @test with.basisset[index].a == last(accepted).exponent
  end
end

@testset "SVM warm start" begin
  hydrogen = Hamiltonian(Kinetic(hbar=1, m=1), Coulomb(coefficient=-1))
  initial = BasisSet(SimpleGaussianBasis(0.2), SimpleGaussianBasis(2.0))
  method = SVM(max_basis=4, candidates=6, amin=1e-2, amax=1e+2)

  result = solve(
    hydrogen, initial, method; rng=Random.MersenneTwister(7), info=1,
  )
  @test length(result.basisset) == 4
  @test length(result.history) == 2
  @test [basis.a for basis in result.basisset.basis][1:2] == [0.2, 2.0]
  @test result.history[1].improvement ≈
        solve(hydrogen, initial, info=-1).E[1] - result.history[1].energy
  @test result.E[1] <= solve(hydrogen, initial, info=-1).E[1]
  @test result.initialbasisset === initial

  # the prototype defaults to the last function of the initial basis set
  angular = BasisSet(GaussianBasis(a=0.5, l=1, m=0), GaussianBasis(a=2.0, l=1, m=0))
  angular_result = solve(
    hydrogen,
    angular,
    SVM(max_basis=3, candidates=4, amin=1e-2, amax=1e+2);
    rng=Random.MersenneTwister(2),
    info=1,
  )
  @test all(basis -> basis.l == 1 && basis.m == 0, angular_result.basisset.basis)
  @test angular_result.E[1] < solve(hydrogen, angular, info=-1).E[1]
  @test angular_result.E[1] >= -0.125  # the 2p energy is the variational bound

  # any basis with an exponent parameter can be used
  slater = solve(
    hydrogen,
    PowerSlaterBasis(0, 1.0),
    SVM(max_basis=2, candidates=8, amin=1e-1, amax=1e+1);
    rng=Random.MersenneTwister(4),
    info=1,
  )
  @test all(basis -> basis isa PowerSlaterBasis, slater.basisset.basis)
  @test slater.E[1] < -0.49
end

@testset "SVM stopping and errors" begin
  hydrogen = Hamiltonian(Kinetic(hbar=1, m=1), Coulomb(coefficient=-1))

  patient = solve(
    hydrogen,
    SimpleGaussianBasis(1.0),
    SVM(max_basis=8, candidates=4, amin=1e-2, amax=1e+2, abstol=1.0, patience=1);
    rng=Random.MersenneTwister(1),
    info=1,
  )
  @test length(patient.history) == 2
  @test length(patient.basisset) == 2

  @test_throws ArgumentError solve(
    hydrogen,
    BasisSet(SimpleGaussianBasis(0.2), SimpleGaussianBasis(2.0)),
    SVM(max_basis=1),
  )
  @test_throws ArgumentError solve(hydrogen, BasisSet(), SVM(max_basis=2))

  degenerate = try
    solve(
      hydrogen,
      BasisSet(SimpleGaussianBasis(1.0), SimpleGaussianBasis(1.0)),
      SVM(max_basis=3, candidates=2),
    )
    nothing
  catch error
    error
  end
  @test degenerate isa ErrorException
  @test occursin("SVM cannot start from the given BasisSet", sprint(showerror, degenerate))

  exhausted = try
    solve(
      hydrogen,
      SimpleGaussianBasis(1.0),
      SVM(max_basis=3, candidates=4, amin=1.0, amax=1.0 + 1e-9);
      rng=Random.MersenneTwister(1),
    )
    nothing
  catch error
    error
  end
  @test exhausted isa ErrorException
  diagnostic = sprint(showerror, exhausted)
  @test occursin("4 rejected of 4 step evaluations", diagnostic)
  @test occursin("overlap_tol=1.0e-10", diagnostic)
end
