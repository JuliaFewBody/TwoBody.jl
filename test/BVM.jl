using Test
using TwoBody
import Random

@testset "BayesianVariationalMethod" begin
  method = BVM(max_basis=12)
  @test method isa BayesianVariationalMethod
  @test method.max_basis == 12
  @test method.pool_size == 100
  @test method.tuple_size == 3
  @test method.initial_samples == 100
  @test method.batch_size == 50
  @test method.rounds == 4
  @test method.search_size == 10_000
  @test method.beta == 2.0
  @test method.abstol == 1e-8
  @test method.patience == 3
  @test method.overlap_tol == 1e-10
  @test sprint(show, method) == string(method)

  @test_throws ArgumentError BVM(max_basis=0)
  @test_throws ArgumentError BVM(max_basis=3, tuple_size=1)
  @test_throws ArgumentError BVM(max_basis=3, tuple_size=4, pool_size=3)
  @test_throws ArgumentError BVM(max_basis=2, tuple_size=3)
  @test_throws ArgumentError BVM(max_basis=3, beta=-1)
  @test_throws ArgumentError BVM(max_basis=3, abstol=-1)
  @test_throws ArgumentError BVM(max_basis=3, overlap_tol=0)

  pool = collect(1:6)
  all_tuples = TwoBody._all_bvm_tuples(pool, 3)
  @test length(all_tuples) == 20
  @test length(unique(all_tuples)) == 20
  @test all(t -> length(t) == 3 && issorted(collect(t)), all_tuples)

  repeated_pool = [1, 1, 2]
  @test_throws ArgumentError TwoBody._all_bvm_tuples(repeated_pool, 2)
  @test_throws ArgumentError TwoBody._sample_bvm_tuples(
    Random.MersenneTwister(11), repeated_pool, 2, 3, Set{Tuple}(),
  )

  excluded = Set{Tuple}((all_tuples[1], all_tuples[2]))
  first_draw = TwoBody._sample_bvm_tuples(
    Random.MersenneTwister(7), pool, 3, 10, excluded,
  )
  second_draw = TwoBody._sample_bvm_tuples(
    Random.MersenneTwister(7), pool, 3, 10, excluded,
  )
  @test first_draw == second_draw
  @test length(first_draw) == 10
  @test length(unique(first_draw)) == 10
  @test all(tuple -> tuple ∉ excluded, first_draw)

  exhausted = TwoBody._sample_bvm_tuples(
    Random.MersenneTwister(9), pool, 3, 100, Set{Tuple}(),
  )
  @test Set(exhausted) == Set(all_tuples)
end

@testset "BVM candidate evaluation" begin
  hydrogen = Hamiltonian(Kinetic(hbar=1, m=1), Coulomb(coefficient=-1))
  candidates = BasisSet(
    GaussianBasis(0.2),
    GaussianBasis(0.7),
    GaussianBasis(2.0),
  )
  method = BVM(max_basis=2, pool_size=3, tuple_size=2)

  evaluation = TwoBody._evaluate_bvm_tuple(
    hydrogen, candidates, Int[], (1, 2), method, nothing,
  )
  direct = solve(hydrogen, BasisSet(candidates[1], candidates[2]), info=-1).E[1]
  @test evaluation.valid
  @test evaluation.reason === nothing
  @test evaluation.energy ≈ direct atol=1e-13
  @test evaluation.basis_indices == [1, 2]

  duplicate_candidates = BasisSet(
    GaussianBasis(1.0),
    GaussianBasis(1.0),
  )
  duplicate_method = BVM(max_basis=2, pool_size=2, tuple_size=2)
  duplicate = TwoBody._evaluate_bvm_tuple(
    hydrogen, duplicate_candidates, Int[], (1, 2), duplicate_method, nothing,
  )
  @test !duplicate.valid
  @test duplicate.reason == :ill_conditioned_overlap

  nonfinite_candidates = BasisSet(
    SimpleGaussianBasis(NaN),
    SimpleGaussianBasis(1.0),
  )
  nonfinite = TwoBody._evaluate_bvm_tuple(
    hydrogen, nonfinite_candidates, Int[], (1, 2), duplicate_method, nothing,
  )
  @test !nonfinite.valid
  @test nonfinite.reason == :nonfinite_matrix

  increase = TwoBody._evaluate_bvm_tuple(
    hydrogen, candidates, Int[], (1, 2), method, -1.0,
  )
  @test !increase.valid
  @test increase.reason == :energy_increase
end

@testset "BVM Gaussian process" begin
  tuples = Tuple[(1, 2, 3), (1, 2, 4), (4, 5, 6)]
  @test TwoBody._bvm_tanimoto(tuples[1], tuples[1]) == 1.0
  @test TwoBody._bvm_tanimoto(tuples[1], tuples[2]) == 0.5
  @test TwoBody._bvm_tanimoto(tuples[1], tuples[3]) == 0.0
  @test TwoBody._bvm_tanimoto(tuples[1], tuples[2]) ==
        TwoBody._bvm_tanimoto(tuples[2], tuples[1])

  covariance = [TwoBody._bvm_tanimoto(a, b) for a in tuples, b in tuples]
  @test TwoBody.LinearAlgebra.eigmin(TwoBody.LinearAlgebra.Symmetric(covariance)) >= -1e-12

  energies = [-1.0, -0.8, -0.4]
  model = TwoBody._fit_bvm_gp(tuples, energies)
  for (tuple, energy) in zip(tuples, energies)
    prediction = TwoBody._predict_bvm_gp(model, tuple)
    @test prediction.mean ≈ energy atol=2e-5
    @test 0 <= prediction.std < 1e-2
  end

  unseen = TwoBody._predict_bvm_gp(model, (1, 5, 6))
  @test isfinite(unseen.mean)
  @test isfinite(unseen.std)
  @test unseen.std >= 0

  flat = TwoBody._fit_bvm_gp(Tuple[(1, 2), (1, 3)], [-0.5, -0.5])
  flat_prediction = TwoBody._predict_bvm_gp(flat, (2, 3))
  @test flat_prediction.mean == -0.5
  @test isfinite(flat_prediction.std)
end

@testset "BVM solve" begin
  hydrogen = Hamiltonian(Kinetic(hbar=1, m=1), Coulomb(coefficient=-1))
  geometric = GeometricBasisSet(GaussianBasis, 0.15, 8.0, 8)
  candidates = BasisSet(geometric.basis...)
  method = BVM(
    max_basis=6,
    pool_size=8,
    tuple_size=2,
    initial_samples=3,
    batch_size=2,
    rounds=1,
    search_size=10,
    beta=2.0,
    abstol=0.0,
    patience=3,
  )

  first_result = solve(
    hydrogen, candidates, method; rng=Random.MersenneTwister(11), info=1,
  )
  second_result = solve(
    hydrogen, candidates, method; rng=Random.MersenneTwister(11), info=1,
  )

  @test first_result.selected_indices == second_result.selected_indices
  @test first_result.E == second_result.E
  @test length(first_result.selected_indices) == 6
  @test length(unique(first_result.selected_indices)) == 6

  default_first = solve(hydrogen, candidates, method; info=1)
  default_second = solve(hydrogen, candidates, method; info=1)
  @test default_first.selected_indices == default_second.selected_indices
  @test default_first.E == default_second.E

  all_evaluations = collect(Iterators.flatten(
    (step.evaluations for step in first_result.history),
  ))
  @test first_result.n_evaluations == length(all_evaluations)
  @test first_result.n_rejected == count(record -> !record.valid, all_evaluations)
  for (step_index, step) in enumerate(first_result.history)
    step_evaluations = collect(Iterators.flatten(
      (prior.evaluations for prior in first_result.history[1:step_index]),
    ))
    @test step.n_evaluations == length(step_evaluations)
    @test step.n_rejected == count(record -> !record.valid, step_evaluations)

    valid_records = filter(record -> record.valid, step.evaluations)
    adopted_records = filter(record -> record.tuple == step.adopted_tuple, valid_records)
    @test length(adopted_records) == 1
    @test step.accepted_energy == adopted_records[1].energy
    @test step.accepted_energy == minimum(record.energy for record in valid_records)
  end

  gp_records = filter(record -> record.source == :gp, all_evaluations)
  @test !isempty(gp_records)
  @test all(record ->
    record.acquisition ≈ record.mean - method.beta * record.std,
    gp_records,
  )

  accepted_energies = [step.accepted_energy for step in first_result.history]
  @test all(diff(accepted_energies) .<= 1e-12)
  direct = solve(hydrogen, first_result.basisset, info=-1)
  full = solve(hydrogen, candidates, info=-1)
  @test first_result.E[1] ≈ direct.E[1] atol=1e-13
  @test full.E[1] <= first_result.E[1] + 1e-12
  @test first_result.expectation[:S][1] ≈ 1.0 atol=1e-12
  @test first_result.expectation[:H][1] ≈ first_result.E[1] atol=1e-12
  @test isfinite(TwoBody.ψ(first_result, 0.5))

  @test_throws ArgumentError solve(
    hydrogen,
    BasisSet(GaussianBasis(0.5)),
    method;
    rng=Random.MersenneTwister(1),
  )

  patience_method = BVM(
    max_basis=8,
    pool_size=8,
    tuple_size=2,
    initial_samples=3,
    batch_size=2,
    rounds=1,
    search_size=10,
    abstol=1.0,
    patience=1,
  )
  patience_result = solve(
    hydrogen, candidates, patience_method; rng=Random.MersenneTwister(12), info=1,
  )
  @test length(patience_result.history) == 2
  @test length(patience_result.selected_indices) == 4
  @test length(patience_result.selected_indices) < patience_method.max_basis

  mixed_candidates = BasisSet(
    GaussianBasis(1.0),
    GaussianBasis(1.0),
    GaussianBasis(0.2),
    GaussianBasis(2.0),
  )
  mixed_method = BVM(
    max_basis=2,
    pool_size=4,
    tuple_size=2,
    initial_samples=6,
    batch_size=1,
    rounds=1,
    search_size=2,
  )
  mixed_result = solve(
    hydrogen, mixed_candidates, mixed_method; rng=Random.MersenneTwister(2), info=1,
  )
  mixed_evaluations = only(mixed_result.history).evaluations
  @test any(record -> !record.valid, mixed_evaluations)
  @test mixed_result.n_evaluations == length(mixed_evaluations)
  @test mixed_result.n_rejected == count(record -> !record.valid, mixed_evaluations)

  duplicates = BasisSet((GaussianBasis(1.0) for _ in 1:4)...)
  invalid_method = BVM(
    max_basis=2,
    pool_size=4,
    tuple_size=2,
    initial_samples=2,
    batch_size=1,
    rounds=1,
    search_size=2,
  )
  all_invalid = try
    solve(hydrogen, duplicates, invalid_method; rng=Random.MersenneTwister(2))
    nothing
  catch error
    error
  end
  @test all_invalid isa ErrorException
  diagnostic = sprint(showerror, all_invalid)
  @test occursin("6 rejected of 6 step evaluations", diagnostic)
  @test occursin("overlap_tol=$(invalid_method.overlap_tol)", diagnostic)
end
