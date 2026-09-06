using TwoBody
import Random

function truth_table_probe(; seed=1)
  hydrogen = Hamiltonian(Kinetic(hbar=1, m=1), Coulomb(coefficient=-1))
  geometric = GeometricBasisSet(GaussianBasis, 0.15, 8.0, 8)
  candidates = BasisSet(geometric.basis...)
  method = BVM(
    max_basis=3,
    pool_size=8,
    tuple_size=3,
    initial_samples=12,
    batch_size=4,
    rounds=1,
    search_size=100,
  )
  tuples = TwoBody._all_bvm_tuples(collect(1:8), 3)
  table = [
    (
      tuple=tuple,
      energy=TwoBody._evaluate_bvm_tuple(
        hydrogen, candidates, Int[], tuple, method, nothing,
      ).energy,
    )
    for tuple in tuples
  ]
  sort!(table, by=row -> row.energy)

  rng = Random.MersenneTwister(seed)
  teachers = TwoBody._sample_bvm_tuples(rng, collect(1:8), 3, 12, Set{Tuple}())
  energy_by_tuple = Dict(row.tuple => row.energy for row in table)
  model = TwoBody._fit_bvm_gp(teachers, [energy_by_tuple[t] for t in teachers])
  predictions = [
    (
      tuple=row.tuple,
      exact=row.energy,
      predicted=TwoBody._predict_bvm_gp(model, row.tuple).mean,
    )
    for row in table if row.tuple ∉ teachers
  ]
  sort!(predictions, by=row -> row.predicted)
  best_exact = first(table)
  predicted_rank = findfirst(row -> row.tuple == best_exact.tuple, predictions)
  println("truth_best_tuple,truth_best_energy,predicted_rank")
  println("$(best_exact.tuple),$(best_exact.energy),$(something(predicted_rank, 0))")
  return (table=table, predictions=predictions)
end

function random_tuple_search(
  hamiltonian,
  candidates,
  method,
  evaluations_per_step;
  rng,
)
  selected = Int[]
  current_energy = nothing
  evaluations = 0
  rejections = 0

  for budget in evaluations_per_step
    length(selected) + method.tuple_size <= method.max_basis || break
    available = setdiff(collect(eachindex(candidates.basis)), selected)
    length(available) >= method.tuple_size || break
    Random.shuffle!(rng, available)
    pool = sort(available[1:min(method.pool_size, length(available))])
    tuples = TwoBody._sample_bvm_tuples(
      rng, pool, method.tuple_size, budget, Set{Tuple}(),
    )
    records = [
      TwoBody._evaluate_bvm_tuple(
        hamiltonian, candidates, selected, tuple, method, current_energy,
      )
      for tuple in tuples
    ]
    evaluations += length(records)
    rejections += count(record -> !record.valid, records)
    valid = filter(record -> record.valid, records)
    isempty(valid) && error("random baseline found no valid tuple")
    best_index = argmin(map(record -> record.energy, valid))
    best = valid[best_index]
    selected = best.basis_indices
    current_energy = best.energy
  end
  return (
    energy=current_energy,
    basis_size=length(selected),
    evaluations=evaluations,
    rejections=rejections,
  )
end

function hydrogen_benchmark(; seeds=1:20)
  hydrogen = Hamiltonian(Kinetic(hbar=1, m=1), Coulomb(coefficient=-1))
  geometric = GeometricBasisSet(GaussianBasis, 0.05, 80.0, 24)
  candidates = BasisSet(geometric.basis...)
  reference = solve(hydrogen, candidates, info=-1).E[1]
  common = (
    max_basis=12,
    pool_size=24,
    tuple_size=3,
    initial_samples=24,
    batch_size=12,
    rounds=2,
    search_size=1_000,
    abstol=0.0,
    patience=4,
  )

  println("seed,method,energy_gap,basis_size,evaluations,rejections")
  rows = NamedTuple[]
  for seed in seeds
    lcb = solve(
      hydrogen,
      candidates,
      BVM(; common..., beta=2.0);
      rng=Random.MersenneTwister(seed),
      info=0,
    )
    mean_only = solve(
      hydrogen,
      candidates,
      BVM(; common..., beta=0.0);
      rng=Random.MersenneTwister(seed),
      info=0,
    )
    budgets = [length(step.evaluations) for step in lcb.history]
    random = random_tuple_search(
      hydrogen,
      candidates,
      BVM(; common..., beta=2.0),
      budgets;
      rng=Random.MersenneTwister(seed),
    )

    for row in (
      (method="lcb", energy=lcb.E[1], basis_size=length(lcb.selected_indices),
       evaluations=lcb.n_evaluations, rejections=lcb.n_rejected),
      (method="mean", energy=mean_only.E[1], basis_size=length(mean_only.selected_indices),
       evaluations=mean_only.n_evaluations, rejections=mean_only.n_rejected),
      (method="random", energy=random.energy, basis_size=random.basis_size,
       evaluations=random.evaluations, rejections=random.rejections),
    )
      output = (seed=seed, method=row.method, energy_gap=row.energy-reference,
                basis_size=row.basis_size, evaluations=row.evaluations,
                rejections=row.rejections)
      push!(rows, output)
      println(join(values(output), ','))
    end
  end
  return (reference=reference, rows=rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
  truth_table_probe()
  hydrogen_benchmark()
end
