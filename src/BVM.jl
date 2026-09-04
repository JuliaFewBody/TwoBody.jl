export BayesianVariationalMethod, BVM

import LinearAlgebra
import Random

"""
    BayesianVariationalMethod(; max_basis, pool_size=100, tuple_size=3,
                              initial_samples=100, batch_size=50, rounds=4,
                              search_size=10_000, beta=2.0, abstol=1e-8,
                              patience=3, overlap_tol=1e-10)

Configure Bayesian selection of fixed-size groups from a finite candidate basis.
`max_basis` limits the accepted basis dimension; `pool_size` is the temporary
candidate pool; `tuple_size` is the number of functions adopted per outer step;
`initial_samples`, `batch_size`, `rounds`, and `search_size` control the
Gaussian-process search budget; and `beta` weighs posterior uncertainty in the
lower-confidence-bound acquisition. `abstol` is an absolute energy improvement
in the Hamiltonian's units, `patience` counts consecutive negligible steps, and
`overlap_tol` rejects nearly linearly dependent subsets.

"""
struct BayesianVariationalMethod{T<:AbstractFloat}
  max_basis::Int
  pool_size::Int
  tuple_size::Int
  initial_samples::Int
  batch_size::Int
  rounds::Int
  search_size::Int
  beta::T
  abstol::T
  patience::Int
  overlap_tol::T
end

function BayesianVariationalMethod(;
  max_basis::Int,
  pool_size::Int=100,
  tuple_size::Int=3,
  initial_samples::Int=100,
  batch_size::Int=50,
  rounds::Int=4,
  search_size::Int=10_000,
  beta::Real=2.0,
  abstol::Real=1e-8,
  patience::Int=3,
  overlap_tol::Real=1e-10,
)
  max_basis > 0 || throw(ArgumentError("max_basis must be positive"))
  pool_size > 0 || throw(ArgumentError("pool_size must be positive"))
  2 <= tuple_size <= pool_size ||
    throw(ArgumentError("tuple_size must satisfy 2 <= tuple_size <= pool_size"))
  max_basis >= tuple_size ||
    throw(ArgumentError("max_basis must be at least tuple_size"))
  initial_samples > 0 || throw(ArgumentError("initial_samples must be positive"))
  batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
  rounds > 0 || throw(ArgumentError("rounds must be positive"))
  search_size > 0 || throw(ArgumentError("search_size must be positive"))
  isfinite(beta) && beta >= 0 ||
    throw(ArgumentError("beta must be nonnegative and finite"))
  isfinite(abstol) && abstol >= 0 ||
    throw(ArgumentError("abstol must be nonnegative and finite"))
  patience > 0 || throw(ArgumentError("patience must be positive"))
  isfinite(overlap_tol) && overlap_tol > 0 ||
    throw(ArgumentError("overlap_tol must be positive and finite"))

  beta_value, abstol_value, overlap_tol_value =
    promote(float(beta), float(abstol), float(overlap_tol))
  return BayesianVariationalMethod{typeof(beta_value)}(
    max_basis,
    pool_size,
    tuple_size,
    initial_samples,
    batch_size,
    rounds,
    search_size,
    beta_value,
    abstol_value,
    patience,
    overlap_tol_value,
  )
end

"""
    BVM(; kwargs...)

Alias for [`BayesianVariationalMethod`](@ref).
"""
const BVM = BayesianVariationalMethod

Base.string(method::BayesianVariationalMethod) =
  "BayesianVariationalMethod(" *
  join(
    ("$(name)=$(getproperty(method, name))" for name in fieldnames(typeof(method))),
    ", ",
  ) *
  ")"
Base.show(io::IO, method::BayesianVariationalMethod) = print(io, string(method))

function _all_bvm_tuples(pool::AbstractVector{<:Integer}, tuple_size::Int)
  length(unique(pool)) == length(pool) ||
    throw(ArgumentError("pool entries must be unique"))
  tuple_size <= length(pool) || return Tuple[]
  tuples = Tuple[]
  chosen = Int[]

  function visit(start::Int, remaining::Int)
    if remaining == 0
      push!(tuples, Tuple(chosen))
      return
    end
    last_start = length(pool) - remaining + 1
    for position in start:last_start
      push!(chosen, Int(pool[position]))
      visit(position + 1, remaining - 1)
      pop!(chosen)
    end
  end

  visit(1, tuple_size)
  return tuples
end

function _sample_bvm_tuples(
  rng::Random.AbstractRNG,
  pool::AbstractVector{<:Integer},
  tuple_size::Int,
  n_tuples::Int,
  excluded::AbstractSet,
)
  length(unique(pool)) == length(pool) ||
    throw(ArgumentError("pool entries must be unique"))
  n_tuples <= 0 && return Tuple[]
  total = binomial(big(length(pool)), tuple_size)
  excluded_here = Base.count(
    tuple -> length(tuple) == tuple_size && all(index -> index in pool, tuple),
    excluded,
  )
  available = max(big(0), total - excluded_here)
  target = Int(min(big(n_tuples), available))
  target == 0 && return Tuple[]

  if total <= big(n_tuples + excluded_here)
    tuples = filter(tuple -> tuple ∉ excluded, _all_bvm_tuples(pool, tuple_size))
    Random.shuffle!(rng, tuples)
    return tuples[1:target]
  end

  sampled = Set{Tuple}()
  while length(sampled) < target
    positions = Random.randperm(rng, length(pool))[1:tuple_size]
    tuple = Tuple(sort(Int[pool[position] for position in positions]))
    tuple ∈ excluded || push!(sampled, tuple)
  end
  return sort!(collect(sampled))
end

function _bvm_tanimoto(A::Tuple, B::Tuple)
  overlap = count(index -> index in B, A)
  return overlap / (length(A) + length(B) - overlap)
end

function _fit_bvm_gp(tuples::AbstractVector{<:Tuple}, energies::AbstractVector{<:Real})
  isempty(tuples) && throw(ArgumentError("the GP requires at least one teacher tuple"))
  length(tuples) == length(energies) ||
    throw(DimensionMismatch("teacher tuples and energies must have equal length"))

  values = float.(collect(energies))
  energy_mean = sum(values) / length(values)
  centered = values .- energy_mean
  energy_scale = sqrt(sum(abs2, centered) / length(values))
  iszero(energy_scale) && (energy_scale = one(energy_scale))
  targets = centered ./ energy_scale

  covariance = [
    _bvm_tanimoto(tuples[i], tuples[j]) + (i == j ? 1e-6 : 0.0)
    for i in eachindex(tuples), j in eachindex(tuples)
  ]
  factor = LinearAlgebra.cholesky(LinearAlgebra.Hermitian(covariance))
  alpha = factor \ targets
  return (
    tuples=collect(tuples),
    energy_mean=energy_mean,
    energy_scale=energy_scale,
    factor=factor,
    alpha=alpha,
  )
end

function _predict_bvm_gp(model::NamedTuple, tuple::Tuple)
  covariance = [_bvm_tanimoto(teacher, tuple) for teacher in model.tuples]
  mean = model.energy_mean + model.energy_scale * LinearAlgebra.dot(covariance, model.alpha)
  whitened = model.factor.L \ covariance
  standardized_variance = max(0.0, 1.0 - LinearAlgebra.dot(whitened, whitened))
  std = model.energy_scale * sqrt(standardized_variance)
  return (mean=mean, std=std)
end

_invalid_bvm_evaluation(reason::Symbol, message::AbstractString, basis_indices) = (
  valid=false,
  energy=Inf,
  reason=reason,
  message=String(message),
  basis_indices=collect(basis_indices),
)

function _is_expected_bvm_numerical_error(error)
  return error isa ArgumentError ||
         error isa DomainError ||
         error isa LinearAlgebra.PosDefException ||
         error isa LinearAlgebra.SingularException ||
         error isa LinearAlgebra.LAPACKException
end

function _evaluate_bvm_tuple(
  hamiltonian::Hamiltonian,
  candidates::BasisSet,
  selected::AbstractVector{<:Integer},
  tuple::Tuple,
  method::BayesianVariationalMethod,
  current_energy::Union{Nothing,Real},
)
  basis_indices = vcat(Int.(selected), collect(tuple))
  basisset = BasisSet((candidates[index] for index in basis_indices)...)

  try
    overlap = matrix(basisset)
    hamiltonian_matrix = matrix(hamiltonian, basisset)
    if !all(isfinite, overlap) || !all(isfinite, hamiltonian_matrix)
      return _invalid_bvm_evaluation(
        :nonfinite_matrix, "Hamiltonian or overlap matrix is non-finite", basis_indices,
      )
    end

    diagonal = LinearAlgebra.diag(overlap)
    if any(value -> !isfinite(value) || value <= 0, diagonal)
      return _invalid_bvm_evaluation(
        :nonpositive_overlap_diagonal,
        "overlap diagonal entries must be positive and finite",
        basis_indices,
      )
    end

    scale = LinearAlgebra.Diagonal(inv.(sqrt.(diagonal)))
    normalized_overlap = LinearAlgebra.Symmetric(scale * overlap * scale)
    minimum_overlap = LinearAlgebra.eigmin(normalized_overlap)
    if minimum_overlap < method.overlap_tol
      return _invalid_bvm_evaluation(
        :ill_conditioned_overlap,
        "normalized overlap eigenvalue $(minimum_overlap) is below $(method.overlap_tol)",
        basis_indices,
      )
    end

    energy = first(LinearAlgebra.eigen(hamiltonian_matrix, overlap).values)
    isfinite(energy) || return _invalid_bvm_evaluation(
      :nonfinite_energy, "generalized eigenvalue is non-finite", basis_indices,
    )

    if !isnothing(current_energy)
      tolerance = 100eps(Float64) * max(1, abs(current_energy))
      energy <= current_energy + tolerance || return _invalid_bvm_evaluation(
        :energy_increase,
        "candidate energy $(energy) exceeds current energy $(current_energy)",
        basis_indices,
      )
    end

    return (
      valid=true,
      energy=energy,
      reason=nothing,
      message="",
      basis_indices=basis_indices,
    )
  catch error
    _is_expected_bvm_numerical_error(error) || rethrow()
    return _invalid_bvm_evaluation(
      :numerical_failure, sprint(showerror, error), basis_indices,
    )
  end
end

function _validate_bvm_candidates(candidates::BasisSet, method::BayesianVariationalMethod)
  length(candidates) >= method.tuple_size || throw(ArgumentError(
    "candidate BasisSet must contain at least tuple_size=$(method.tuple_size) functions",
  ))
  return nothing
end

function _bvm_record(evaluation, tuple, source; prediction=nothing)
  mean = isnothing(prediction) ? missing : prediction.mean
  std = isnothing(prediction) ? missing : prediction.std
  acquisition = isnothing(prediction) ? missing : prediction.acquisition
  return (
    tuple=tuple,
    source=source,
    valid=evaluation.valid,
    energy=evaluation.energy,
    reason=evaluation.reason,
    message=evaluation.message,
    mean=mean,
    std=std,
    acquisition=acquisition,
  )
end

"""
    solve(
        hamiltonian::Hamiltonian,
        candidates::BasisSet,
        method::BayesianVariationalMethod;
        rng=Random.MersenneTwister(123),
        perturbation=Hamiltonian(),
        info=4,
    )

Select groups of basis functions from the finite candidate `BasisSet` with a
Tanimoto-kernel Gaussian-process surrogate, then solve the accepted basis with
the Rayleigh-Ritz method. Candidate energy evaluations are deterministic; the
Bayesian uncertainty describes the discrete search surrogate, not uncertainty
in the returned physical energy.

The result is a `ResultRayleighRitz` with additional `method`,
`candidate_basisset`, `selected_indices`, `history`, `n_evaluations`, and
`n_rejected` properties. Supply an explicit RNG for reproducibility.
"""
function solve(
  hamiltonian::Hamiltonian,
  candidates::BasisSet,
  method::BayesianVariationalMethod;
  rng::Random.AbstractRNG=Random.MersenneTwister(123),
  perturbation::Hamiltonian=Hamiltonian(),
  info::Int=4,
)
  info >= 0 || throw(ArgumentError("BVM requires info to be nonnegative"))
  _validate_bvm_candidates(candidates, method)

  selected = Int[]
  history = NamedTuple[]
  current_energy = nothing
  negligible_steps = 0
  n_evaluations = 0
  n_rejected = 0

  while length(selected) + method.tuple_size <= method.max_basis
    available = setdiff(collect(eachindex(candidates.basis)), selected)
    length(available) >= method.tuple_size || break
    Random.shuffle!(rng, available)
    pool = sort(available[1:min(method.pool_size, length(available))])

    seen = Set{Tuple}()
    teacher_tuples = Tuple[]
    teacher_energies = Float64[]
    evaluations = NamedTuple[]

    function evaluate!(tuples, source; predictions=Dict{Tuple,NamedTuple}())
      for tuple in tuples
        push!(seen, tuple)
        evaluation = _evaluate_bvm_tuple(
          hamiltonian, candidates, selected, tuple, method, current_energy,
        )
        n_evaluations += 1
        n_rejected += evaluation.valid ? 0 : 1
        prediction = get(predictions, tuple, nothing)
        push!(evaluations, _bvm_record(evaluation, tuple, source; prediction=prediction))
        if evaluation.valid
          push!(teacher_tuples, tuple)
          push!(teacher_energies, evaluation.energy)
        end
      end
    end

    while length(teacher_tuples) < method.initial_samples
      tuples = _sample_bvm_tuples(
        rng,
        pool,
        method.tuple_size,
        method.initial_samples - length(teacher_tuples),
        seen,
      )
      isempty(tuples) && break
      evaluate!(tuples, :random)
    end

    isempty(teacher_tuples) && error(
      "BVM found no valid tuple in step $(length(history) + 1); " *
      "$(count(record -> !record.valid, evaluations)) rejected of " *
      "$(length(evaluations)) step evaluations; " *
      "inspect the candidate library or overlap_tol=$(method.overlap_tol)",
    )

    for _ in 1:method.rounds
      model = _fit_bvm_gp(teacher_tuples, teacher_energies)
      proposals = _sample_bvm_tuples(
        rng, pool, method.tuple_size, method.search_size, seen,
      )
      isempty(proposals) && break
      predictions = Dict{Tuple,NamedTuple}()
      for tuple in proposals
        posterior = _predict_bvm_gp(model, tuple)
        predictions[tuple] = (
          mean=posterior.mean,
          std=posterior.std,
          acquisition=posterior.mean - method.beta * posterior.std,
        )
      end
      sort!(proposals, by=tuple -> predictions[tuple].acquisition)
      batch = proposals[1:min(method.batch_size, length(proposals))]
      evaluate!(batch, :gp; predictions=predictions)
    end

    valid_evaluations = filter(record -> record.valid, evaluations)
    isempty(valid_evaluations) && error(
      "BVM found no valid tuple in step $(length(history) + 1); " *
      "$(count(record -> !record.valid, evaluations)) rejected of " *
      "$(length(evaluations)) step evaluations",
    )
    adopted_index = argmin(map(record -> record.energy, valid_evaluations))
    adopted = valid_evaluations[adopted_index]
    previous_energy = current_energy
    append!(selected, adopted.tuple)
    current_energy = adopted.energy
    improvement = isnothing(previous_energy) ? nothing : previous_energy - current_energy
    negligible_steps = !isnothing(improvement) && improvement <= method.abstol ?
      negligible_steps + 1 : 0

    push!(history, (
      step=length(history) + 1,
      pool=pool,
      evaluations=evaluations,
      adopted_tuple=adopted.tuple,
      accepted_energy=current_energy,
      improvement=improvement,
      n_evaluations=n_evaluations,
      n_rejected=n_rejected,
    ))
    negligible_steps >= method.patience && break
  end

  isempty(selected) && error("BVM did not accept any basis functions")
  selected_basisset = BasisSet((candidates[index] for index in selected)...)
  final_result = solve(
    hamiltonian, selected_basisset; perturbation=perturbation, info=info,
  )
  return ResultRayleighRitz(;
    getfield(final_result, :data)...,
    method=method,
    candidate_basisset=candidates,
    selected_indices=copy(selected),
    history=history,
    n_evaluations=n_evaluations,
    n_rejected=n_rejected,
  )
end
