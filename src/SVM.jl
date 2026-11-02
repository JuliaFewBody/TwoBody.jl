export StochasticVariationalMethod, SVM

import Random

# type

"""
    StochasticVariationalMethod(; max_basis, candidates=25, amin=1e-4, amax=1e+4,
                                sweeps=0, abstol=0.0, patience=3,
                                overlap_tol=1e-10)

Configure the stochastic variational method (SVM) of Kukulin, Krasnopol'sky,
Varga, and Suzuki. The basis set is grown one function at a time: `candidates`
exponents are drawn log-uniformly from `[amin, amax]`, each one is scored by
the resulting Rayleigh-Ritz ground-state energy, and the best admissible
candidate is adopted. `max_basis` is the final number of basis functions,
including the functions of a warm start.

`sweeps` cyclic refinement passes follow the growth phase. Each pass revisits
every basis function in turn, draws `candidates` replacements for its exponent,
and keeps the best of the current and the trial functions.

`abstol` is an absolute energy improvement in the Hamiltonian's units and
`patience` counts the consecutive growth steps whose improvement does not
exceed `abstol`; growth stops when the count reaches `patience`. `overlap_tol`
rejects a candidate that makes the basis set nearly linearly dependent.

The exponent range is basis dependent, for example ``a = 1/r^2`` for
[`GaussianBasis`](@ref) and ``a = 1/r`` for [`PowerSlaterBasis`](@ref).
Narrowing `[amin, amax]` to the length scale of the system is the most
effective way of improving the convergence.

Reference: K. Varga, Y. Suzuki, *Phys. Rev. C* **52**, 2885 (1995),
[arXiv:nucl-th/9508023](https://arxiv.org/abs/nucl-th/9508023).
"""
struct StochasticVariationalMethod{T<:AbstractFloat} <: SolverMethod
  max_basis::Int
  candidates::Int
  amin::T
  amax::T
  sweeps::Int
  abstol::T
  patience::Int
  overlap_tol::T
end

function StochasticVariationalMethod(;
  max_basis::Int,
  candidates::Int=25,
  amin::Real=1e-4,
  amax::Real=1e+4,
  sweeps::Int=0,
  abstol::Real=0.0,
  patience::Int=3,
  overlap_tol::Real=1e-10,
)
  max_basis > 0 || throw(ArgumentError("max_basis must be positive"))
  candidates > 0 || throw(ArgumentError("candidates must be positive"))
  isfinite(amin) && amin > 0 ||
    throw(ArgumentError("amin must be positive and finite"))
  isfinite(amax) && amax > amin ||
    throw(ArgumentError("amax must be finite and greater than amin"))
  sweeps >= 0 || throw(ArgumentError("sweeps must be nonnegative"))
  isfinite(abstol) && abstol >= 0 ||
    throw(ArgumentError("abstol must be nonnegative and finite"))
  patience > 0 || throw(ArgumentError("patience must be positive"))
  isfinite(overlap_tol) && overlap_tol > 0 ||
    throw(ArgumentError("overlap_tol must be positive and finite"))

  amin_value, amax_value, abstol_value, overlap_tol_value =
    promote(float(amin), float(amax), float(abstol), float(overlap_tol))
  return StochasticVariationalMethod{typeof(amin_value)}(
    max_basis,
    candidates,
    amin_value,
    amax_value,
    sweeps,
    abstol_value,
    patience,
    overlap_tol_value,
  )
end

"""
    SVM(; kwargs...)

Alias for [`StochasticVariationalMethod`](@ref).
"""
const SVM = StochasticVariationalMethod

# sampling

function _svm_exponent(rng::Random.AbstractRNG, method::StochasticVariationalMethod)
  return exp(log(method.amin) + rand(rng) * (log(method.amax) - log(method.amin)))
end

function _svm_record(exponent::Real, evaluation)
  return (
    exponent=exponent,
    valid=evaluation.valid,
    energy=evaluation.energy,
    reason=evaluation.reason,
    message=evaluation.message,
  )
end

# solver

"""
    solve(
        hamiltonian::Hamiltonian,
        prototype::Basis,
        method::StochasticVariationalMethod;
        rng=Random.MersenneTwister(123),
        perturbation=Hamiltonian(),
        info=4,
    )
    solve(
        hamiltonian::Hamiltonian,
        basisset::BasisSet,
        method::StochasticVariationalMethod;
        prototype=last(basisset.basis),
        rng=Random.MersenneTwister(123),
        perturbation=Hamiltonian(),
        info=4,
    )

Grow a basis set by stochastic trial and error, then solve it with the
Rayleigh-Ritz method. The first form starts from an empty basis set and
samples exponents into copies of `prototype`, which fixes the type of the
basis function and its quantum numbers. The second form warm-starts from
`basisset` and keeps its functions; `prototype` defaults to its last function.

The result is a `ResultRayleighRitz` with additional `method`,
`initialbasisset`, `history`, `n_evaluations`, and `n_rejected` properties.
Each entry of `history` is one growth (`kind=:growth`) or refinement
(`kind=:refinement`) step and records every candidate evaluated in it. Supply
an explicit RNG for reproducibility.
"""
function solve(
  hamiltonian::Hamiltonian,
  basisset::BasisSet,
  method::StochasticVariationalMethod;
  prototype::Basis=_svm_prototype(basisset),
  rng::Random.AbstractRNG=Random.MersenneTwister(123),
  perturbation::Hamiltonian=Hamiltonian(),
  info::Int=4,
)
  # Forward `info` to Rayleigh-Ritz semantics (negative values are allowed there).
  length(basisset) <= method.max_basis || throw(ArgumentError(
    "the initial BasisSet has $(length(basisset)) functions, more than max_basis=$(method.max_basis)",
  ))

  basis = Basis[basisset.basis...]
  history = NamedTuple[]
  n_evaluations = 0
  n_rejected = 0
  negligible_steps = 0

  current = if isempty(basis)
    nothing
  else
    evaluation = _variational_energy(
      hamiltonian, BasisSet(basis...); overlap_tol=method.overlap_tol,
    )
    evaluation.valid || error(
      "SVM cannot start from the given BasisSet: $(evaluation.message)",
    )
    evaluation.energy
  end

  # growth
  while length(basis) < method.max_basis
    evaluations = NamedTuple[]
    best_exponent = nothing
    best_energy = Inf
    for _ in 1:method.candidates
      exponent = _svm_exponent(rng, method)
      evaluation = _variational_energy(
        hamiltonian,
        BasisSet(basis..., _replace_exponent(prototype, exponent));
        overlap_tol=method.overlap_tol,
        reference=current,
      )
      n_evaluations += 1
      n_rejected += evaluation.valid ? 0 : 1
      push!(evaluations, _svm_record(exponent, evaluation))
      if evaluation.valid && evaluation.energy < best_energy
        best_exponent = exponent
        best_energy = evaluation.energy
      end
    end

    isnothing(best_exponent) && error(
      "SVM found no valid candidate in step $(length(history) + 1); " *
      "$(count(record -> !record.valid, evaluations)) rejected of " *
      "$(length(evaluations)) step evaluations; " *
      "inspect the exponent range [$(method.amin), $(method.amax)] or " *
      "overlap_tol=$(method.overlap_tol)",
    )

    push!(basis, _replace_exponent(prototype, best_exponent))
    improvement = isnothing(current) ? nothing : current - best_energy
    current = best_energy
    negligible_steps = !isnothing(improvement) && improvement <= method.abstol ?
      negligible_steps + 1 : 0

    push!(history, (
      step=length(history) + 1,
      kind=:growth,
      sweep=0,
      index=length(basis),
      accepted=true,
      exponent=best_exponent,
      energy=current,
      improvement=improvement,
      evaluations=evaluations,
      n_evaluations=n_evaluations,
      n_rejected=n_rejected,
    ))
    negligible_steps >= method.patience && break
  end

  isempty(basis) && error("SVM did not accept any basis function")

  # refinement
  for sweep in 1:method.sweeps
    for index in eachindex(basis)
      evaluations = NamedTuple[]
      replaced = basis[index]
      best_exponent = nothing
      best_energy = current
      for _ in 1:method.candidates
        exponent = _svm_exponent(rng, method)
        trial = copy(basis)
        trial[index] = _replace_exponent(replaced, exponent)
        evaluation = _variational_energy(
          hamiltonian, BasisSet(trial...); overlap_tol=method.overlap_tol,
        )
        n_evaluations += 1
        n_rejected += evaluation.valid ? 0 : 1
        push!(evaluations, _svm_record(exponent, evaluation))
        if evaluation.valid && evaluation.energy < best_energy
          best_exponent = exponent
          best_energy = evaluation.energy
        end
      end

      accepted = !isnothing(best_exponent)
      improvement = current - best_energy
      if accepted
        basis[index] = _replace_exponent(replaced, best_exponent)
        current = best_energy
      end

      push!(history, (
        step=length(history) + 1,
        kind=:refinement,
        sweep=sweep,
        index=index,
        accepted=accepted,
        exponent=accepted ? best_exponent : _exponent(basis[index]),
        energy=current,
        improvement=improvement,
        evaluations=evaluations,
        n_evaluations=n_evaluations,
        n_rejected=n_rejected,
      ))
    end
  end

  # result
  result = solve(
    hamiltonian, BasisSet(basis...); perturbation=perturbation, info=info,
  )
  return ResultRayleighRitz(;
    getfield(result, :data)...,
    method=method,
    initialbasisset=basisset,
    history=history,
    n_evaluations=n_evaluations,
    n_rejected=n_rejected,
  )
end

function solve(
  hamiltonian::Hamiltonian,
  prototype::Basis,
  method::StochasticVariationalMethod;
  rng::Random.AbstractRNG=Random.MersenneTwister(123),
  perturbation::Hamiltonian=Hamiltonian(),
  info::Int=4,
)
  return solve(
    hamiltonian,
    BasisSet(),
    method;
    prototype=prototype,
    rng=rng,
    perturbation=perturbation,
    info=info,
  )
end

function _svm_prototype(basisset::BasisSet)
  isempty(basisset.basis) && throw(ArgumentError(
    "a cold start needs a prototype basis function: " *
    "call solve(hamiltonian, prototype, method) or pass prototype=...",
  ))
  return last(basisset.basis)
end
