# Bayesian Variational Method Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible Bayesian variational optimizer that selects compact Gaussian basis subsets and returns a normal `ResultRayleighRitz` with complete search diagnostics.

**Architecture:** Add one concrete `BayesianVariationalMethod` and one `solve` overload. Keep tuple generation, the Tanimoto Gaussian process, numerical candidate evaluation, and the search loop as private helpers in one focused source file; reuse `BasisSet`, matrix construction, and `ResultRayleighRitz`. Establish selector quality with dense diagonalization before any incremental eigensolver or trimming work.

**Tech Stack:** Julia 1.10+, TwoBody.jl's existing `LinearAlgebra` and `Random` dependencies, existing Rayleigh-Ritz/GEM matrix elements, `Test`, and Documenter.

**Spec:** `docs/superpowers/specs/2026-08-28-bayesian-variational-method-design.md`

## Global Constraints

- Add no package dependency; implement the small GP with `LinearAlgebra`.
- Export only `BayesianVariationalMethod` and its alias `BVM`; add no abstract optimizer hierarchy.
- The public keyword is ASCII `beta`, not `β`.
- Require `tuple_size >= 2`; singleton tuples are outside this implementation.
- Candidate energies are deterministic generalized-eigenvalue results over a finite `BasisSet`.
- Use an explicit `Random.AbstractRNG`; default to a fresh `Random.MersenneTwister(123)`.
- Reject unsafe candidate subsets using a unit-diagonal overlap eigenvalue check with default `overlap_tol=1e-10`.
- Do not add an energy floor, continuous candidate generation, VMC/VNN support, incremental diagonalization, distributed execution, or trimming.
- Keep stochastic superiority assertions out of CI; evaluate them in `dev/bvm.jl` across approximately 20 seeds.
- Follow test-driven development: observe every focused test fail before adding its implementation.

## File Structure

- Create `src/BVM.jl`: method configuration, tuple sampling, GP helpers, candidate evaluation, BVM search, docstrings, and the public `solve` overload.
- Modify `src/TwoBody.jl`: include `BVM.jl` after the Rayleigh-Ritz and GEM definitions.
- Create `test/BVM.jl`: deterministic unit, numerical-safety, integration, and error-path tests.
- Modify `test/runtests.jl`: include the new focused test file.
- Create `dev/bvm.jl`: exhaustive small-pool probe and multi-seed hydrogen comparison with a private random baseline.
- Create `docs/src/BVM.md`: method explanation, limitations, and a runnable hydrogen example.
- Modify `docs/make.jl`: add the BVM manual page to navigation.
- Modify `README.md`: add the BVM guide link to the existing method list.

---

### Task 1: Method Configuration and Unique Tuple Sampling

**Files:**
- Create: `src/BVM.jl`
- Modify: `src/TwoBody.jl:12-18`
- Create: `test/BVM.jl`
- Modify: `test/runtests.jl:14-24`

**Interfaces:**
- Consumes: `BasisSet` indexing and length; `Random.AbstractRNG`.
- Produces: `BayesianVariationalMethod`, `BVM`, `_all_bvm_tuples(pool, tuple_size)`, and `_sample_bvm_tuples(rng, pool, tuple_size, n_tuples, excluded)`.

- [ ] **Step 1: Write failing constructor and sampling tests**

Create `test/BVM.jl` with imports that also allow the file to run directly:

```julia
using Test
using TwoBody
import LinearAlgebra
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
```

Add `include("BVM.jl")` after `include("GEM.jl")` in `test/runtests.jl`.

- [ ] **Step 2: Run the focused test and verify the API is missing**

Run:

```bash
julia --project=. test/BVM.jl
```

Expected: failure with `UndefVarError: BVM not defined`.

- [ ] **Step 3: Implement the validated method type**

Start `src/BVM.jl` with:

```julia
export BayesianVariationalMethod, BVM

import LinearAlgebra
import Random

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

const BVM = BayesianVariationalMethod

Base.string(method::BayesianVariationalMethod) =
  "BayesianVariationalMethod(" *
  join(
    ("$(name)=$(getproperty(method, name))" for name in fieldnames(typeof(method))),
    ", ",
  ) *
  ")"
Base.show(io::IO, method::BayesianVariationalMethod) = print(io, string(method))
```

Add `include("./BVM.jl")` after `include("./GEM.jl")` in `src/TwoBody.jl`.

- [ ] **Step 4: Implement bounded tuple enumeration and random sampling**

Add these private helpers to `src/BVM.jl`:

```julia
function _all_bvm_tuples(pool::AbstractVector{<:Integer}, tuple_size::Int)
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
```

The exhaustive branch only executes when enumeration is bounded by the requested count plus already evaluated tuples. The large-space branch samples without materializing a combinatorial candidate set.

- [ ] **Step 5: Run the focused and full tests**

Run:

```bash
julia --project=. test/BVM.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: the new BVM tests pass; the existing suite remains green.

- [ ] **Step 6: Commit the configuration and sampling unit**

```bash
git add src/BVM.jl src/TwoBody.jl test/BVM.jl test/runtests.jl
git commit -m "Add BVM configuration and tuple sampling"
```

---

### Task 2: Tanimoto Gaussian-Process Surrogate

**Files:**
- Modify: `src/BVM.jl`
- Modify: `test/BVM.jl`

**Interfaces:**
- Consumes: sorted fixed-size tuples produced by `_sample_bvm_tuples`.
- Produces: `_bvm_tanimoto(A, B)::Float64`, `_fit_bvm_gp(tuples, energies)::NamedTuple`, and `_predict_bvm_gp(model, tuple)::NamedTuple{(:mean, :std)}`.

- [ ] **Step 1: Add failing kernel and posterior tests**

Append inside the top-level test file:

```julia
@testset "BVM Gaussian process" begin
  tuples = Tuple[(1, 2, 3), (1, 2, 4), (4, 5, 6)]
  @test TwoBody._bvm_tanimoto(tuples[1], tuples[1]) == 1.0
  @test TwoBody._bvm_tanimoto(tuples[1], tuples[2]) == 0.5
  @test TwoBody._bvm_tanimoto(tuples[1], tuples[3]) == 0.0
  @test TwoBody._bvm_tanimoto(tuples[1], tuples[2]) ==
        TwoBody._bvm_tanimoto(tuples[2], tuples[1])

  covariance = [TwoBody._bvm_tanimoto(a, b) for a in tuples, b in tuples]
  @test LinearAlgebra.eigmin(LinearAlgebra.Symmetric(covariance)) >= -1e-12

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
```

- [ ] **Step 2: Run the focused test and verify the GP helpers are missing**

Run `julia --project=. test/BVM.jl`.

Expected: failure with `_bvm_tanimoto not defined`.

- [ ] **Step 3: Implement the kernel and Cholesky-based GP fit**

Add to `src/BVM.jl`:

```julia
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
```

- [ ] **Step 4: Implement posterior mean and standard deviation**

```julia
function _predict_bvm_gp(model::NamedTuple, tuple::Tuple)
  covariance = [_bvm_tanimoto(teacher, tuple) for teacher in model.tuples]
  mean = model.energy_mean + model.energy_scale * dot(covariance, model.alpha)
  whitened = model.factor.L \ covariance
  standardized_variance = max(0.0, 1.0 - dot(whitened, whitened))
  std = model.energy_scale * sqrt(standardized_variance)
  return (mean=mean, std=std)
end
```

Do not compute `inv(covariance)`. The Cholesky solve supplies both the posterior coefficients and predictive variance.

- [ ] **Step 5: Run tests and commit the surrogate**

Run:

```bash
julia --project=. test/BVM.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: all tests pass.

Commit:

```bash
git add src/BVM.jl test/BVM.jl
git commit -m "Add Tanimoto surrogate for BVM"
```

---

### Task 3: Safe Candidate Energy Evaluation

**Files:**
- Modify: `src/BVM.jl`
- Modify: `test/BVM.jl`

**Interfaces:**
- Consumes: `Hamiltonian`, candidate `BasisSet`, accepted indices, one tuple, `BayesianVariationalMethod`, and `current_energy::Union{Nothing,Real}`.
- Produces: `_evaluate_bvm_tuple(...)` returning `(valid, energy, reason, message, basis_indices)`.

- [ ] **Step 1: Add failing exact-energy and rejection tests**

Append:

```julia
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
```

- [ ] **Step 2: Run the focused test and verify the evaluator is missing**

Run `julia --project=. test/BVM.jl`.

Expected: failure with `_evaluate_bvm_tuple not defined`.

- [ ] **Step 3: Implement explicit invalid-result records**

Add:

```julia
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
```

- [ ] **Step 4: Implement normalized-overlap screening and dense energy evaluation**

```julia
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
```

Unexpected programming errors must propagate; only the listed numerical or candidate-domain failures become rejected tuple records.

- [ ] **Step 5: Run tests and commit the evaluator**

Run:

```bash
julia --project=. test/BVM.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: all tests pass, including the duplicate-basis rejection.

Commit:

```bash
git add src/BVM.jl test/BVM.jl
git commit -m "Add safe BVM candidate evaluation"
```

---

### Task 4: End-to-End Bayesian Variational Search

**Files:**
- Modify: `src/BVM.jl`
- Modify: `test/BVM.jl`

**Interfaces:**
- Consumes: all helpers from Tasks 1-3 and the existing `solve(Hamiltonian, BasisSet)`.
- Produces: `solve(hamiltonian::Hamiltonian, candidates::BasisSet, method::BayesianVariationalMethod; rng, perturbation, info)::ResultRayleighRitz`.

- [ ] **Step 1: Add failing deterministic integration tests**

Append:

```julia
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
  @test first_result.n_evaluations > 0
  @test first_result.n_rejected >= 0

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
  @test_throws ErrorException solve(
    hydrogen, duplicates, invalid_method; rng=Random.MersenneTwister(2),
  )
end
```

- [ ] **Step 2: Run the focused test and verify the solve overload is missing**

Run `julia --project=. test/BVM.jl`.

Expected: `MethodError` for `solve(::Hamiltonian, ::BasisSet, ::BayesianVariationalMethod)`.

- [ ] **Step 3: Add solve-time validation and evaluation-record construction**

Add these helpers before the public method:

```julia
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
```

- [ ] **Step 4: Implement one outer search step inside the public solve method**

Use the following state and loop structure. Keep the per-step `evaluate!` function nested so it captures the accepted basis and teacher data without introducing a public evaluator object:

```julia
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
```

- [ ] **Step 5: Add the public docstrings**

Place this docstring immediately before the type:

```julia
@doc raw"""
`BayesianVariationalMethod(; max_basis, pool_size=100, tuple_size=3,
initial_samples=100, batch_size=50, rounds=4, search_size=10_000,
beta=2.0, abstol=1e-8, patience=3, overlap_tol=1e-10)`

Configure Bayesian selection of fixed-size groups from a finite candidate basis.
`max_basis` limits the accepted basis dimension; `pool_size` is the temporary
candidate pool; `tuple_size` is the number of functions adopted per outer step;
`initial_samples`, `batch_size`, `rounds`, and `search_size` control the
Gaussian-process search budget; and `beta` weighs posterior uncertainty in the
lower-confidence-bound acquisition. `abstol` is an absolute energy improvement
in the Hamiltonian's units, `patience` counts consecutive negligible steps, and
`overlap_tol` rejects nearly linearly dependent subsets.

`BVM` is an alias for `BayesianVariationalMethod`.
""" BayesianVariationalMethod
```

Place this docstring immediately before the solve method:

```julia
@doc raw"""
`solve(hamiltonian, candidates, method::BayesianVariationalMethod; rng, perturbation, info)`

Select groups of basis functions from the finite candidate `BasisSet` with a
Tanimoto-kernel Gaussian-process surrogate, then solve the accepted basis with
the Rayleigh-Ritz method. Candidate energy evaluations are deterministic; the
Bayesian uncertainty describes the discrete search surrogate, not uncertainty
in the returned physical energy.

The result is a `ResultRayleighRitz` with additional `method`,
`candidate_basisset`, `selected_indices`, `history`, `n_evaluations`, and
`n_rejected` properties. Supply an explicit RNG for reproducibility.
""" solve(
  hamiltonian::Hamiltonian,
  candidates::BasisSet,
  method::BayesianVariationalMethod;
  rng::Random.AbstractRNG=Random.MersenneTwister(123),
  perturbation::Hamiltonian=Hamiltonian(),
  info::Int=4,
)
```

- [ ] **Step 6: Run focused and full verification**

Run:

```bash
julia --project=. test/BVM.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: deterministic BVM selections match for equal RNG states; the full suite passes.

- [ ] **Step 7: Commit the working solver**

```bash
git add src/BVM.jl test/BVM.jl
git commit -m "Implement Bayesian variational basis search"
```

---

### Task 5: Hydrogen Acceptance Benchmark

**Files:**
- Create: `dev/bvm.jl`

**Interfaces:**
- Consumes: public BVM API and private `_all_bvm_tuples`, `_fit_bvm_gp`, `_predict_bvm_gp`, `_sample_bvm_tuples`, and `_evaluate_bvm_tuple` for diagnostic-only benchmark logic.
- Produces: terminal CSV tables for an exhaustive small-pool probe and a 20-seed matched-budget comparison.

- [ ] **Step 1: Create the exhaustive truth-table probe**

Start `dev/bvm.jl` with:

```julia
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
```

This probe reports zero rank when the true optimum was already in the teacher set; otherwise it reports its mean-prediction rank among unseen tuples.

- [ ] **Step 2: Add a private equal-budget random baseline**

Implement this script-local function:

```julia
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
```

- [ ] **Step 3: Add the 20-seed matched-budget comparison**

Append:

```julia
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
```

- [ ] **Step 4: Run the benchmark and apply the selector gate**

Run:

```bash
julia --project=. dev/bvm.jl
```

Expected: one truth-table line followed by 60 CSV rows, comprising LCB, mean-only, and random results for 20 seeds.

Compute the median `energy_gap` for each method from the emitted rows. Proceed without changing the kernel if the LCB or mean-only GP has a smaller median gap than random at a comparable evaluation count. If neither GP method improves on random, stop feature expansion and open a follow-up design discussion for a basis-parameter or matrix-element kernel; do not add incremental diagonalization or trimming.

- [ ] **Step 5: Commit the reproducible benchmark**

```bash
git add dev/bvm.jl
git commit -m "Add hydrogen benchmark for BVM selection"
```

---

### Task 6: User Documentation and Final Verification

**Files:**
- Create: `docs/src/BVM.md`
- Modify: `docs/make.jl:17-32`
- Modify: `README.md:31-42`

**Interfaces:**
- Consumes: the final public BVM constructor, solve overload, and result properties.
- Produces: a discoverable manual page and API documentation with a runnable hydrogen example.

- [ ] **Step 1: Create the BVM manual page**

Create `docs/src/BVM.md` with this structure and runnable example:

````markdown
```@meta
CurrentModule = TwoBody
```

# Bayesian Variational Method

The Bayesian Variational Method (BVM) constructs a compact Rayleigh-Ritz
ansatz by selecting groups of functions from a finite candidate basis. A
Gaussian-process surrogate predicts the energy obtained by adding each group,
and a lower-confidence-bound acquisition rule balances low predicted energy
against uncertain candidates.

The posterior uncertainty belongs to the search surrogate. It is not an error
bar on the returned energy or wavefunction.

## Usage

```@example bvm-hydrogen
using TwoBody

H = Hamiltonian(
  Kinetic(hbar=1, m=1),
  Coulomb(coefficient=-1),
)
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
)
result = solve(H, candidates, method)
(energy=result.E[1], selected=result.selected_indices,
 evaluations=result.n_evaluations)
```

`history` records every exact tuple evaluation, rejection, GP prediction, and
adoption decision. Supplying an explicit RNG gives reproducible searches.

The initial implementation uses dense diagonalization and does not yet include
continuous candidate generation, incremental diagonalization, or trimming.

## API reference

```@docs; canonical=false
TwoBody.BayesianVariationalMethod
TwoBody.solve(
  hamiltonian::Hamiltonian,
  candidates::BasisSet,
  method::BayesianVariationalMethod,
)
```
````

- [ ] **Step 2: Add navigation and README discovery links**

In `docs/make.jl`, add:

```julia
"Bayesian Variational Method" => "BVM.md",
```

immediately after the Gaussian Expansion Method page.

In the README user-guide list, add:

```markdown
- [Bayesian Variational Method](https://juliafewbody.github.io/TwoBody.jl/dev/BVM/)
```

immediately after the Gaussian Expansion Method link.

- [ ] **Step 3: Run documentation and package verification**

Run:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=docs docs/make.jl
git diff --check
```

Expected: all tests pass, Documenter reports no missing exported-symbol documentation, the BVM example executes, and `git diff --check` prints nothing.

- [ ] **Step 4: Review the final diff against the specification**

Run:

```bash
git diff --stat HEAD~6
git diff HEAD~6 -- src/BVM.jl test/BVM.jl dev/bvm.jl docs/src/BVM.md docs/make.jl README.md
```

Confirm all of the following from the diff:

- only `BayesianVariationalMethod` and `BVM` are newly exported;
- `Project.toml` is unchanged;
- no incremental solver, trimming, generic callback interface, or noisy-objective code was added;
- every result property named in the spec is populated;
- benchmark superiority is not asserted in CI or documentation.

- [ ] **Step 5: Commit the documentation and final integration**

```bash
git add docs/src/BVM.md docs/make.jl README.md
git commit -m "Document Bayesian variational method"
```

- [ ] **Step 6: Record final verification evidence**

Run:

```bash
git status --short
git log -6 --oneline
```

Expected: the worktree is clean and the BVM implementation is represented by the configuration, GP, evaluator, solver, benchmark, and documentation commits from this plan.
