# Bayesian Variational Method Design

Date: 2026-08-28

## Purpose

Add the Bayesian Variational Method (BVM) to TwoBody.jl as another variational-ansatz optimizer. BVM constructs a compact additive ansatz by selecting groups of basis functions from a finite candidate library. The linear expansion coefficients remain the responsibility of the existing Rayleigh-Ritz solver.

The first implementation establishes whether Gaussian-process-guided subset selection is useful. It deliberately favors a small, verifiable implementation over the paper's full performance machinery. Incremental diagonalization, trimming, continuous candidate generation, and noisy objectives are later phases, gated on evidence that the selector works.

## Scientific interpretation

For a candidate library \(\{\phi_i\}_{i=1}^M\), BVM constructs

\[
\psi_A = \sum_{i\in A} c_i\phi_i,
\]

where \(A\) is the selected subset. For fixed \(A\), the coefficients and energy are obtained from

\[
H_A c = E_A S_A c.
\]

BVM applies Bayesian optimization to the discrete set function \(A\mapsto E_A\). It does not infer a posterior distribution over the physical energy, wavefunction, or observables. The term "Bayesian" refers specifically to the Gaussian-process surrogate used to select which tuples to evaluate.

## Scope

The first release supports:

- deterministic candidate-energy evaluations over a finite `BasisSet`;
- additive basis expansions evaluated by Rayleigh-Ritz;
- fixed-cardinality tuples with at least two basis functions;
- a Tanimoto-kernel Gaussian process and lower-confidence-bound acquisition;
- reproducible stochastic search with an explicit RNG;
- dense candidate evaluation through the existing solver;
- numerical rejection of unsafe or invalid candidate subsets;
- complete optimization history sufficient for tests and benchmark analysis.

The first release does not support:

- direct optimization of VMC or VNN parameters;
- calibrated Bayesian uncertainty on energies or observables;
- continuous Gaussian-width optimization;
- a general callback-based Bayesian-optimization framework;
- incremental or distributed eigensolution;
- trimming of an already accepted basis;
- a new result-type hierarchy;
- a new external dependency.

## Public API

Export one concrete method type and a short alias:

```julia
BayesianVariationalMethod
const BVM = BayesianVariationalMethod
```

The intended workflow is:

```julia
method = BVM(max_basis=12)
result = solve(hamiltonian, candidates, method; rng=Random.MersenneTwister(123))
```

where `candidates` is an existing `BasisSet`. The method constructor has the following conceptual interface:

```julia
BVM(;
    max_basis,
    pool_size=100,
    tuple_size=3,
    initial_samples=100,
    batch_size=50,
    rounds=4,
    search_size=10_000,
    beta=2.0,
    abstol=1e-8,
    patience=3,
    overlap_tol=1e-10,
)
```

ASCII `beta` is the public keyword; the stored field may use the same name. `abstol` is an absolute energy improvement in the Hamiltonian's units.

The constructor validates all configuration-only constraints. `solve` validates constraints that depend on the candidate library. In particular:

- `max_basis`, `pool_size`, sample counts, batch size, rounds, search size, and patience are positive;
- `2 <= tuple_size <= pool_size`;
- `beta`, `abstol`, and `overlap_tol` are finite and nonnegative, with `overlap_tol > 0`;
- the candidate library contains at least `tuple_size` functions;
- `max_basis >= tuple_size`.

When fewer than `pool_size` unselected candidates remain, the current pool uses all remaining candidates. The search stops before a new tuple would exceed `max_basis`, or when fewer than `tuple_size` candidates remain. Thus `max_basis` need not be divisible by `tuple_size`.

## Package integration

BVM follows the package's existing concrete-method pattern. It adds a `solve` overload rather than pretending that discrete subset selection is an `Optim.jl` continuous optimizer. It reuses:

- `Hamiltonian` as the physical problem;
- `BasisSet` as both candidate library and final ansatz;
- existing matrix-element construction;
- the existing Rayleigh-Ritz matrix construction and generalized-eigenvalue formulation for exact candidate energies;
- `ResultRayleighRitz` for the final state and downstream operations such as `ψ` and expectation values.

No abstract solver hierarchy or one-implementation interface is introduced.

## Algorithm

The selected index set is initially empty. Each outer step performs the following operations.

1. **Pool construction.** Draw up to `pool_size` unselected candidate indices without replacement.
2. **Initial teacher set.** Generate and evaluate unique `tuple_size`-element subsets of the pool until `initial_samples` valid tuples have been collected or the finite tuple space is exhausted. Every attempted evaluation, including a rejection, counts toward the reported evaluation budget.
3. **GP fit.** Standardize the finite teacher energies and fit the Tanimoto-kernel Gaussian process described below.
4. **Proposal search.** Generate up to `search_size` unique, unevaluated tuples. Rank them by lower confidence bound.
5. **Explicit evaluation.** Evaluate up to `batch_size` top-ranked tuples and append their results to the teacher set.
6. **Update.** Repeat GP fitting and proposal evaluation for `rounds` rounds, or stop early when there are no unevaluated tuples.
7. **Adoption.** Among every valid tuple explicitly evaluated during the step, accept the tuple with the lowest exact energy.
8. **Stopping.** Stop when adding another tuple would exceed `max_basis`, too few candidates remain, or the accepted energy improvement is no larger than `abstol` for `patience` consecutive outer steps.

Rejected tuple evaluations remain in the trace but do not enter GP training. If the initial sampling finds no valid teacher tuple after exhausting the available tuples, the outer step fails with the all-tuples-invalid error. The same exact tuple is never evaluated twice within an outer step.

### Gaussian-process model

For fixed-size tuples \(A\) and \(B\), use the Tanimoto kernel

\[
k(A,B)=\frac{|A\cap B|}{|A\cup B|}.
\]

The covariance matrix is

\[
K_{ij}=k(A_i,A_j)+10^{-6}\delta_{ij}.
\]

Teacher energies are standardized to zero mean and unit population variance. If their variance is zero, use centered zero targets and unit scale. Fit and predict using `LinearAlgebra.cholesky` and triangular solves; do not form a matrix inverse.

For an unevaluated tuple, compute posterior mean \(\mu(A)\) and standard deviation \(\sigma(A)\), then rank by

\[
a(A)=\mu(A)-\beta\sigma(A).
\]

`beta = 0` provides the mean-only ablation without a separate method. `tuple_size == 1` is excluded because the Tanimoto kernel has zero covariance between all distinct singleton tuples and therefore cannot generalize among them.

The first implementation may refactor the covariance matrix from scratch after each batch. Sequential GP factor updates are deferred until profiling demonstrates a need.

## Candidate evaluation and numerical safety

For the selected basis plus a trial tuple:

1. Construct the overlap and Hamiltonian matrices with the existing matrix-element functions.
2. Reject non-finite matrix elements.
3. Require every overlap diagonal entry to be finite and strictly positive.
4. Scale the overlap matrix to unit diagonal and compute its smallest eigenvalue.
5. Reject the tuple if that eigenvalue is below `overlap_tol`.
6. Solve the original generalized eigenproblem directly with `LinearAlgebra.eigen`, reusing the matrices already constructed for the safety checks. The final accepted basis is passed through the ordinary public Rayleigh-Ritz `solve` once at the end.
7. Reject non-finite eigenvalues.
8. After the first accepted step, reject a result when

   ```julia
   E_new > E_current + 100eps(Float64) * max(1, abs(E_current))
   ```

   since enlargement of a valid variational subspace cannot raise its minimum energy.

The overlap check is intentionally cubic in the first implementation. Candidate evaluation is already a dense cubic reference path, and correctness is the present goal. The later bordered-Cholesky implementation will replace both operations.

The method does not impose a universal energy floor. TwoBody.jl supports arbitrary Hamiltonians and units, so such a floor would not have a coherent package-wide meaning.

Candidate-local failures are recorded as rejections and do not abort the search. Invalid global configuration throws before work begins. If an outer step contains no valid tuple, `solve` throws an error containing rejection counts and a suggestion to inspect the candidate library or `overlap_tol`.

## Result and diagnostics

After the search, run the ordinary Rayleigh-Ritz solver on the accepted `BasisSet` and return a `ResultRayleighRitz` enriched with:

- `method`;
- `candidate_basisset`;
- `selected_indices` in adoption order;
- `history`;
- `n_evaluations`;
- `n_rejected`.

Each outer-step history entry records:

- step number and pool indices;
- evaluated tuple indices;
- exact energy or rejection reason;
- proposal source (`random` or `gp`);
- GP mean, standard deviation, and acquisition value when applicable;
- adopted tuple;
- accepted energy and energy improvement;
- cumulative evaluation and rejection counts.

The history records enough information to reproduce plots and fixed-budget comparisons without exposing the private GP implementation as public API. An arbitrary supplied RNG is consumed normally; the result does not claim to recover a seed from a mutable RNG object.

## Verification

### Deterministic CI tests

CI tests are small and seeded. They cover:

- exact Tanimoto-kernel values, symmetry, and positive-semidefinite covariance matrices;
- posterior interpolation at teacher tuples within jitter tolerance;
- finite, nonnegative predictive variances;
- unique tuple generation, fixed tuple cardinality, and exclusion of accepted candidates;
- reproducibility from equal initial RNG states;
- unique selected indices and monotonically non-increasing accepted energies;
- equality of the returned BVM energy and a direct solve of the returned basis;
- the full-candidate energy as a lower bound to every selected-subspace energy;
- compatibility of the returned result with `psi`, normalization, and expectation calculations;
- rejection of duplicate or nearly duplicate Gaussian functions;
- validation errors and the all-tuples-invalid error path.

CI does not contain a probabilistic assertion that one stochastic optimizer always beats another.

### Scientific hydrogen benchmark

Use a finite geometric library of Gaussian widths for the hydrogen Hamiltonian. A small case is exhaustively enumerated to obtain the true tuple-energy table. A larger case compares, with identical candidate libraries and explicit-evaluation budgets:

- random tuple selection;
- GP mean-only selection (`beta = 0`);
- lower-confidence-bound BVM;
- the full candidate basis as the reference.

Run approximately 20 seeds and report:

- energy gap to the full-pool solution;
- selected basis dimension;
- number of explicit energy evaluations;
- invalid-tuple rejection rate;
- selected expectation values or sampled wavefunction values relative to the full-basis solution;
- wall time as a secondary metric.

The random baseline is benchmark code, not a second public package method.

## Acceptance and pivot criteria

The selector succeeds when GP-guided selection achieves a lower median energy error than matched random selection, or reaches the same accuracy with a smaller accepted basis, at equal explicit-evaluation budget across the multi-seed hydrogen benchmark.

The result is promising but incomplete when the GP ranks the exhaustive small-pool truth table well but provides little end-to-end advantage. In that case, adjust the pool, sample, and search schedule before changing the kernel.

The design pivots when the Tanimoto GP fails to improve over random search even on exhaustive small problems across reasonable pool sizes. The next investigation is then a physics-informed kernel using basis parameters or matrix-element information. Incremental diagonalization and trimming must not be implemented merely to accelerate an unhelpful selector.

## Later phases

After the selector passes the acceptance criterion:

1. Add bordered overlap updates and warm-started iterative eigensolution, verified against the dense reference path.
2. Add trimming as a separately testable optimization, with an explicit energy-increase budget.
3. Add a candidate-generator hook that can draw or adaptively refine continuous Gaussian widths while presenting a finite pool to the unchanged selector.
4. Reconsider a general optimizer abstraction only when a second backend, such as a noisy VMC objective, has a demonstrated compatible contract.

These phases are not part of the first implementation plan.
