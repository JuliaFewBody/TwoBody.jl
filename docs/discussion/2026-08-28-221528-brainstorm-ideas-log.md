# Ideas Session — 2026-08-28 22:15

## Opening context

**User:** I want to work on a new method called Bayesian Variational Method (BVM) based on the attached paper, *Bayesian Variational Method for Precision Few-Body Calculations*. Let's first discuss how to do this and what is needed.

**Mentor:** Classified the work as an architectural research-design discussion. Before reading the paper into a proposed implementation, the session will clarify the scientific objective, distinguish possible meanings of BVM, inspect the mathematical and numerical requirements, and define validation criteria. No implementation will begin without agreement on the design.

The TwoBody.jl repository already contains several relevant baselines: Rayleigh-Ritz, Gaussian expansion method (GEM), variational Monte Carlo (VMC), variational neural networks (VNN), finite differences, and QTT. There is no prior brainstorming log or project knowledge base. Two optional advisor perspectives are available: Xi Dai and Lei Wang.

## Advisor selection and first paper reading

**Mentor:** Offered three ways to continue: Lei Wang (recommended for variational methods, Monte Carlo, probabilistic modeling, and principled validation), Xi Dai (basis expansions, limiting cases, and hidden assumptions), or no additional advisor.

**User:** Selected Lei Wang.

**Paper-grounded reading:** The paper's BVM is a bundle of three separable ideas:

1. A Bayesian-optimization search layer chooses a tuple `A` of `K` candidate basis functions. The exact target `E = f(A)` is the ground-state energy obtained after adding that tuple to the accepted basis. A Gaussian process with a Tanimoto/Jaccard kernel on multi-hot tuple encodings predicts a mean and uncertainty, and the lower confidence bound `a(A) = mu(A) - beta sigma(A)` selects proposals.
2. A numerical-linear-algebra layer evaluates each tuple with a bordered Cholesky update and a warm-started LOBPCG solve, empirically reducing candidate evaluation from full cubic rediagonalization to quadratic cost when the iteration count remains bounded.
3. A basis-management layer trims accepted functions whose constrained removal raises the energy by less than a threshold, controlling redundancy and near-linear dependence.

The Bayesian component is therefore a posterior over the unknown discrete objective function used for candidate selection. The Ritz solve, pool generation, greedy tuple adoption, incremental eigensolver, and trimming are not Bayesian. Equation (3), written as `P(theta | E)`, is motivational rather than an implemented posterior over physical parameters, final wavefunctions, energies, or observables.

**Paper results:** For a fixed grid of 32,000 GEM basis functions for the helium-4 pentamer, the method reaches the full-grid energy within 0.01 K using 705--820 functions and within 0.001 K using 2,127--2,209 functions over ten seeds. Against an equal-evaluation random tuple search, the GPR-guided method reaches the 0.001 K target with a 10--23% smaller basis. The headline time comparison uses 501 nodes for BVM and one node for conventional SVM; the equal-evaluation random-tuple ablation is the cleaner evidence for the surrogate's contribution.

**Lei Wang advisor perspective:** Treat the method as GP-guided variational basis selection unless the project explicitly intends to add calibrated uncertainty on physical quantities. The central assumptions to test are whether Jaccard overlap predicts tuple energy, whether the exploration term contributes beyond mean-only ranking, whether the greedy pool-local search generalizes across systems, and whether overlap conditioning keeps the iterative solve stable. Validation should include exhaustive small-pool truth tables, fixed-budget random/mean-only/LCB ablations, energy variance and observable convergence, conditioning sensitivity, and at least one on-demand matrix-element calculation rather than only a precomputed full matrix.

**Repository fit:** TwoBody.jl already provides `BasisSet`, Gaussian bases, matrix-element construction, and a Rayleigh--Ritz `solve`, but the current path constructs the complete `H` and `S` matrices and calls a dense generalized eigensolver. It has no candidate-pool abstraction, incremental solver state, GP selector, trimming state, or basis-search result/history type. The present two-body benchmarks also use bases of tens to low hundreds of functions, so reproducing the paper's scalability claim inside TwoBody.jl alone may require a deliberately larger benchmark or eventual transfer to FewBody.jl.

## Initial direction

**Mentor:** Asked whether BVM should be a faithful basis-compression accelerator, a genuinely Bayesian solver with uncertainty on physical quantities, or a general variational-ansatz optimizer. Recommended beginning with the first interpretation as a minimum viable result while preserving room to generalize.

**User:** Chose the general optimizer framing: TwoBody.jl already contains many variational methods, so BVM should be another variational-ansatz optimizer.

**Mentor's interpretation:** BVM should have a method-level identity in the package and use GEM/basis selection as the first backend, rather than being hard-coded as a GEM feature. However, the paper's mechanism directly supports only *additive discrete ansatzes*: a finite library of candidate components whose linear coefficients are eliminated by a generalized eigenproblem. Extending the same name to continuous neural-network parameters or noisy VMC objectives would require different kernels, likelihoods, acquisition logic, and optimization contracts.

**Mentor:** Proposed a package-level interface in which `BayesianVariationalMethod` selects an optimized subset from a candidate library and returns the optimized basis, coefficients, energy, selection history, GP predictions, and conditioning diagnostics. Asked whether the initial implementation should accept only a finite basis library or also generate/optimize continuous parameters such as Gaussian widths.

**User:** Asked for a recommendation.

**Recommended staged scope:** Build the scientific core first for deterministic, discrete additive basis libraries. This matches the paper, fits the existing `BasisSet`/Rayleigh--Ritz machinery, permits exhaustive small-case ground truth, and makes the Bayesian contribution measurable. Design the pool behind a generator interface from the outset, so a second stage can draw or adaptively refine continuous Gaussian widths while presenting a finite pool to the same selector at each step. Defer direct VMC/VNN support: their continuous and noisy objectives require a materially different probabilistic model and should not complicate or weaken the first BVM result.

**User:** Approved the staged scope.

## Benchmark and architecture

**Mentor:** Recommended hydrogen as the first correctness benchmark, with a small Gaussian candidate library for which tuple energies can be enumerated exactly, followed by a larger package benchmark comparing BVM with matched random and greedy selection across seeds. Proposed success criteria: reproducibility, preservation of the variational upper bound, safe handling of nearly singular overlap matrices, and improvement over baselines at equal evaluation budget.

**User:** Approved hydrogen as the first benchmark.

**Architecture alternatives considered:**

1. Treat BVM as an `Optim.jl` optimizer used through the current `optimize` path. This reuses a familiar name but is a poor conceptual fit: the current optimizer varies continuous Gaussian exponents, while BVM selects discrete subsets and needs persistent pool/teacher/selection state.
2. Add a concrete `BayesianVariationalMethod` used as `solve(H, candidates::BasisSet, method; rng, info)`. The method owns discrete selection, delegates each exact energy evaluation to the existing Rayleigh--Ritz machinery, and returns the final Rayleigh--Ritz result enriched with selected indices and optimization history. This follows the existing solver style and is the recommended architecture.
3. Add a fully generic callback API such as `bayesian_select(objective, candidates, method)`. This is flexible enough for future VMC/VNN backends but introduces an abstraction before a second compatible consumer exists and obscures the physics-level package interface.

**Minimal implementation recommendation:** Choose alternative 2. Add one concrete method type and one `solve` overload; reuse `BasisSet`, `Hamiltonian`, matrix construction, and `ResultRayleighRitz`. Implement the first GP with `LinearAlgebra` and the Tanimoto kernel rather than adding a GP dependency. Validate the search with ordinary dense rediagonalization first. Incremental diagonalization and trimming are performance phases to add only after fixed-budget tests show that the Bayesian selector itself works.

**User:** Approved the concrete-solver architecture.

**Component/data-flow design presented:** Export `BayesianVariationalMethod` and the alias `BVM`. Configure a maximum accepted basis size, pool and tuple sizes, initial teacher count, proposed batch size, number of GP rounds, search-set size, exploration weight, and energy-based stopping criteria. Each outer step draws a pool, evaluates initial random tuples, fits a standardized near-noiseless Tanimoto-kernel GP, ranks randomly generated candidate tuples by lower confidence bound, evaluates the best batch, updates the GP, and adopts the explicitly evaluated tuple with the lowest actual energy. The final `ResultRayleighRitz` carries the method, candidate library, selected indices, evaluation history, and total evaluation count. The first implementation uses the existing dense Rayleigh--Ritz solver for correctness. A tuple size of at least two was recommended because the overlap kernel has no generalization between distinct singleton tuples.

**User:** Approved the component and data-flow design.

**Numerical-safety design presented:** Normalize candidate overlap matrices to unit diagonal and reject tuples whose smallest normalized overlap eigenvalue is below a configurable tolerance (initially `1e-10`). Reject non-finite matrix elements and energies, verify variational monotonicity within floating-point tolerance, and record rejection reasons. Invalid method configurations throw before the search; an outer step throws clearly only when no valid tuple exists. Use explicit RNG injection with a reproducible default, unique generated tuples, fixed internal GP jitter, and a safe equal-target path when the teacher-energy variance vanishes. Do not expose a universal energy floor because the package supports arbitrary Hamiltonians and units.

**User:** Approved the numerical-safety design.

**Verification design presented:** Separate deterministic CI checks from a multi-seed scientific benchmark. CI covers the Tanimoto kernel, GP interpolation and variances, tuple uniqueness and reproducibility, solver invariants, result composability, numerical rejections, and validation errors. The hydrogen benchmark compares random tuples, mean-only GP, lower-confidence-bound BVM, and the full candidate reference with identical evaluation budgets over roughly 20 seeds. Success means a lower median energy gap or a smaller basis at matched accuracy; good truth-table ranking without end-to-end improvement triggers schedule tuning; no advantage over random even on exhaustive small cases triggers a kernel pivot before performance work.

**User:** Approved the verification design. All design sections are now approved; the architectural specification is being written for review before implementation planning.
