```@meta
CurrentModule = TwoBody
```

# Stochastic Variational Method

The stochastic variational method (SVM) chooses the nonlinear parameters of the
basis set by trial and error. The basis set is grown one function at a time.
At each step a number of exponents

```math
a \in [a_\mathrm{min}, a_\mathrm{max}]
```

is drawn log-uniformly, the ground-state energy

```math
E[\phi_1, \cdots, \phi_k, \phi_\mathrm{trial}]
```

is evaluated for each trial function by the Rayleigh-Ritz method, and the
candidate with the lowest energy is adopted. Since the energy is an upper bound
of the exact eigenvalue and never increases when a function is added, the
competition of random candidates is a safe and derivative-free way of finding a
compact basis set.

The growth phase is followed by `sweeps` cyclic refinement passes. Each pass
revisits every basis function in turn, draws replacements for its exponent, and
keeps the best of the current and the trial functions, which is the refinement
procedure of [K. Varga, Y. Suzuki (1995)](https://arxiv.org/abs/nucl-th/9508023).

The energy of a rejected candidate, for example one that makes the basis set
nearly linearly dependent, is not an error: the candidate is simply not
adopted, and the reason is recorded in the history.

## Usage

```@example svm-hydrogen
using TwoBody

H = Hamiltonian(
  Kinetic(hbar=1, m=1),
  Coulomb(coefficient=-1),
)
method = SVM(
  max_basis=10,
  candidates=20,
  amin=1e-2,
  amax=1e+2,
  sweeps=1,
)
result = solve(H, SimpleGaussianBasis(), method, info=1)
(energy=result.E[1], evaluations=result.n_evaluations, rejected=result.n_rejected)
```

Supplying an explicit RNG, `rng=Random.MersenneTwister(2026)`, makes a search
reproducible; the default RNG is seeded, so the example above is reproducible
as well.

The second positional argument is a prototype: only its type and its quantum
numbers are used, and the exponents are sampled. A `BasisSet` is a warm start
instead, whose functions are kept and extended up to `max_basis`:

```@example svm-hydrogen
grown = solve(H, result.basisset, SVM(max_basis=12, candidates=20, amin=1e-2, amax=1e+2), info=1)
(energy=grown.E[1], size=length(grown.basisset))
```

Every step of `history` records the adopted exponent, the accepted energy, and
all the candidates evaluated in that step:

```@example svm-hydrogen
step = result.history[3]
(kind=step.kind, index=step.index, exponent=step.exponent, energy=step.energy,
 improvement=step.improvement, candidates=length(step.evaluations))
```

The exponents of the stochastic basis set are unstructured, unlike the
geometric progression of the
[Gaussian Expansion Method](@ref Gaussian-Expansion-Method). Refining them
further with the [Gradient Variational Method](@ref Gradient-Variational-Method)
is the recommended workflow:

```@example svm-hydrogen
solve(H, result.basisset, GVM(), info=1, progress=false).E[1]
```

## API reference

```@docs; canonical=false
TwoBody.StochasticVariationalMethod
TwoBody.solve(
  hamiltonian::Hamiltonian,
  basisset::BasisSet,
  method::StochasticVariationalMethod,
)
```
