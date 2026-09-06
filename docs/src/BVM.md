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

`history` records every exact tuple evaluation and the GP prediction attached to
each evaluated GP proposal. Supplying an explicit RNG gives reproducible searches.

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
