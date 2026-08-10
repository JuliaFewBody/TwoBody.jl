# TwoBody.jl release notes

## v0.0.11 — 2026-08-10

TwoBody.jl v0.0.11 adds variational Monte Carlo calculations and a benchmark
Hamiltonian database, improves the Rayleigh–Ritz workflow, and simplifies the
public operator API. This release contains breaking API changes; see the
migration notes below.

### Highlights

- Added `VariationalMonteCarlo` with symmetric Metropolis sampling, burn-in,
  thinning, multiple walkers, reproducible random-number generation, sampling
  diagnostics, and ForwardDiff-based local-energy evaluation.
- Added a benchmark Hamiltonian database with lookup and registration support
  for reusable physical systems.
- Improved Rayleigh–Ritz results and examples, including updated Antique.jl
  integration.
- Added tested VMC and database guides and made database examples executable in
  Documenter.

### Breaking changes

Hamiltonian operator types now use concise mathematical names. The old names
have been removed rather than retained as aliases:

| Before | v0.0.11 |
| --- | --- |
| `NonRelativisticKinetic` | `Kinetic` |
| `ConstantPotential` | `Constant` |
| `LinearPotential` | `Linear` |
| `CoulombPotential` | `Coulomb` |
| `PowerLawPotential` | `PowerLaw` |
| `GaussianPotential` | `Gaussian` |
| `ExponentialPotential` | `Exponential` |
| `YukawaPotential` | `Yukawa` |
| `DeltaPotential` | `Delta` |
| `FunctionPotential` | `Custom` |
| `UniformGridPotential` | `Tabulated` |

For example, update

```julia
Hamiltonian(
  NonRelativisticKinetic(ℏ=1, m=1),
  CoulombPotential(coefficient=-1),
)
```

to

```julia
Hamiltonian(
  Kinetic(ℏ=1, m=1),
  Coulomb(coefficient=-1),
)
```

### Compatibility and maintenance

- Updated Antique.jl examples and tests for Antique.jl 0.15.
- Fixed documentation deployment after the repository moved to the
  `JuliaFewBody` organization.
- Updated stable, development, repository, and documentation links for the new
  repository location.

### Included pull requests

- #16 — Add the benchmark Hamiltonian database.
- #19 — Update Antique.jl test API usage.
- #22 — Add variational Monte Carlo.
- #23 — Simplify Hamiltonian operator names.
- #24 — Fix documentation deployment after the repository transfer.
- #25 — Make database documentation examples executable.
