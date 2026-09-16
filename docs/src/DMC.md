```@meta
CurrentModule = TwoBody
```

# Diffusion Monte Carlo

Pure diffusion Monte Carlo propagates walkers in imaginary time. For
``H=-D\nabla^2+V``, one step combines Gaussian diffusion with branching weights

```math
w = \exp[-\Delta t(V-E_\mathrm{ref})].
```

`DiffusionMonteCarlo` keeps the population fixed by systematic resampling and
estimates the ground-state energy from the reference-energy history. A seeded
random-number generator makes calculations reproducible.

## Usage

```@example dmc
using TwoBody

H = Hamiltonian(
  Kinetic(hbar=1, m=1),
  PowerLaw(coefficient=1 / 2, exponent=2),
)
method = DiffusionMonteCarlo(
  n_steps=800,
  equilibration=200,
  n_walkers=1_000,
  Δt=0.01,
)
result = solve(H, method)
result.E
```

Finite time steps and walker populations introduce bias. Convergence should be
checked by reducing ``\Delta t`` and increasing `n_walkers`. See
[Reynolds et al. (1990)](https://doi.org/10.1063/1.4822960) and
[Kosztin et al. (1996)](https://doi.org/10.1119/1.18168).
The reported `standard_error` does not account for autocorrelation.

## API reference

```@docs; canonical=false
DiffusionMonteCarlo
ResultDiffusionMonteCarlo
solve(hamiltonian::Hamiltonian, method::DiffusionMonteCarlo)
```
