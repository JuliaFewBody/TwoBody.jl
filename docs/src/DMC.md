```@meta
CurrentModule = TwoBody
```

# Diffusion Monte Carlo

Pure diffusion Monte Carlo propagates walkers in imaginary time. For
``H=-D\nabla^2+V``, one step moves every walker by a Gaussian displacement of
variance ``2D\Delta t`` from ``\pmb{r}`` to ``\pmb{r}'`` and assigns the branching weight

```math
w = \exp\left[-\Delta t\left(\frac{V(\pmb{r})+V(\pmb{r}')}{2}-E_\mathrm{ref}\right)\right],
```

with the exponent capped at `log_weight_cap` so that a walker near a
singularity of the potential cannot take over the population. `DiffusionMonteCarlo`
keeps the population fixed by systematic resampling and estimates the
ground-state energy as the average of the growth estimator
``E_\mathrm{ref}-\log\langle w\rangle/\Delta t`` after equilibration. A seeded
random-number generator makes a calculation reproducible within one Julia
version.

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
The reported `standard_error` is estimated from block averages and still
underestimates the spread between independent runs by a factor of about three
at the default settings; compare several seeds for a reliable error. No
importance sampling is used, so the Coulomb potential carries a bias of a few
percent at these settings and relies on the cap of the branching weight.

## API reference

```@docs; canonical=false
DiffusionMonteCarlo
ResultDiffusionMonteCarlo
solve(hamiltonian::Hamiltonian, method::DiffusionMonteCarlo)
```
