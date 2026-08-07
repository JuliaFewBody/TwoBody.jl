```@meta
CurrentModule = TwoBody
```

# Variational Monte Carlo

Variational Monte Carlo estimates the energy of a trial wavefunction from the
local energy

```math
E_\mathrm{loc}(\mathbf{r}) =
\frac{\hat{H}\psi(\mathbf{r})}{\psi(\mathbf{r})}
```

at positions sampled from ``|\psi(\mathbf{r})|^2``. The Laplacian in a
non-relativistic kinetic term is evaluated by automatic differentiation with
ForwardDiff.jl.

## Usage

The following example uses a one-Gaussian trial wavefunction for the hydrogen
atom. A nonzero initial position avoids starting exactly at the Coulomb
singularity.

```julia
using Random
using TwoBody

H = Hamiltonian(
  NonRelativisticKinetic(ℏ=1, m=1),
  CoulombPotential(coefficient=-1),
)

α = 0.2829
ψ(r) = exp(-α * sum(abs2, r))

method = VariationalMonteCarlo(
  n_steps=100_000,
  burn_in=1_000,
  δ=0.5,
  r₀=[1.0, 0.0, 0.0],
)

result = solve(H, ψ, method; rng=MersenneTwister(123))
result.E
```

The result also contains the sample variance, a naive standard error, the
acceptance rate, local energies, and sampled positions. Because
successive Markov-chain samples are correlated, use batching or an autocorrelation
analysis when a rigorous uncertainty estimate is required.

Non-finite local energies at isolated singular points are excluded from the
average and counted in `result.n_discarded`.

## API reference

```@docs; canonical=false
TwoBody.VariationalMonteCarlo
TwoBody.local_energy
TwoBody.solve(hamiltonian::Hamiltonian, wavefunction::Function, method::VariationalMonteCarlo)
```
