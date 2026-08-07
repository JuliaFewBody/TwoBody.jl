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

at positions sampled from ``|\psi(\mathbf{r})|^2``. This local-energy
formulation and its use in VMC are reviewed by Foulkes *et al.* [1]. Samples are
generated with a symmetric random-walk proposal and the Metropolis acceptance
rule [2]. The Laplacian in a non-relativistic kinetic term is evaluated by
forward-mode automatic differentiation with ForwardDiff.jl [4].

## Usage

The following example uses a one-Gaussian trial wavefunction for the hydrogen
atom. A nonzero initial position avoids starting exactly at the Coulomb
singularity.

```@example vmc-hydrogen
using Random
using TwoBody

H = Hamiltonian(
  NonRelativisticKinetic(ℏ=1, m=1),
  CoulombPotential(coefficient=-1),
)

α = 0.2829
ψ(r) = exp(-α * sum(abs2, r))

method = VariationalMonteCarlo(
  n_steps=2_000,
  burn_in=500,
  δ=2.0,
  r₀=[1.0, 0.0, 0.0],
)

result = solve(H, ψ, method; rng=MersenneTwister(123), info=0)
gaussian_expectation = 3α / 2 - 2sqrt(2α / π)

(;
  VMC_energy=round(result.E; digits=4),
  gaussian_expectation=round(gaussian_expectation; digits=4),
  exact_energy=-0.5,
  acceptance_rate=round(result.acceptance_rate; digits=3),
)
```

The sampled VMC energy is close to the analytical expectation value of the same
Gaussian trial wavefunction. That expectation value is above the exact hydrogen
ground-state energy, as required by the variational principle. Increasing the
number of Monte Carlo samples reduces statistical noise but does not remove the
variational bias caused by the restricted one-Gaussian trial wavefunction. A
finite-sample VMC estimate can fluctuate to either side of its expectation value.

The result also contains the sample variance, a naive standard error, the
acceptance rate, local energies, and sampled positions. Because
successive Markov-chain samples are correlated, use batching or an autocorrelation
analysis when a rigorous uncertainty estimate is required; the blocking method
of Flyvbjerg and Petersen [3] is one standard approach.

Non-finite local energies at isolated singular points are excluded from the
average and counted in `result.n_discarded`.

## Bibliography

1. W. M. C. Foulkes, L. Mitas, R. J. Needs, and G. Rajagopal,
   ["Quantum Monte Carlo simulations of solids"](https://doi.org/10.1103/RevModPhys.73.33),
   *Reviews of Modern Physics* **73**, 33–83 (2001).
2. N. Metropolis, A. W. Rosenbluth, M. N. Rosenbluth, A. H. Teller, and E. Teller,
   ["Equation of State Calculations by Fast Computing Machines"](https://doi.org/10.1063/1.1699114),
   *The Journal of Chemical Physics* **21**, 1087–1092 (1953).
3. H. Flyvbjerg and H. G. Petersen,
   ["Error estimates on averages of correlated data"](https://doi.org/10.1063/1.457480),
   *The Journal of Chemical Physics* **91**, 461–466 (1989).
4. J. Revels, M. Lubin, and T. Papamarkou,
   ["Forward-Mode Automatic Differentiation in Julia"](https://arxiv.org/abs/1607.07892),
   arXiv:1607.07892 (2016).

## API reference

```@docs; canonical=false
TwoBody.VariationalMonteCarlo
TwoBody.local_energy
TwoBody.solve(hamiltonian::Hamiltonian, wavefunction::Function, method::VariationalMonteCarlo)
```
