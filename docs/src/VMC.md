```@meta
CurrentModule = TwoBody
```

# Variational Monte Carlo

Combining the variational energy expectation value with the probability density
and local energy gives the single expression

```math
\begin{aligned}
\langle E \rangle
&= \frac{
  \displaystyle \int \mathrm{d}\mathbf{r}\,
  \psi^*(\mathbf{r})\hat{H}\psi(\mathbf{r})
}{
  \displaystyle \int \mathrm{d}\mathbf{r}\,
  |\psi(\mathbf{r})|^2
} \\
&= \int \mathrm{d}\mathbf{r}\,
\underbrace{
  \frac{|\psi(\mathbf{r})|^2}
  {\displaystyle \int \mathrm{d}\mathbf{r}'\,|\psi(\mathbf{r}')|^2}
}_{P(\mathbf{r})}
\underbrace{
  \frac{\hat{H}\psi(\mathbf{r})}{\psi(\mathbf{r})}
}_{E_\mathrm{loc}(\mathbf{r})}
= \int \mathrm{d}\mathbf{r}\,P(\mathbf{r})E_\mathrm{loc}(\mathbf{r})
\approx \frac{1}{N}\sum_{i=1}^{N}E_\mathrm{loc}(\mathbf{r}_i),
\qquad \mathbf{r}_i \sim P.
\end{aligned}
```

Thus VMC estimates ``\langle E\rangle`` by averaging the local energy over
positions sampled from the normalized density ``P(\mathbf{r})``. This
local-energy formulation and its use in VMC are reviewed by Foulkes *et al.*
[1]. Samples are generated with a symmetric random-walk proposal and the
Metropolis acceptance rule [2]. The Laplacian in a non-relativistic kinetic term
is evaluated by forward-mode automatic differentiation with ForwardDiff.jl [4].

## Usage

The following example uses a one-Gaussian trial wavefunction for the hydrogen
atom. A nonzero initial position avoids starting exactly at the Coulomb
singularity.

```@example vmc-hydrogen
using Random
using TwoBody

# Hamiltonian
H = Hamiltonian(
  NonRelativisticKinetic(ℏ=1, m=1),
  CoulombPotential(coefficient=-1),
)

# Trial wave function
α = 0.2829
ψ(r) = exp(-α * sum(abs2, r))

# VMC options
method = VariationalMonteCarlo(
  n_steps=2_000,
  burn_in=500,
  δ=2.0,
  r₀=[1.0, 0.0, 0.0],
)

# Solve
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

## Multiple walkers and equilibration

Thijssen [5, Table 12.1] describes calculations with 300 walkers, 12,000
attempted displacements per walker, and the first 2,000 states of each walker
discarded for equilibration. With `thinning=1`, the same sampling counts are
specified by retaining the remaining 10,000 states per walker:

```julia
method = VariationalMonteCarlo(
  n_walkers=300,
  n_steps=10_000,
  burn_in=2_000,
  thinning=1,
  δ=2.0,
  r₀=[1.0, 0.0, 0.0],
)

result = solve(H, ψ, method; rng=MersenneTwister(123), info=0)

result.E        # expectation value of the energy
result.variance # sample variance of the retained local energies
result.n_attempted          # 3_600_000 attempted displacements
result.n_burn_in_discarded  # 600_000 states discarded for equilibration
```

Each walker performs `burn_in + n_steps * thinning` transitions, so the
configuration above attempts 12,000 displacements per walker and retains
3,000,000 samples in total. Samples and local energies are stored consecutively
by walker; for example, they can be grouped as
`reshape(result.local_energies, method.n_steps, method.n_walkers)`. Walkers are
advanced sequentially using separate random draws from the supplied random
number generator.

`result.n_discarded` has a different meaning: it counts retained samples whose
local energy was non-finite and therefore excluded from the energy statistics.

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
5. J. M. ティッセン 著, 松田和典, 道廣嘉隆, 谷村吉隆, 高須昌子, 吉江友照 訳,
   『計算物理学』, 丸善出版 (2012).

## API reference

```@docs; canonical=false
TwoBody.VariationalMonteCarlo
TwoBody.local_energy
TwoBody.solve(hamiltonian::Hamiltonian, wavefunction::Function, method::VariationalMonteCarlo)
```
