```@meta
CurrentModule = TwoBody
```

# Variational Monte Carlo

VMC estimates ``\langle E\rangle`` by averaging the local energy over positions sampled from the normalized density ``P(\mathbf{r})``:

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
&= \frac{1}{Z}\int \mathrm{d}\mathbf{r}\,
\psi^*(\mathbf{r})\hat{H}\psi(\mathbf{r})
&& \qquad\cdots\qquad {\textstyle
Z = \int \mathrm{d}\mathbf{r}\,|\psi(\mathbf{r})|^2} \\
&= \int \mathrm{d}\mathbf{r}\,
\frac{1}{Z}|\psi(\mathbf{r})|^2
\cdot
\frac{\hat{H}\psi(\mathbf{r})}{\psi(\mathbf{r})}
&& \qquad\cdots\qquad {\textstyle
|\psi(\mathbf{r})|^2 = \psi^*(\mathbf{r})\psi(\mathbf{r})} \\
&= \int \mathrm{d}\mathbf{r}\,P(\mathbf{r})\cdot E_\mathrm{loc}(\mathbf{r})
&& \qquad\cdots\qquad {\textstyle
P(\mathbf{r}) = \frac{1}{Z}|\psi(\mathbf{r})|^2,\quad
E_\mathrm{loc}(\mathbf{r}) =
\frac{\hat{H}\psi(\mathbf{r})}{\psi(\mathbf{r})}}
\\
&\approx \frac{1}{N}\sum_{i=1}^{N}E_\mathrm{loc}(\mathbf{r}_i)
&& \qquad\cdots\qquad {\textstyle \mathbf{r}_i \sim P.}
\end{aligned}
```

This local-energy formulation is reviewed by Foulkes *et al.*
[1]. Samples are generated with a symmetric random-walk proposal and the
Metropolis acceptance rule [2]. The Laplacian in a non-relativistic kinetic term
is evaluated by forward-mode automatic differentiation with ForwardDiff.jl [4].

## Usage

The following example uses the hydrogen trial wavefunction ``\psi(r)=\exp(-0.8r)`` and sampling conditions from Thijssen [5, Table 12.1]: 300 walkers, 12,000 attempted displacements per walker, and the first 2,000 states of each walker discarded for equilibration (`burn_in=2_000`; `n_steps=10_000` counts the retained samples). A nonzero initial position avoids starting exactly at the Coulomb singularity.

```@example vmc-hydrogen
using TwoBody

# Hamiltonian
H = Hamiltonian(
  Kinetic(hbar=1, m=1),
  Coulomb(coefficient=-1),
)

# Trial wave function
α = 0.8
ψ(r) = exp(-α * sqrt(sum(abs2, r)))

# VMC options
method = VariationalMonteCarlo(
  n_walkers=300,
  n_steps=10_000,
  burn_in=2_000,
  thinning=1,
  δ=2.0,
  r₀=[1.0, 0.0, 0.0],
)

# Solve
result = solve(H, ψ, method)

# Display
println("This work: $(result.E)")
println("Reference: -0.4813(6)")
println("Exact    : -0.4800")
```

For ``\psi(r)=\exp(-\alpha r)``, the analytical expectation value in atomic units is ``\alpha^2/2-\alpha=-0.48`` at ``\alpha=0.8``; Thijssen reports ``-0.4813(6)``. A finite-sample estimate fluctuates around the analytical expectation value; more samples reduce statistical noise but not the trial wavefunction's variational bias.

`solve` also returns sampling diagnostics and retained data; see the [API reference](#API-reference) for details.

Because successive Markov-chain samples are correlated, the naive `standard_error` reported by `solve` typically underestimates the true uncertainty; rigorous estimates require batching or autocorrelation analysis, such as the blocking method of Flyvbjerg and Petersen [3].

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
5. J. M. Thijssen,
   [*Computational Physics*, 2nd ed.](https://doi.org/10.1017/CBO9781139171397),
   Cambridge University Press (2007).

## API reference

```@docs; canonical=false
TwoBody.VariationalMonteCarlo
TwoBody.local_energy
TwoBody.solve(hamiltonian::Hamiltonian, wavefunction::Function, method::VariationalMonteCarlo)
```
