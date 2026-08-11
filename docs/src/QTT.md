```@meta
CurrentModule = TwoBody
```

# Quantics Tensor Train

The quantics tensor train (QTT) solver rewrites a radial grid of ``N=2^q`` points as
``q`` binary sites and approximates the sampled wave function by a chain of tensor
cores,

```math
f(r_i)=F_{b_1\ldots b_q}
\simeq G^{(1)}(b_1)\cdots G^{(q)}(b_q),
\qquad
i-1=\sum_{k=1}^{q}b_k2^{q-k}.
```

The Hamiltonian is stored as a matrix product operator (MPO). Two-site DMRG finds
its low-energy states, with projector penalties exposing successive excitations,

```math
H^{(n)}_{\mathrm{QTT}}u_n=E_nu_n,
\qquad
H^{(n)}_{\mathrm{QTT}}
=H_{\mathrm{QTT}}+\mu\sum_{k=0}^{n-1}|u_k\rangle\langle u_k|.
```

## Theory

Each vector core has dimensions
``G^{(k)}\in\mathbb{R}^{\chi_{k-1}\times2\times\chi_k}``, with
``\chi_0=\chi_q=1``. A matrix is quantized in both its row and column indices and
represented as

```math
A_{b_1\ldots b_q,\,c_1\ldots c_q}
\simeq W^{(1)}(b_1,c_1)W^{(2)}(b_2,c_2)\cdots W^{(q)}(b_q,c_q).
```

The central second-difference operator can be written using the one-point shift
matrices ``S_-`` and ``S_+`` as

```math
D^{(2)} = \frac{S_- - 2I + S_+}{\Delta r^2}.
```

This tridiagonal operator has an exact QTT/MPO representation with maximum bond
dimension three. Potential functions and initial wave functions are compressed by
tensor cross interpolation. Two-site DMRG sweeps optimize the tensor cores and adapt
their bond dimensions; intermediate MPOs are compressed between solves.

If ``\chi`` and ``\rho`` bound the vector and MPO bond dimensions, their storage is
bounded by

```math
\operatorname{storage}(F_{\mathrm{QTT}}) \leq 2q\chi^2,
\qquad
\operatorname{storage}(A_{\mathrm{QTT}}) \leq 4q\rho^2.
```

Thus, when the QTT ranks remain moderate, storage grows as ``O(\log N)`` instead of
``O(N)`` for a dense vector or ``O(N^2)`` for a dense matrix.

## Usage

For the three-dimensional spherical oscillator in atomic units, configure the QTT
grid with `quantics`; the number of interior grid points is `2^quantics`. Here a
coarse grid is used so the example runs quickly.

```@example qtt
using TwoBody
using Random

Random.seed!(1234)

H = Hamiltonian(
  Kinetic(hbar=1, m=1),
  PowerLaw(coefficient=1 / 2, exponent=2),
)

method = QuanticsTensorTrainMethod(
  quantics=5,
  r₀=0.0,
  rₘₐₓ=8.0,
  l=0,
  tolerance=1e-8,
  maxbonddim=16,
  maxoperatorbonddim=32,
  sweeps=4,
)

result = solve(
  H,
  method;
  initial=r -> exp(-r^2 / 2),
  nₘₐₓ=2,
  info=0,
)
result.E
```

The values approach the exact ``l=0`` oscillator energies ``E_0=3/2`` and
``E_1=7/2`` as the grid is refined. Inspect the eigenvector ranks without expanding
them:

```@example qtt
ranks.(result.C)
```

The overlap matrix and residual norms provide convergence diagnostics:

```@example qtt
result.overlaps
```

```@example qtt
result.residuals
```

Individual entries can also be contracted directly:

```@example qtt
qttvalue(result.ψ[1], 1)
```

!!! note "Current scope"
    `QuanticsTensorTrainMethod` currently supports `Kinetic`, `RestEnergy`,
    `Constant`, `Linear`, `Coulomb`, `PowerLaw`, `Gaussian`, `Exponential`, `Yukawa`,
    and `Custom` operators. Choose `deflation_shift` above the relevant spectral gaps,
    monitor MPO ranks, overlaps, and residuals, and perform a grid-convergence study.

## Acknowledgments

The proof of concept for this QTT solver was developed at
[CompPhysHack 2026](https://qc-hybrid.github.io/CompPhysHack2026/) in collaboration
with Lucas Arenstein. We thank him and the hackathon organizers for their
contributions and support.

## Bibliography

The corresponding
[proof-of-concept implementation](https://github.com/ohno/CompPhysHack2026Ohno/blob/main/julia/qtt.jl)
is archived in the CompPhysHack repository. The current solver is based on the
tensor-train operations and two-site DMRG eigensolver provided by
[TensorTrainNumerics.jl](https://github.com/MartinMikkelsen/TensorTrainNumerics.jl).
Background on the QTT construction is given by Arenstein, Mikkelsen, and Kastoryano in
[Fast and Flexible Quantum-Inspired Differential Equation Solvers with Data Integration](https://doi.org/10.48550/arXiv.2505.17046).

## API reference

```@docs; canonical=false
QuanticsTensorTrainMethod
solve(hamiltonian::Hamiltonian, method::QuanticsTensorTrainMethod)
QTTVector
QTTMatrix
order
ranks
qttvalue
```
