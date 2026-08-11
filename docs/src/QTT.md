```@meta
CurrentModule = TwoBody
```

# Quantics Tensor Train

The quantics tensor train (QTT) method represents a radial grid of ``2^q`` points as
``q`` binary tensor sites. A sampled radial function is then approximated by a chain
of small tensor cores rather than stored as a vector with ``2^q`` entries,

```math
f(r_i)=F_{b_1\ldots b_q}
\simeq G^{(1)}(b_1)G^{(2)}(b_2)\cdots G^{(q)}(b_q).
```

The Hamiltonian is represented in the same form as a matrix product operator (MPO).
DMRG solves the eigenvalue problem directly in tensor-train form; previously found
states are shifted out of the low-energy spectrum to expose successive excitations,

```math
H^{(n)}_{\mathrm{QTT}}u_n=E_nu_n,
\qquad
H^{(n)}_{\mathrm{QTT}}
=H_{\mathrm{QTT}}+\mu\sum_{k=0}^{n-1}|u_k\rangle\langle u_k|.
```

When the tensor-train ranks remain moderate, both storage and contraction costs grow
with ``q=\log_2 N`` rather than with the full grid size ``N``.

## Theory

For ``N=2^q`` interior grid points, write the zero-based grid index in binary,

```math
i-1 = \sum_{k=1}^{q} b_k 2^{q-k},
\qquad b_k \in \{0,1\}.
```

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
their bond dimensions. After each state is found, its projector is added to the MPO
and the result is compressed before the next DMRG solve.

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
``E_1=7/2`` as the grid is refined. `result.ψ`, `result.u`, and `result.C` contain
the compressed radial wave functions, physically normalized reduced radial wave
functions, and unit-norm DMRG eigenvectors. Their tensor-train ranks can be inspected
without expanding any object:

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
    and `Custom` operators. Excited states use projector deflation, so
    `deflation_shift` must exceed the relevant spectral gaps; also monitor MPO ranks,
    overlaps, and residuals. `r₀` and `rₘₐₓ` are zero-value (Dirichlet) boundaries and
    the ``2^q`` grid points lie strictly between them, so choose boundaries where the
    reduced radial wave function is zero or sufficiently small and perform a
    grid-convergence study.

## Acknowledgments

This work was developed in collaboration with Lucas Arenstein at
[CompPhysHack 2026](https://qc-hybrid.github.io/CompPhysHack2026/). The authors
gratefully acknowledge all those involved in organizing the hackathon for their
contributions and support.

## Bibliography

The implementation follows the finite-difference QTT construction explored in the
[CompPhysHack 2026 proof of concept](https://github.com/ohno/CompPhysHack2026Ohno/blob/main/julia/qtt.jl)
and uses the DMRG and tensor-train operations provided by
[TensorTrainNumerics.jl](https://github.com/MartinMikkelsen/TensorTrainNumerics.jl).
The QTT construction is described by Arenstein, Mikkelsen, and Kastoryano in
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
