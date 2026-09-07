```@meta
CurrentModule = TwoBody
```

# Free Complement Method

The Free Complement (FC) method systematically improves a trial wave function
by generating functions from the Schrödinger equation. Starting from
``\psi_n``, one iteration is

```math
\psi_{n+1} = \left[1 + C_n g(H-E_n)\right]\psi_n,
```

where ``g`` is a scaling function that regularizes singular terms. The general
theory is described by Nakatsuji [1].

!!! note
    TwoBody.jl implements `FC` for `PowerSlaterBasis`. For an s-wave hydrogen Hamiltonian and the default ``g(r)=r``, it generates the basis-function forms in ``g(H-E_n)``, removes duplicates, and discards functions singular at the origin. The coefficients ``C_n`` are determined variationally by the existing Rayleigh–Ritz solver, whose analytic matrix elements for this basis cover the `Kinetic` and `Coulomb` terms used in the example below.

## Theory

This procedure can be interpreted as an algorithm for generating basis
functions. Consider the basis function ``\phi(r)=r^n\mathrm{e}^{-ar}``. When
the Hamiltonian acts on it,

```math
\begin{aligned}
\hat{H} \phi(r)
=&
- \frac{1}{2} \frac{\partial^2 \phi}{{\partial r}^2}(r)
- \frac{1}{r} \frac{\partial\phi}{\partial r}(r)
- \frac{1}{r} \phi(r) \\
=
&- \frac{1}{2} a^2 r^n \mathrm{e}^{-ar}
 + a n r^{n-1} \mathrm{e}^{-ar}
 - \frac{1}{2} n(n-1)r^{n-2} \mathrm{e}^{-ar} \\
&- a r^{n-1} \mathrm{e}^{-ar}
 + n r^{n-2} \mathrm{e}^{-ar}
 - r^{n-1} \mathrm{e}^{-ar} \\
\end{aligned}
```

the expression splits into several terms. Their coefficients are not important
for generating the basis. What matters is that the following three basis
functions are obtained:

```math
r^n \mathrm{e}^{-ar} \\
r^{n-1} \mathrm{e}^{-ar} \\
r^{n-2} \mathrm{e}^{-ar}
```

Thus, when the coefficients are ignored, it is easy to predict which basis
functions will be generated. However, basis functions that diverge at ``r=0``
are removed, so generating ``r^{n+1}`` is preferable to generating
``r^{n-1}``. Multiplication by ``g(r)=r`` gives the following three basis
functions:

```math
r^{n+1} \mathrm{e}^{-ar} \\
r^{n  } \mathrm{e}^{-ar} \\
r^{n-1} \mathrm{e}^{-ar}
```

These include newly generated basis functions that do not diverge. Starting
from ``\phi(r)=\mathrm{e}^{-ar}``, the nonsingular basis functions increase as
follows:

```math
\mathrm{e}^{-ar} \\
\Downarrow \\
\mathrm{e}^{-ar} \\
r \mathrm{e}^{-ar} \\
\Downarrow \\
\mathrm{e}^{-ar} \\
r \mathrm{e}^{-ar} \\
r^2 \mathrm{e}^{-ar} \\
\Downarrow \\
\mathrm{e}^{-ar} \\
r \mathrm{e}^{-ar} \\
r^2 \mathrm{e}^{-ar} \\
r^3 \mathrm{e}^{-ar} \\
\Downarrow \\
\vdots
```

The implementation only needs to encode this rule.

## Usage

The following REPL session shows how repeated application of `FC` expands a
power-Slater basis set:

```@repl fc-basis-generation
using TwoBody
H = Hamiltonian(
  Kinetic(hbar = 1, m = 1),
  Coulomb(coefficient = -1),
)
BS = BasisSet(PowerSlaterBasis(0, 1.5))
FC(H, BS)
FC(H, FC(H, BS))
FC(H, FC(H, FC(H, BS)))
```

### Example of Hydrogen Atom

Define the non-relativistic hydrogen Hamiltonian in atomic units and start from
``e^{-1.5r}``:

```@example fc
using TwoBody

H = Hamiltonian(
  Kinetic(hbar = 1, m = 1),
  Coulomb(coefficient = -1),
)

basisset = BasisSet(PowerSlaterBasis(0, 1.5))
nothing # hide
```

At each iteration, solve the generalized eigenvalue problem and generate the
next complement. The table compares the resulting variational energies with
the reference values:

```@example fc
using Printf

reference_energies = [
  "-0.375",
  "-0.491 025 404",
  "-0.499 316 143",
  "-0.499 954 132",
  "-0.499 997 229",
  "-0.499 999 844",
  "-0.499 999 992",
  "-0.500 000 000",
  "-0.500 000 000",
]
@printf("%5s  %-18s  %s\n", "M_n", "This work", "Ref.")
println("-----  ------------------  ------------")
for reference_energy in reference_energies
  result = solve(H, basisset)
  @printf(
    "%5d  %18.15f  %s\n",
    length(basisset),
    result.E[1],
    reference_energy,
  )
  global basisset = FC(H, basisset)
end
```

You can also generate a complement from one basis function:

```@repl fc-single
using TwoBody
H = Hamiltonian(
  Kinetic(hbar = 1, m = 1),
  Coulomb(coefficient = -1),
)
FC(H, PowerSlaterBasis(0, 1.5))
```

## Acknowledgments

This work was developed on the basis of the fourth lecture in Section I of the
[64th Summer School of the Young Researchers’ Association for Molecular
Science](https://www.natsugaku2025.ymsa.jp/home), held in Kanazawa on August
20, 2025. The authors gratefully acknowledge Professor Hiroshi Nakatsuji and
all those involved in organizing the summer school for their contributions and
support.

## Bibliography

1. H. Nakatsuji, “Scaled Schrödinger Equation and the Exact Wave Function,”
   [*Phys. Rev. Lett.* **93**, 030403 (2004)](https://doi.org/10.1103/PhysRevLett.93.030403).

## API reference

```@docs; canonical=false
PowerSlaterBasis
FC
```
