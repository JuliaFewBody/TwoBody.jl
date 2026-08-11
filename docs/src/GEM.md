```@meta
CurrentModule = TwoBody
```

# Gaussian Expansion Method

The Gaussian expansion method (GEM) uses the existing Rayleigh–Ritz solver
with normalized Gaussian primitives,

```math
\psi_{lm}(\boldsymbol r) = \sum_{n=1}^{n_\mathrm{max}} c_n
N_{nl} r^l e^{-\nu_n r^2}Y_{lm}(\hat{\boldsymbol r}).
```

The ranges are normally placed in a geometric progression so that one basis
set covers both short- and long-distance behavior. The implementation follows
the formulation reviewed by [Hiyama, Kino, and Kamimura
(2003)](https://www.sciencedirect.com/science/article/abs/pii/S0146641003900159?via%3Dihub)
and uses TwoBody.jl's ordinary `solve` function for the generalized
Rayleigh–Ritz eigenvalue problem.

## Usage

Run the following code before each use.

```@example gem
using TwoBody
```

Define the Hamiltonian. This example uses the non-relativistic hydrogen atom
in atomic units,

```math
\hat{H} = -\frac{1}{2}\nabla^2 - \frac{1}{r}.
```

```@example gem
H = Hamiltonian(
  Kinetic(hbar = 1, m = 1),
  Coulomb(coefficient = -1),
)
nothing # hide
```

Define a basis set. Here, 20 Gaussian ranges are placed in a geometric
progression from 0.1 to 10.0.

```@example gem
BS = GeometricBasisSet(GaussianBasis, 0.1, 10.0, 20)
nothing # hide
```

`GaussianBasis` includes the spherical harmonic and is radially normalized.
For a central Hamiltonian, all primitives in one calculation should use the
same `l` and `m`. For example, a ``p``-wave basis can be made from the
exponents returned by `geometric`:

```@example gem
exponents = TwoBody.geometric(0.1, 10.0, 20)
p_wave_basis = BasisSet((GaussianBasis(a, 1, 0) for a in exponents)...)
nothing # hide
```

Solve the generalized eigenvalue problem with the Rayleigh–Ritz solver.

```@example gem
result = solve(H, BS)
result.E[1]
```

The position- and momentum-space definitions and unit conventions are given
in the `GaussianBasis`, `φp`, and `ψp` docstrings in the API reference below.

## Example of Hydrogen Atom

For highly excited states, Hiyama and Kamimura use the equivalent real
cosine/sine form of complex-range Gaussians,

```math
r^l e^{-\nu_n r^2}\cos(\omega\nu_n r^2),\qquad
r^l e^{-\nu_n r^2}\sin(\omega\nu_n r^2).
```

The following reproduces the parameters in Appendix A.6.2 and Table A5 of
[Hiyama and Kamimura
(2018)](https://link.springer.com/article/10.1007/s11467-018-0828-5):
80 geometrically spaced ranges from 0.015 to 2000 a.u., with ``\omega=1.5``.
There are 160 real basis functions because both the cosine and sine member are
included at every range.

```@example gem
hiyama_basis = ComplexGaussianBasisSet(0.015, 2000.0, 80; ω=1.5)
hiyama = solve(H, hiyama_basis)

levels = [1, 3, 10, 26, 30, 36, 40]
published = [
  -4.999999845e-1,
  -5.555555494e-2,
  -4.999999983e-3,
  -7.396449686e-4,
  -5.555555323e-4,
  -3.856834714e-4,
  -3.106429115e-4,
]
maximum(abs.(hiyama.E[levels] .- published)) < 5e-11
```

| principal level ``n`` | TwoBody.jl (hartree) | Hiyama Table A5 (hartree) |
|:--:|--:|--:|
| 1  | -0.499999984540 | -0.4999999845 |
| 3  | -0.0555555549404 | -0.05555555494 |
| 10 | -0.00499999998265 | -0.004999999983 |
| 26 | -0.000739644964372 | -0.0007396449686 |
| 30 | -0.000555555534152 | -0.0005555555323 |
| 36 | -0.000385683465190 | -0.0003856834714 |
| 40 | -0.000310642943837 | -0.0003106429115 |

The maximum absolute difference from the published values is
``4.1\times10^{-11}`` hartree (at least seven significant digits agree for
every listed level). The residual last-digit differences are expected for the
ill-conditioned 160-function generalized eigenproblem. The loss of accuracy
relative to the exact Coulomb value at the highest states is the same
finite-basis behavior shown in the reference.

## Example of Charmonium

The following calculation uses one Gaussian with ``\nu=0.2443`` and the SGA
parameters of [Arifi et al.
(2024)](https://arxiv.org/abs/2401.07933). Natural units are used, so energies
and masses are in GeV and lengths are in GeV``^{-1}``.

```@example gem
ν = 0.2443
masses = (1.6324, 1.6324)
a = -0.4235
b = 0.1655
αs = 0.4410
Λ = 0.9639
spin = -3/4

reduced_mass = inv(inv(masses[1]) + inv(masses[2]))
λ = Λ * sqrt(reduced_mass)
hyperfine = 32π * αs * (λ / sqrt(π))^3 /
            (9 * masses[1] * masses[2]) * spin

arifi_hamiltonian = Hamiltonian(
  RestEnergy(m=masses[1]), RelativisticKinetic(m=masses[1]),
  RestEnergy(m=masses[2]), RelativisticKinetic(m=masses[2]),
  Constant(constant=a),
  Linear(coefficient=b),
  Coulomb(coefficient=-4αs/3),
  Gaussian(coefficient=hyperfine, exponent=λ^2),
)

eta_c = solve(arifi_hamiltonian, GaussianBasis(ν))
round(eta_c.E[1] * 1000; digits=6)
```

| calculation | ``\eta_c`` mass (MeV) |
|:--|--:|
| TwoBody.jl with the parameters above | 3013.183414 |
| Arifi et al. SGA, Table 2 | 3012 |

The 1.18 MeV difference from the paper table comes from using the rounded
parameters and rounded variational exponent shown above, rather than from the
matrix-element implementation. The SGA and GEM parameter sets should not be
mixed.

## API reference

```@docs; canonical=false
TwoBody.GaussianBasis
TwoBody.ComplexGaussianBasis
TwoBody.ComplexGaussianBasisSet
TwoBody.φp
TwoBody.ψp
```
