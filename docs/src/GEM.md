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

GEM uses the same Hamiltonian construction and `solve` interface as the
[Rayleigh–Ritz method](@ref "Rayleigh-Ritz Method"). Only the basis set is
changed below.

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

Define the Gaussian basis set.

```@example gem
BS = GeometricBasisSet(GaussianBasis, 0.1, 10.0, 20)
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

Appendix A.2 and Table VII of [Hiyama and Kamimura
(2018)](https://link.springer.com/article/10.1007/s11467-018-0828-5)
calculate the lowest seven ``l=0`` states of the hydrogen atom with 20
real-range Gaussians,

```math
\phi_n(r) = N_n e^{-\nu_n r^2}.
```

The Gaussian ranges are placed in a geometric progression with
``n_{\max}=20``, ``r_1=0.1`` a.u., and ``r_{20}=80`` a.u.

```@example gem
H = Hamiltonian(
  Kinetic(hbar = 1, m = 1),
  Coulomb(coefficient = -1),
)
BS = GeometricBasisSet(GaussianBasis, 0.1, 80.0, 20)
solve(H, BS)
```

| principal level ``n`` | TwoBody.jl (hartree) | Hiyama Table VII (hartree) | exact (hartree) |
|:--:|--:|--:|--:|
| 1 | ``-0.499~982`` | ``-0.499~982`` | ``-0.500~000`` |
| 2 | ``-0.124~998`` | ``-0.124~998`` | ``-0.125~000`` |
| 3 | ``-0.055~555`` | ``-0.055~555`` | ``-0.055~556`` |
| 4 | ``-0.031~249`` | ``-0.031~249`` | ``-0.031~250`` |
| 5 | ``-0.019~998`` | ``-0.019~998`` | ``-0.020~000`` |
| 6 | ``-0.013~883`` | ``-0.013~883`` | ``-0.013~889`` |
| 7 | ``-0.010~203`` | ``-0.010~203`` | ``-0.010~204`` |

The TwoBody.jl results agree with all seven values in Table VII at the six
decimal places reported there. The complex-range hydrogen calculation in
Appendix A.6.2 is a separate example for highly excited states and is not the
calculation reproduced here.

## Example of Charmonium

!!! warning
    This comparison is currently incorrect. It uses the SGA parameters and
    result from Arifi et al. rather than reproducing their GEM calculation.
    The example will be updated after the optimized GEM parameters have been
    recovered from the calculation notes.

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

H = Hamiltonian(
  RestEnergy(m=masses[1]), RelativisticKinetic(m=masses[1]),
  RestEnergy(m=masses[2]), RelativisticKinetic(m=masses[2]),
  Constant(constant=a),
  Linear(coefficient=b),
  Coulomb(coefficient=-4αs/3),
  Gaussian(coefficient=hyperfine, exponent=λ^2),
)

eta_c = solve(H, GaussianBasis(ν))
round(eta_c.E[1] * 1000; digits=6)
```

| calculation | ``\eta_c`` mass (MeV) |
|:--|--:|
| TwoBody.jl with the parameters above | 3013.183414 |
| Arifi et al. SGA, Table 2 | 3012 |

## API reference

```@docs; canonical=false
TwoBody.GaussianBasis
TwoBody.ComplexGaussianBasis
TwoBody.ComplexGaussianBasisSet
TwoBody.φp
TwoBody.ψp
```
