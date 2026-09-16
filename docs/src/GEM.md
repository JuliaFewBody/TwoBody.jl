```@meta
CurrentModule = TwoBody
```

!!! warning
    This method was implemented by [GPT-5.6 Sol High](https://github.com/JuliaFewBody/TwoBody.jl/pull/42) based on the original code developed for [arXiv:2401.07933](https://arxiv.org/abs/2401.07933). A detailed review by Shuhei Ohno has not yet been completed.

# [Gaussian Expansion Method](@id GEM)

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
BS = BasisSet(
  GaussianBasis(a=13.00773, l=0, m=0),
  GaussianBasis(a=1.962079, l=0, m=0),
  GaussianBasis(a=0.444529, l=0, m=0),
  GaussianBasis(a=0.1219492, l=0, m=0),
)
nothing # hide
```

Solve the generalized eigenvalue problem with the Rayleigh–Ritz solver.

```@example gem
result = solve(H, BS)
result.E[1]
```

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

result = solve(H, BS)

for i in 1:7
  println(result.E[i])
end
```

| ``n`` |     TwoBody.jl |      Reference |          Exact |
| ----- | -------------- | -------------- | -------------- |
|     1 | ``-0.499~982`` | ``-0.499~982`` | ``-0.500~000`` |
|     2 | ``-0.124~998`` | ``-0.124~998`` | ``-0.125~000`` |
|     3 | ``-0.055~555`` | ``-0.055~555`` | ``-0.055~556`` |
|     4 | ``-0.031~249`` | ``-0.031~249`` | ``-0.031~250`` |
|     5 | ``-0.019~998`` | ``-0.019~998`` | ``-0.020~000`` |
|     6 | ``-0.013~883`` | ``-0.013~883`` | ``-0.013~889`` |
|     7 | ``-0.010~203`` | ``-0.010~203`` | ``-0.010~204`` |

The TwoBody.jl results agree with all seven values in Table VII at the six
decimal places reported there.

## Example of ``\Lambda_c(1/2^+)``

Parameters follow [Kim, Hiyama, Oka, and Suzuki (2020)](https://doi.org/10.1103/PhysRevD.102.014004). The charm quark and scalar diquark are treated as a two-body system, with
``\mu=M_{qq}M_c/(M_{qq}+M_c)`` and

```math
\hat H = \frac{\boldsymbol p^2}{2\mu} + M_{qq} + M_c
- \frac{\alpha}{r} + \lambda r + C.
```

```@example lambda_c
using TwoBody

Mqq = 0.725
Mc = 1.750
μ = inv(inv(Mqq) + inv(Mc))
α = 0.06 / μ

H = Hamiltonian(
  RestEnergy(m=Mqq),
  RestEnergy(m=Mc),
  Kinetic(hbar=1, m=μ),
  Coulomb(coefficient=-α),
  Linear(coefficient=0.165),
  Constant(constant=-0.83116597),
)

BS = GeometricBasisSet(GaussianBasis, 0.01, 9.0, 40)

round(1000 * solve(H, BS; info=0).E[1]; digits=3)
```

This result is in good agreement with the value of **2286 MeV** reported in Table II of the reference because the parameters were provided by the authors. Natural units (``\hbar=c=1``) are used: energies and masses are in GeV, and Gaussian ranges are in GeV``^{-1}``. The displayed ground-state masses are converted to MeV.

## Example of ``\eta_c(1S)``

Parameters follow [Meng, Wang, and Oka (2024)](https://doi.org/10.48550/arXiv.2404.01238). The AL1 Hamiltonian is

```math
\hat H = \frac{\boldsymbol p^2}{2\mu} + m_1 + m_2
- \frac{\kappa}{r} + \lambda r - \Lambda
+ \frac{2\pi\kappa'}{3m_1m_2}
  \frac{e^{-r^2/r_0^2}}{\pi^{3/2}r_0^3}
  \boldsymbol\sigma_1\cdot\boldsymbol\sigma_2.
```

Here ``r_0=A(2\mu)^{-B}`` and
``\langle\boldsymbol\sigma_1\cdot\boldsymbol\sigma_2\rangle=-3``
for the spin-singlet state.

```@example meng_charmonium
using TwoBody

m₁ = 1.836
m₂ = 1.836
κ = 0.5069
κ′ = 1.8609
spin = -3
μ = inv(inv(m₁) + inv(m₂))
r₀ = 1.6553 * (2m₁ * m₂ / (m₁ + m₂))^(-0.2204)

H = Hamiltonian(
  RestEnergy(m=m₁),
  RestEnergy(m=m₂),
  Kinetic(hbar=1, m=μ),
  Coulomb(coefficient=-κ),
  Linear(coefficient=0.1653),
  Constant(constant=-0.8321),
  Gaussian(
    coefficient=2π * κ′ * spin / (3m₁ * m₂ * (sqrt(π) * r₀)^3),
    exponent=inv(r₀^2),
  ),
)

BS = GeometricBasisSet(GaussianBasis, 0.1, 80.0, 20)

round(1000 * solve(H, BS; info=0).E[1]; digits=3)
```

This result is in good agreement with the value of **3005 MeV** reported in Table 6 of the reference.

### Example of ``J/\psi(1S)``

Parameters follow [Arifi, Happ, Ohno, and Oka (2024)](https://arxiv.org/abs/2401.07933). The semirelativistic Hamiltonian is

```math
\hat H = \sqrt{m_1^2+\boldsymbol p^2}+\sqrt{m_2^2+\boldsymbol p^2}
+ a + br - \frac{4\alpha_s}{3r}
+ \frac{32\pi\alpha_s}{9m_1m_2}
  \left(\frac{\lambda}{\sqrt{\pi}}\right)^3 e^{-\lambda^2r^2}
  \boldsymbol S_1\cdot\boldsymbol S_2.
```

Here ``\lambda=\Lambda\sqrt{\mu}`` and
``\langle\boldsymbol S_1\cdot\boldsymbol S_2\rangle=-3/4``.
`RelativisticKinetic` represents ``\sqrt{m^2+\boldsymbol p^2}-m``,
so each constituent also needs a `RestEnergy` term.

```@example arifi_charmonium
using TwoBody

m₁ = 1.5147
m₂ = 1.5147
αs = 0.2850
spin = -3/4
μ = inv(inv(m₁) + inv(m₂))
λ = 1.4376 * sqrt(μ)
a = -0.1895
b = 0.0924

H = Hamiltonian(
  RestEnergy(m=m₁),
  RestEnergy(m=m₂),
  RelativisticKinetic(m=m₁),
  RelativisticKinetic(m=m₂),
  Constant(constant=a),
  Linear(coefficient=b),
  Coulomb(coefficient=-4αs/3),
  Gaussian(
    coefficient=32π * αs * (λ / sqrt(π))^3 * spin / (9m₁ * m₂),
    exponent=λ^2,
  ),
)

BS = GeometricBasisSet(GaussianBasis, 0.358, 2.720, 10)

round(1000 * solve(H, BS; info=0).E[1]; digits=3)
```

This result is in good agreement with the value of **3019 MeV** reported in Table 2 of the reference because the parameters were provided by the authors. Since the parameters in Table I are rounded, using the same values does not reproduce the result exactly.

## API reference

```@docs; canonical=false
TwoBody.GaussianBasis
TwoBody.ComplexGaussianBasis
TwoBody.ComplexGaussianBasisSet
TwoBody.φp
TwoBody.ψp
```

