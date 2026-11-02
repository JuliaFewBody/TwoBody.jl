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

The ranges are normally placed in a geometric progression. The implementation follows
the formulation reviewed by [Hiyama, Kino, and Kamimura
(2003)](https://www.sciencedirect.com/science/article/abs/pii/S0146641003900159?via%3Dihub).

## Usage

Run the following code before each use.

```@example gem
using TwoBody
```

Define the non-relativistic hydrogen Hamiltonian in atomic units,

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

Define the Gaussian basis set (the only change from the [Rayleigh–Ritz example](@ref "Rayleigh-Ritz Method")).

```@example gem
BS = BasisSet(
  GaussianBasis(a=13.00773, l=0, m=0),
  GaussianBasis(a=1.962079, l=0, m=0),
  GaussianBasis(a=0.444529, l=0, m=0),
  GaussianBasis(a=0.1219492, l=0, m=0),
)
nothing # hide
```

Solve the generalized eigenvalue problem.

```@example gem
result = solve(H, BS)
result.E[1]
```

## Example of H (p+e)

The Hamiltonian of the [hydrogen atom](https://ohno.github.io/Antique.jl/stable/HydrogenAtom/) is

```math
\hat{H} =
- \frac{\hbar^2}{2m_\mathrm{p}} {\nabla}_\mathrm{p}^2
- \frac{\hbar^2}{2m_\mathrm{e}} {\nabla}_\mathrm{e}^2
- \frac{e^2}{4\pi\varepsilon_0 r}.
```

Separating the center-of-mass motion gives the relative-motion Hamiltonian
with ``\boldsymbol r=\boldsymbol r_\mathrm{e}-\boldsymbol r_\mathrm{p}``
and reduced mass ``1/\mu=1/m_\mathrm{p}+1/m_\mathrm{e}``,

```math
\hat{H} =
- \frac{\hbar^2}{2\mu} \nabla^2
- \frac{e^2}{4\pi\varepsilon_0 r}.
```

Nondimensionalizing the Hamiltonian using atomic units and setting
``\mu=m_\mathrm{e}=1`` (the infinite-proton-mass approximation) gives

```math
\hat{H} =
-\frac{1}{2}\nabla^2
- \frac{1}{r}.
```

With ``n_{\max}=20`` real-range Gaussians with geometrically spaced ranges from ``r_1=0.1`` to ``r_{20}=80`` a.u., the calculation below reproduces the seven lowest ``l=0`` energies in Table VII of [Hiyama and Kamimura (2018)](https://link.springer.com/article/10.1007/s11467-018-0828-5).

```@example gem
using TwoBody

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

## Example of H₂ (H+H)

We use the Morse potential used by [A. K. Roy, *Results in Physics* **3**, 103–108 (2013)](https://doi.org/10.1016/j.rinp.2013.06.001). For the ``n=\ell=0`` state, the relative-motion Hamiltonian is

```math
\hat H = -\frac{\hbar^2}{2\mu}\nabla^2
+ D_e\left[e^{-2\alpha(r-r_e)/r_e}-2e^{-\alpha(r-r_e)/r_e}\right].
```

The parameters follow Table I, and ``K=\hbar^2/(2\mu)`` is calculated using the conversion factors in Sec. III. Here ``\alpha`` is dimensionless. Distances are in Å, and energies are in eV relative to the dissociation threshold. The calculated binding energy agrees well with ``4.47601313~\mathrm{eV}`` in Table II.

```@example roy_hydrogen
using TwoBody

Eₕ = 27.21138505       # eV per hartree, as used by Roy (2013)
a₀ = 0.52917721092     # Å per bohr
mₑ = 5.48577990946e-4  # u per electron mass
Dₑ = 4.7446            # eV
rₑ = 0.7416            # Å
μ = 0.50391            # u
α = 1.440558
K = Eₕ * a₀^2 * mₑ / (2μ) # eV Å²

H = Hamiltonian(
  Laplacian(coefficient=-K),
  Exponential(coefficient=Dₑ * exp(2α), exponent=2α / rₑ),
  Exponential(coefficient=-2Dₑ * exp(α), exponent=α / rₑ),
)

BS = GeometricBasisSet(GaussianBasis, 0.05, 4.0, 30)

result = solve(H, BS; info=0)
result.E[1]
```

## Example of He₂ (He+He)

We use the soft-core Gaussian potential employed by [M. Gattobigio, A. Kievsky, and M. Viviani, _Phys. Rev. A_ **84**, 052503 (2011)](https://doi.org/10.1103/PhysRevA.84.052503) to approximate the low-energy properties of the LM2M2 interaction. For two ground-state ``{}^4\mathrm{He}`` atoms in the ``\ell=0`` channel, the relative-motion Hamiltonian with ``\mu=m_{{}^4\mathrm{He}}/2`` is

```math
\frac{\hat H}{k_B} = -K\nabla^2 + V_0 e^{-r^2/R^2},
\qquad K = \frac{\hbar^2}{2\mu k_B}.
```

With the parameters of Eq. (1) and ``K=43.281307~\mathrm{K}\,a_0^2`` in Sec. II, the calculated energy agrees with ``E/k_B=-1.296~\mathrm{mK}`` reported there. Distances are in bohr, and energies divided by ``k_B`` are in K relative to the two-atom threshold. The output is in mK.

```@example helium2
using TwoBody

K = 43.281307
V₀ = -1.227
R = 10.03

H = Hamiltonian(
  Laplacian(coefficient=-K),
  Gaussian(coefficient=V₀, exponent=inv(R^2)),
)

BS = GeometricBasisSet(GaussianBasis, 1.0, 1000.0, 30)

result = solve(H, BS)
1000result.E[1] # E/k_B in mK
```

## Example of d (p+n)

We use the Minnesota ``NN`` potential introduced by [D. R. Thompson, M. Lemere, and Y. C. Tang, _Nucl. Phys. A_ **286**, 53 (1977)](https://doi.org/10.1016/0375-9474(77)90007-0). In the ``{}^3S_1`` state, the potential reduces to ``V_R + V_t``, with ``P^\sigma=P^r=+1``. For proton and neutron masses ``m_\mathrm{p}`` and ``m_\mathrm{n}``, the relative-motion Hamiltonian with ``1/\mu = 1/m_\mathrm{p} + 1/m_\mathrm{n}`` is

```math
\hat{H}
= -\frac{\hbar^2}{2\mu}\nabla^2
+ V_{0R}e^{-\kappa_R r^2}
- V_{0t}e^{-\kappa_t r^2}.
```

[Weiss-Attia, Schäfer, and Bazak (2024)](https://arxiv.org/abs/2402.04817) use the kinetic coefficient ``\frac{\hbar^2}{2\mu} = \frac{(\hbar c)^2}{m} \simeq 41.471~\mathrm{MeV\,fm^2}``, with ``m = m_\mathrm{p}c^2 = m_\mathrm{n}c^2`` implicitly assumed for equal proton and neutron masses. They report a free-space deuteron energy of ``E_d \simeq -2.202~\mathrm{MeV}`` in Sec. V.1. Distances are in fm, and energies are in MeV.

```@example deuteron
using TwoBody

ℏ²c²m⁻¹ = 41.471
V₀ᵣ = 200.0
V₀ₜ = 178.0
κᵣ = 1.487
κₜ = 0.639

H = Hamiltonian(
  Laplacian(coefficient = -ℏ²c²m⁻¹),
  Gaussian(coefficient = V₀ᵣ, exponent = κᵣ),
  Gaussian(coefficient = -V₀ₜ, exponent = κₜ),
)

BS = GeometricBasisSet(GaussianBasis, 0.1, 40.0, 30)

result = solve(H, BS)
result.E[1]
```

## Example of ⁶Li (α+d)

We use the Gaussian ``\alpha d`` potential of [E. M. Tursunov, S. A. Turakulov, and P. Descouvemont, _Phys. At. Nucl._ **78**, 193 (2015)](https://doi.org/10.1134/S1063778815010196), treating both clusters as inert. In the ``{}^3S_1`` channel, the relative-motion Hamiltonian with ``1/\mu = 1/m_\alpha + 1/m_d`` is

```math
\hat H = -\frac{\hbar^2}{2\mu}\nabla^2
- V_0 e^{-\kappa r^2}
+ \frac{2e^2}{4\pi\varepsilon_0 r}.
```

The parameters follow the ``V_T`` model in Tables I–III of [Grassi, Mangano, Marcucci, and Pisanti (2017)](https://arxiv.org/abs/1707.09232), with ``K=\hbar^2/(2\mu)`` and a reported binding energy ``B=1.475~\mathrm{MeV}``. The potential supports one Pauli-forbidden ``S``-wave bound state (the lowest, nodeless ``{}^3S_1`` state), so the physical ``1^+`` ground state is the second eigenstate. Distances are in fm, and energies are in MeV relative to the ``\alpha+d`` threshold, with ``E=-B``.

```@example lithium6
using TwoBody

K = 15.550725
e² = 1.439975 # e²/(4πε₀), in MeV fm
V₀ = 92.44
κ = 0.25

H = Hamiltonian(
  Laplacian(coefficient = -K),
  Gaussian(coefficient = -V₀, exponent = κ),
  Coulomb(coefficient = 2e²),
)

BS = GeometricBasisSet(GaussianBasis, 0.1, 40.0, 30)

result = solve(H, BS)
result.E[2] # Skip the Pauli-forbidden state.
```

## Example of ⁷Li (α+t)

We use the Gaussian cluster potential of [S. B. Dubovichenko, _Phys. At. Nucl._ **73**, 1526 (2010)](https://doi.org/10.1134/S1063778810090073). For inert ``\alpha`` and triton clusters in the ``{}^2P_{3/2}`` channel, the relative-motion Hamiltonian with ``1/\mu = 1/m_\alpha + 1/m_t`` is

```math
\begin{aligned}
\hat H &= -\frac{\hbar^2}{2\mu}\nabla^2 + V_0 e^{-\alpha r^2} + V_C(r),\\
V_C(r) &= \frac{2e^2}{4\pi\varepsilon_0}
\begin{cases}
\displaystyle \frac{3-(r/R_C)^2}{2R_C}, & r\le R_C,\\[6pt]
\displaystyle \frac{1}{r}, & r>R_C.
\end{cases}
\end{aligned}
```

With the parameters and cluster masses in Tables 2–3 and ``K=\hbar^2/(2\mu)=20.7343(1/A_\alpha+1/A_t)~\mathrm{MeV\,fm^2}``, where ``A_i=m_i/m_{\mathrm u}``, the calculated energy agrees well with ``-2.467~\mathrm{MeV}`` in Table 2. The potential supports one Pauli-forbidden ``P``-wave bound state (the lowest, nodeless ``{}^2P_{3/2}`` state), so the physical ``3/2^-`` ground state is the second eigenstate in an ``l=1`` basis. Distances are in fm, and energies are in MeV relative to the ``\alpha+t`` threshold.

```@example lithium7
using TwoBody

Aα = 4.001506179127
At = 3.0155007134
K = 20.7343 * (1/Aα + 1/At)
e² = 1.439975 # e²/(4πε₀), in MeV fm
V₀ = -83.616808
α = 0.15747
R_C = 3.095
V_C(r) = 2e² * (r <= R_C ? (3 - (r/R_C)^2)/(2R_C) : 1/r)

H = Hamiltonian(
  Laplacian(coefficient = -K),
  Gaussian(coefficient = V₀, exponent = α),
  Custom(V_C),
)

BS = GeometricBasisSet(a -> GaussianBasis(a; l=1), 0.1, 40.0, 25)

result = solve(H, BS)
result.E[2] # Skip the Pauli-forbidden state.
```

## Example of ``^5_\Lambda\mathrm{He}`` (α+Λ)

We use the Isle ``\alpha\Lambda`` potential introduced by Kurihara et al., with parameters from Eq. (3.2) of [I. Kumagai-Fuse and Y. Akaishi, _Prog. Theor. Phys._ **92**, 815 (1994)](https://doi.org/10.1143/ptp/92.4.815). For an inert ``\alpha`` core and a ``\Lambda`` hyperon in the ``l=0``, ``J^\pi=1/2^+`` state, the relative-motion Hamiltonian with ``1/\mu = 1/m_\alpha + 1/m_\Lambda`` is

```math
\hat H = -\frac{\hbar^2}{2\mu}\nabla^2
+ V_{\mathrm r}e^{-(r/\beta_{\mathrm r})^2}
- V_{\mathrm a}e^{-(r/\beta_{\mathrm a})^2}.
```

The calculated binding energy is about ``24~\mathrm{keV}`` below ``3.12~\mathrm{MeV}`` quoted in Sec. 3.1. Distances are in fm, and energies are in MeV relative to the ``\alpha+\Lambda`` threshold.

```@example hyperhelium5
using TwoBody

ℏc = 197.3269804
mα = 3727.379
mΛ = 1115.683
μ = inv(inv(mα) + inv(mΛ)) # μc², in MeV
Vᵣ = 450.4
Vₐ = 404.9
βᵣ = 1.25
βₐ = 1.41

H = Hamiltonian(
  Kinetic(hbar = ℏc, m = μ),
  Gaussian(coefficient = Vᵣ, exponent = inv(βᵣ^2)),
  Gaussian(coefficient = -Vₐ, exponent = inv(βₐ^2)),
)

BS = GeometricBasisSet(GaussianBasis, 0.1, 40.0, 30)

result = solve(H, BS)
result.E[1]
```

## Example of ``\Lambda_c`` (``ud+c``)

We use the quark–diquark model of [Kim, Hiyama, Oka, and Suzuki (2020)](https://doi.org/10.1103/PhysRevD.102.014004) for ``\Lambda_c(1/2^+)``. The charm quark and scalar diquark are treated as a two-body system, with reduced mass
``\mu=M_{qq}M_c/(M_{qq}+M_c)`` and Hamiltonian

```math
\hat H = \frac{\boldsymbol p^2}{2\mu} + M_{qq} + M_c
- \frac{\alpha}{r} + \lambda r + C.
```

Using parameters provided by the authors, the calculation below reproduces **2286 MeV** in Table II. These three quark-model examples use natural units (``\hbar=c=1``): energies and masses are in GeV, and Gaussian ranges are in GeV``^{-1}``. Output masses are in MeV.

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

## Example of ``\eta_c`` (``c+\bar{\mathrm{c}}``)

We use the AL1 quark model employed by [Meng, Wang, and Oka (2024)](https://doi.org/10.48550/arXiv.2404.01238) for ``\eta_c(1S)``. The Hamiltonian is

```math
\hat H = \frac{\boldsymbol p^2}{2\mu} + m_1 + m_2
- \frac{\kappa}{r} + \lambda r - \Lambda
+ \frac{2\pi\kappa'}{3m_1m_2}
  \frac{e^{-r^2/r_0^2}}{\pi^{3/2}r_0^3}
  \boldsymbol\sigma_1\cdot\boldsymbol\sigma_2.
```

Here ``r_0=A(2\mu)^{-B}`` and
``\langle\boldsymbol\sigma_1\cdot\boldsymbol\sigma_2\rangle=-3``
for the spin-singlet state. The calculation below reproduces **3005 MeV** in Table 6 to MeV precision.

```@example meng_charmonium
using TwoBody

m₁ = 1.836
m₂ = 1.836
κ = 0.5069
κ′ = 1.8609
σ₁σ₂ = -3
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
    coefficient=2π * κ′ * σ₁σ₂ / (3m₁ * m₂ * (sqrt(π) * r₀)^3),
    exponent=inv(r₀^2),
  ),
)

BS = GeometricBasisSet(GaussianBasis, 0.1, 80.0, 20)

round(1000 * solve(H, BS; info=0).E[1]; digits=3)
```

## Example of ``J/\psi`` (``c+\bar{\mathrm{c}}``)

We use the semirelativistic quark model of [Arifi, Happ, Ohno, and Oka (2024)](https://arxiv.org/abs/2401.07933) for ``J/\psi(1S)``. The Hamiltonian is

```math
\hat H = \sqrt{m_1^2+\boldsymbol p^2}+\sqrt{m_2^2+\boldsymbol p^2}
+ a + br - \frac{4\alpha_s}{3r}
+ \frac{32\pi\alpha_s}{9m_1m_2}
  \left(\frac{\lambda}{\sqrt{\pi}}\right)^3 e^{-\lambda^2r^2}
  \boldsymbol S_1\cdot\boldsymbol S_2.
```

Here ``\lambda=\Lambda\sqrt{\mu}`` and
``\langle\boldsymbol S_1\cdot\boldsymbol S_2\rangle=1/4``
for the spin-triplet state.
`RelativisticKinetic` represents ``\sqrt{m^2+\boldsymbol p^2}-m``,
so each constituent also needs a `RestEnergy` term.
The Gaussian range was extended for MeV precision. With parameters provided by the authors, the calculation below reproduces **3059 MeV** in Table 2. The rounded values in Table I do not reproduce it exactly.

```@example arifi_charmonium
using TwoBody

m₁ = 1.5147
m₂ = 1.5147
αs = 0.2850
S₁S₂ = 1/4
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
    coefficient=32π * αs * (λ / sqrt(π))^3 * S₁S₂ / (9m₁ * m₂),
    exponent=λ^2,
  ),
)

BS = GeometricBasisSet(GaussianBasis, 0.05, 20.0, 30)

round(1000 * solve(H, BS; info=0).E[1]; digits=3)
```

## API reference

```@docs; canonical=false
TwoBody.geometric
TwoBody.GaussianBasis
TwoBody.φp
TwoBody.ψp
```

