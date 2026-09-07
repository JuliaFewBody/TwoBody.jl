```@meta
CurrentModule = TwoBody
```

# Faddeev-Yakubovsky Method

The Faddeev-Yakubovsky method decomposes a few-body wavefunction into components
associated with interacting pairs and cluster partitions. This implementation
is a two-body proof of concept for FewBody.jl: there is only one pair, so its
component is the entire wavefunction. It finds the lowest bound state in the
supplied basis sector; scattering and coupled three- or four-body equations
are outside its scope.

## Theory

For ``H=T+V``, the Schrödinger equation can be written as

```math
(E-T)\psi=V\psi,\qquad \psi=G_0(E)V\psi,\qquad G_0(E)=(E-T)^{-1}.
```

This is the single-pair version of the component definition in
[Lazauskas and Carbonell (2019), Eq. (21)](https://arxiv.org/html/1908.04861v1).
Expanding ``\psi=\sum_i c_i\phi_i`` with the same matrices as the
[Rayleigh-Ritz Method](@ref Rayleigh-Ritz-Method) gives

```math
\boldsymbol{c}=(E\boldsymbol{S}-\boldsymbol{T})^{-1}
\boldsymbol{V}\boldsymbol{c}.
```

For energies below the free spectrum, ``T-ES`` is positive definite.
Factor it as ``T-ES=LL^{\mathsf T}`` and solve the equivalent symmetric problem

```math
K(E)y=\lambda(E)y,\qquad
K(E)=-L^{-1}VL^{-\mathsf T},\qquad c=L^{-\mathsf T}y.
```

The ground-state energy is found by bisection where the largest algebraic
eigenvalue satisfies ``\lambda(E)=1``. The coefficients are normalized to
``c^{\mathsf T}Sc=1``. In a fixed basis this reproduces the lowest Rayleigh-Ritz
energy, up to the search tolerance. All potential terms belong to the single
pair interaction ``V``; kinetic terms, including rest energies, form ``T``.

## Usage

Use the hydrogen Hamiltonian and four Gaussian functions from the
Rayleigh-Ritz example:

```@example fy
using TwoBody

# Hamiltonian
H = Hamiltonian(
  Kinetic(hbar=1, m=1),
  Coulomb(coefficient=-1),
)

# basis set
BS = BasisSet(
  SimpleGaussianBasis(13.00773),
  SimpleGaussianBasis(1.962079),
  SimpleGaussianBasis(0.444529),
  SimpleGaussianBasis(0.1219492),
)

# solve
method = FaddeevYakubovsky(BS; Eₘᵢₙ=-1.0, Eₘₐₓ=0.0)
res = solve(H, method; info=0)
res.E[1]
```

The result is approximately ``-0.499278\,E_{\mathrm h}``, compared with the exact
hydrogen ground-state energy ``-0.5\,E_{\mathrm h}``. Evaluate the normalized
wavefunction with `TwoBody.ψ(res, r)`. A `GeometricBasisSet` or single `Basis`
can also be passed to `FaddeevYakubovsky`.

The energy interval must bracket the lowest state and lie below the lowest
free eigenvalue of ``(T,S)``. If the interval misses that state, or no such
state exists, `solve` raises an error. Adjust the interval for the energy
units and Hamiltonian in use. The default tolerance is `1e-10`, with at most
100 bisection steps. This finite-basis two-body Coulomb example does not
implement the long-range treatment needed in many-body Coulomb scattering.

## API reference

```@docs; canonical=false
FaddeevYakubovsky
solve(hamiltonian::Hamiltonian, method::FaddeevYakubovsky)
```
