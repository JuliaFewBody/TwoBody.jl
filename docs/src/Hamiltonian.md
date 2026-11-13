```@meta
CurrentModule = TwoBody
```

# [Hamiltonian](@id Hamiltonian)

## Shared physical Hamiltonians

Use [FewBodyHamiltonians.jl](https://github.com/JuliaFewBody/FewBodyHamiltonians.jl)
to construct the same particle-indexed physical Hamiltonian for different solvers.
TwoBody accepts its two-particle Hamiltonians directly and lowers them to its
existing relative-coordinate operators. Its own constructors below remain supported.

```@example shared
import FewBodyHamiltonians as FBH
import TwoBody

H = FBH.Kinetic([1.0, 1.0]) + FBH.Coulomb([1, -1])
basis = TwoBody.GeometricBasisSet(TwoBody.GaussianBasis, 0.1, 80.0, 20)
result = TwoBody.solve(H, basis; info=0)
result.E[1] # positronium: approximately -0.25 Ha
```

The two masses give the reduced mass ``\mu = (1/m_1 + 1/m_2)^{-1}``.
Shared terms use ``\hbar=1``; unit labels convert nothing. In particular,
`:nuclear` masses are ``mc^2/(\hbar c)^2`` in MeV⁻¹ fm⁻², as documented by
FewBodyHamiltonians. TwoBody does not multiply by ``\hbar c`` again.

Supported shared potentials are `Coulomb`, `Gaussian` (including products),
two-particle `ManyBodyGaussian`, `Oscillator`, and `Custom`. Pairs must be `(1,2)`
or `(2,1)`. `Tensor`, `SpinOrbit`, legacy individual-particle kinetic operators,
and systems with other than two masses fail with an `ArgumentError`.
`Coulomb(charges)` requires two charges and atomic units; use explicit pair
coefficients in other unit systems. Existing backend restrictions still apply
after conversion, such as the Free Complement method's supported potential forms.

`TwoBody.Hamiltonian(H)` explicitly converts a shared value, including a
potential-only sum for use as `perturbation`. Shared Hamiltonians also work with
`optimize`, `FC`, `local_energy`, `matrix`, and `element`.

```@example shared
perturbation = FBH.Hamiltonian(FBH.Gaussian(1, 2; coefficient=0.01, exponent=1))
TwoBody.solve(H, basis; perturbation, info=1).expectation[:perturbation][1]
```

Shared `Custom` parameters `rtol`, `atol`, and `maxevals` are solver hints.
TwoBody evaluates the same radial function and coefficient using each backend's
existing quadrature or discretization settings; those hints do not override them.

```@docs; canonical=false
TwoBody.Hamiltonian
```

## Operators

```@docs; canonical=false
TwoBody.Kinetic
TwoBody.RestEnergy
TwoBody.RelativisticCorrection
TwoBody.RelativisticKinetic
TwoBody.Constant
TwoBody.Linear
TwoBody.Coulomb
TwoBody.PowerLaw
TwoBody.Gaussian
TwoBody.Exponential
TwoBody.Yukawa
TwoBody.Delta
TwoBody.Custom
TwoBody.Tabulated
```
