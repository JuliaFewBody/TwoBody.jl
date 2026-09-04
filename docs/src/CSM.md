```@meta
CurrentModule = TwoBody
```

# Complex Scaling Method

Complex scaling rotates the radial coordinate by a real angle ``θ``:

```math
r \rightarrow r e^{iθ}, \qquad
H^θ = e^{-2iθ}T + V(r e^{iθ}).
```

The bound-state spectrum is unchanged, while the continuum rotates by ``2θ``.
Resonance poles can therefore be separated from the continuum. See Myo and Kato,
[Prog. Theor. Exp. Phys. 2020, 12A101](https://doi.org/10.1093/ptep/ptaa101).

## Usage

```@example csm
using TwoBody

H = Hamiltonian(Kinetic(hbar=1, m=1), Coulomb(coefficient=-1))
basisset = GeometricBasisSet(GaussianBasis, 0.1, 20.0, 12)
method = ComplexScalingMethod(basisset; θ=0.2)
result = solve(H, method)
result.E[1]
```

`Tabulated` potentials are not supported because their analytic continuation is
undefined.

## API reference

```@docs; canonical=false
ComplexScalingMethod
ResultComplexScaling
solve(hamiltonian::Hamiltonian, method::ComplexScalingMethod)
```
