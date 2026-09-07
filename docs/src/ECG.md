```@meta
CurrentModule = TwoBody
```

# Explicitly Correlated Gaussians

For three-dimensional coordinate vectors collected in ``r``, an ECG basis
function is

```math
G(r) = \prod_k (a_k^T r)\exp(-r^T A r).
```

The positive-definite correlation matrix ``A`` couples the coordinates. Zero,
one, and two prefactors describe scalar, vector, and tensor functions. Matrix
elements follow [Fedorov et al. (2024)](https://doi.org/10.1007/s00601-024-01945-x).

## Usage

```@example ecg
using TwoBody

exponents = 10.0 .^ range(-3, 2, length=10)
prefactor = ([0.0, 0.0, 1.0],)
basisset = BasisSet((ECGBasis([a;;]; prefactors=prefactor) for a in exponents)...)
H = Hamiltonian(ECGKinetic([0.5;;]), ECGCoulomb(-1.0, [1.0]))
result = solve(H, basisset, info=-1)
result.E[1]
```

For multiple coordinates, each prefactor contains consecutive Cartesian
components and has length `3 * size(A, 1)`.

## API reference

```@docs; canonical=false
ExplicitlyCorrelatedGaussianBasis
ECGKinetic
ECGCoulomb
```
