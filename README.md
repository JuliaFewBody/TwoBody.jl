# TwoBody.jl [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliafewbody.github.io/TwoBody.jl/stable) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliafewbody.github.io/TwoBody.jl/dev) [![Build Status](https://github.com/JuliaFewBody/TwoBody.jl/workflows/CI/badge.svg)](https://github.com/JuliaFewBody/TwoBody.jl/actions)

TwoBody.jl: a Julia package for quantum mechanical two-body problems

## Quick Start

Install TwoBody.jl from the Julia REPL or a notebook:

```julia
import Pkg; Pkg.add(url="https://github.com/JuliaFewBody/TwoBody.jl.git")
```

Load the package and solve the hydrogen ground state:

```julia
using TwoBody

H = Hamiltonian(Kinetic(hbar=1, m=1), Coulomb(coefficient=-1))
BS = BasisSet(SimpleGaussianBasis(13.00773), SimpleGaussianBasis(1.962079))
solve(H, BS; info=0).E[1]
```

## Documentation

- Home: https://juliafewbody.github.io/TwoBody.jl
- Developer Guide: https://juliafewbody.github.io/TwoBody.jl/dev/developer
- API Reference: https://juliafewbody.github.io/TwoBody.jl/dev/API
