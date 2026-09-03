# TwoBody.jl

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oc8hfyS4gaCxNvaY00rUiiDmGMPKpfxC?usp=sharing)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliafewbody.github.io/TwoBody.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliafewbody.github.io/TwoBody.jl/dev/)
[![Citation](https://img.shields.io/badge/citation-BibTeX-778899)](CITATION.bib)
[![License](https://img.shields.io/github/license/JuliaFewBody/TwoBody.jl)](LICENSE)
[![Build Status](https://github.com/JuliaFewBody/TwoBody.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaFewBody/TwoBody.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaFewBody/TwoBody.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaFewBody/TwoBody.jl)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-contributor's%20guide-blueviolet)](https://github.com/SciML/ColPrac)

[TwoBody.jl](https://github.com/JuliaFewBody/TwoBody.jl) is a Julia package for solving quantum-mechanical two-body problems. It provides a flexible framework for constructing two-body Hamiltonians and solving the corresponding Schrödinger equation using a variety of numerical methods. It covers approaches ranging from basis-set and grid methods to tensor-network, stochastic, and neural-network methods. Beyond serving as a PoC ot for [FewBody.jl](https://github.com/JuliaFewBody/FewBody.jl), TwoBody.jl is designed to support practical, cross-scale calculations of quantum-mechanical two-body systems, from hadrons to molecules.

## Quick Start

Run the following command in the Julia REPL or a notebook:

```julia
import Pkg; Pkg.add("TwoBody")
```

After installation, load the package and verify it works:

```julia
using TwoBody

H = Hamiltonian(
  Kinetic(hbar = 1, m = 1),
  Coulomb(coefficient = -1),
)

BS = BasisSet(
  SimpleGaussianBasis(13.00773),
  SimpleGaussianBasis(1.962079),
  SimpleGaussianBasis(0.444529),
  SimpleGaussianBasis(0.1219492),
)

solve(H, BS)
```

## Documentation

- Home: https://juliafewbody.github.io/TwoBody.jl
- User Guide:
  - [Hamiltonian](https://juliafewbody.github.io/TwoBody.jl/dev/Hamiltonian/)
  - [Database](https://juliafewbody.github.io/TwoBody.jl/dev/DB/)
  - [Rayleigh-Ritz Method](https://juliafewbody.github.io/TwoBody.jl/dev/Rayleigh-Ritz/)
  - [Gaussian Expansion Method](https://juliafewbody.github.io/TwoBody.jl/dev/GEM)
  - [Free Complement Method](https://juliafewbody.github.io/TwoBody.jl/dev/Free-Complement/)
  - [Finite Difference Method](https://juliafewbody.github.io/TwoBody.jl/dev/FDM/)
  - [Quantics Tensor Train](https://juliafewbody.github.io/TwoBody.jl/dev/QTT/)
  - [Variational Monte Carlo](https://juliafewbody.github.io/TwoBody.jl/dev/VNN/)
  - [Variational Neural Network](https://juliafewbody.github.io/TwoBody.jl/dev/VMC/)
- Developer Guide: https://juliafewbody.github.io/TwoBody.jl/dev/developer
- API Reference: https://juliafewbody.github.io/TwoBody.jl/dev/api
