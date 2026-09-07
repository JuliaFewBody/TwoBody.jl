```@meta
CurrentModule = TwoBody
```

# TwoBody.jl

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oc8hfyS4gaCxNvaY00rUiiDmGMPKpfxC?usp=sharing)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliafewbody.github.io/TwoBody.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliafewbody.github.io/TwoBody.jl/dev/)
[![Citation](https://img.shields.io/badge/citation-BibTeX-778899)](https://github.com/JuliaFewBody/TwoBody.jl/blob/main/CITATION.bib)
[![License](https://img.shields.io/github/license/JuliaFewBody/TwoBody.jl)](https://github.com/JuliaFewBody/TwoBody.jl/blob/main/LICENSE)
[![Build Status](https://github.com/JuliaFewBody/TwoBody.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaFewBody/TwoBody.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaFewBody/TwoBody.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaFewBody/TwoBody.jl)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-contributor's%20guide-blueviolet)](https://github.com/SciML/ColPrac)

[TwoBody.jl](https://github.com/JuliaFewBody/TwoBody.jl): A Julia package for solving quantum-mechanical two-body problems

TwoBody.jl provides a flexible framework for constructing two-body Hamiltonians and solving the corresponding Schrödinger equation using a variety of numerical methods. It covers approaches ranging from basis-set and grid methods to tensor-network, stochastic, and neural-network methods. Beyond serving as a proof of concept for [FewBody.jl](https://github.com/JuliaFewBody/FewBody.jl), TwoBody.jl is designed to support practical, cross-scale calculations of quantum two-body systems, from hadrons to molecules.

## Install

Run the following command in the Julia REPL or a notebook:

```julia
import Pkg; Pkg.add("TwoBody")
```

## Usage

Run the following code before each use.
```@example index
using TwoBody
```

Define the [Hamiltoninan](@ref Hamiltonian). This is an example for the non-relativistic Hamiltonian of hydrogen atom in atomic units:
```math
\hat{H} = 
- \frac{1}{2} \nabla^2
- \frac{1}{r}
```
```@example index
H = Hamiltonian(
  Kinetic(hbar = 1, m = 1),
  Coulomb(coefficient = -1),
)
nothing # hide
```

The usage depends on the method. Define the basis set for the [Rayleigh-Ritz Method](@ref Rayleigh-Ritz-Method):
```math
\begin{aligned}
  \phi_1(r) &= \exp(-13.00773 ~r^2), \\
  \phi_2(r) &= \exp(-1.962079 ~r^2), \\
  \phi_3(r) &= \exp(-0.444529 ~r^2), \\
  \phi_4(r) &= \exp(-0.1219492 ~r^2).
\end{aligned}
```
```@example index
BS = BasisSet(
  SimpleGaussianBasis(13.00773),
  SimpleGaussianBasis(1.962079),
  SimpleGaussianBasis(0.444529),
  SimpleGaussianBasis(0.1219492),
)
nothing # hide
```

You should find
```math
E_{n=1} = -0.499278~E_\mathrm{h},
```
which is amazingly good for only four basis functions according to [Thijssen(2007)](https://doi.org/10.1017/CBO9781139171397). The exact ground-state energy is ``-0.5~E_\mathrm{h}``.

```@repl index
solve(H, BS)
```

The wave function is also good. However, the Gaussian basis does not satisfy [the Kato’s cusp condition](https://doi.org/10.1002/cpa.3160100201).

```@example index
# solve
res = solve(H, BS)

# benchmark
import Antique
HA = Antique.HydrogenAtom(Z=1, Eₕ=1.0, a₀=1.0, mₑ=1.0, ℏ=1.0)

# plot
using CairoMakie
fig = Figure(size=(420,300), fontsize=11, backgroundcolor=:transparent)
axis = Axis(fig[1,1], xlabel=L"$r / a_0$", ylabel=L"$\psi(r) / a_0^{-3/2}$", ylabelsize=16.5, xlabelsize=16.5, limits=(0,4,0,1.1/sqrt(π)))
lines!(axis, 0..5, r -> abs(TwoBody.ψ(res,r)), label="TwoBody.jl")
lines!(axis, 0..5, r -> abs(Antique.wavefunction(HA, r, 0, 0)), linestyle=:dash, color=:black, label="Antique.jl")
axislegend(axis, position=:rt, framevisible=false)
fig
```

## Hydrogen atom benchmark

The following table compares the two lowest ``s``-wave energies of the
hydrogen atom in atomic units. Here ``n=0`` denotes the ground state and
``n=1`` the first excited state. The numerical values use the example settings
from the corresponding method pages: 20 Gaussian functions for RR, the
nine-function complement basis (``M_n=9``) for FC, 20 real-range Gaussians
with ``r_1=0.1`` and ``r_{20}=80`` for GEM, ``\Delta r=0.1`` and
``r_\mathrm{max}=50`` for FDM, the default 1024-point grid
(``\mathtt{quantics}=10``) for QTT+DMRG, and the documented training or
sampling settings for VNN and VMC. A dash indicates that the state is not
currently available.

| Method | ``n=0`` | ``n=1`` |
|:---|---:|---:|
| [RR](@ref Rayleigh-Ritz-Method) | -0.499981735104 | -0.124997703473 |
| [FC](@ref Free-Complement-Method) | -0.499999999978 | -0.123665583532 |
| [GEM](@ref Gaussian-Expansion-Method) | -0.499981735103 | -0.124997703473 |
| [FDM](@ref Finite-Difference-Method) | -0.498756211209 | -0.124921972504 |
| [QTT+DMRG](@ref Quantics-Tensor-Train) | -0.499702911360 | -0.124981415403 |
| [VNN](@ref Variational-Neural-Network) | -0.468779111883 | — |
| [VMC](@ref Variational-Monte-Carlo) | -0.480218518855 | — |
| Exact | -0.500000000000 | -0.125000000000 |

## API reference

```@index
```
