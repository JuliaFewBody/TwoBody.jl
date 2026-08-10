```@meta
CurrentModule = TwoBody
```

# TwoBody.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliafewbody.github.io/TwoBody.jl/stable) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliafewbody.github.io/TwoBody.jl/dev) [![Build Status](https://github.com/JuliaFewBody/TwoBody.jl/workflows/CI/badge.svg)](https://github.com/JuliaFewBody/TwoBody.jl/actions)

[TwoBody.jl](https://github.com/JuliaFewBody/TwoBody.jl): a Julia package for quantum mechanical two-body problems

## Install

Run the following code on the REPL or Jupyter Notebook to install this package.

```julia
import Pkg; Pkg.add(url="https://github.com/JuliaFewBody/TwoBody.jl.git")
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
  NonRelativisticKinetic(ℏ = 1 , m = 1),
  CoulombPotential(coefficient = -1),
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
HA = Antique.HydrogenAtom(Z=1, E_h=1.0, a_0=1.0, m_e=1.0, hbar=1.0)

# plot
using CairoMakie
fig = Figure(size=(420,300), fontsize=11, backgroundcolor=:transparent)
axis = Axis(fig[1,1], xlabel=L"$r / a_0$", ylabel=L"$\psi(r) / a_0^{-3/2}$", ylabelsize=16.5, xlabelsize=16.5, limits=(0,4,0,1.1/sqrt(π)))
lines!(axis, 0..5, r -> abs(TwoBody.ψ(res,r)), label="TwoBody.jl")
lines!(axis, 0..5, r -> abs(Antique.wavefunction(HA,r,0,0)), linestyle=:dash, color=:black, label="Antique.jl")
axislegend(axis, position=:rt, framevisible=false)
fig
```

## API reference

```@index
```
