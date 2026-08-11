```@meta
CurrentModule = TwoBody
```

# Finite Difference Method

This method solve the eigenvalue problem for the Hamiltonian discretized as a sparse matrix with finite difference approximation,
```math
\pmb{H} \pmb{\psi} = E \pmb{\psi}.
```
The eigenvalue ``E`` is an approximation of the exact energy and the eigenvector ``\pmb{\psi}`` is a vector of the approximated values of the exact wavefunction ``\psi(r)`` on points of the grid,
```math
\pmb{\psi}
=
\left(\begin{array}{c}
  \psi(r_1) \\
  \psi(r_2) \\
  \psi(r_3) \\
  \vdots \\
\end{array}\right).
```
A uniform grid spacing is used, ``r_{i+1} = r_{i} + \Delta r``. See the API reference for the expression of the matrix ``\pmb{H}``.

## Usage

Run the following code before each use.

```@example example
using TwoBody
```

Define the [Hamiltoninan](@ref Hamiltonian). This is an example for the non-relativistic Hamiltonian of hydrogen atom in atomic units:

```math
\hat{H} = 
- \frac{1}{2} \nabla^2
- \frac{1}{r}
```
```@example example
H = Hamiltonian(
  Kinetic(hbar = 1, m = 1),
  Coulomb(coefficient = -1),
)
nothing # hide
```

Set the calculation options.

```@example example
FDM = FiniteDifferenceMethod(
  Δr = 0.1,
  rₘₐₓ = 50.0,
  l = 0,
  direction = :c,
  solver = :LinearAlgebra,
)
nothing # hide
```

Solve the eigenvalue problem. You should find reasonable approximations to [the exact eigenvalues](https://ohno.github.io/Antique.jl/stable/HydrogenAtom/#Eigenvalues):
```math
\begin{aligned}
  E_{n=1} &= -0.5,\\
  E_{n=2} &= -0.125,\\
  E_{n=3} &= -0.05555\cdots,\\
  E_{n=4} &= -0.03125.
\end{aligned}
```
By default the eigenvalues and the expectation values are displayed.

```@repl example
solve(H, FDM)
```

## Example of Hydrogen Atom

Analytical solutions are implemented in [Antique.jl](https://ohno.github.io/Antique.jl/stable/HydrogenAtom/).

```@example example
# solve
using TwoBody
H = Hamiltonian(Kinetic(1, 1), Coulomb(-1))
FDM = FiniteDifferenceMethod()
res = solve(H, FDM, info=0, nₘₐₓ=4)

# benchmark
import Antique
HA = Antique.HydrogenAtom(Z=1, Eₕ=1.0, a₀=1.0, mₑ=1.0, ℏ=1.0)

# energy
using Printf
println("Total Energy Eₙ")
println("------------------------------")
println(" n     numerical    analytical")
println("------------------------------")
for n in 1:4
  @printf("%2d  %+.9f  %+.9f\n", n, res.E[n], Antique.E(HA; n=n))
end

# wave function
using CairoMakie
fig = Figure(
  size = (840,600),
  fontsize = 11,
  backgroundcolor = :transparent
)
for n in 1:4
  axis = Axis(
    fig[div(n-1,2)+1,rem(n-1,2)+1],
    xlabel = L"$r~/~a_0$",
    ylabel = L"$4\pi r^2|\psi(r)|^2~ /~{a_0}^{-1}$",
    xlabelsize = 16.5,
    ylabelsize = 16.5,
    limits=(
      0, [5, 15, 30, 50][n],
      0, [0.6, 0.2, 0.11, 0.07][n],
    )
  )
  X = res.method.R
  Y = 4π * X .^2 .* res.ψ[:,n] .^ 2
  scatter!(axis, X, Y, label="TwoBody.jl", markersize=6)
  lines!(axis, 0..50, r -> 4π * r^2 * abs(Antique.ψ(HA, r, 0, 0; n=n))^2, label="Antique.jl", color=:black)
  axislegend(axis, "n = $n", position=:rt, framevisible=false)
end
save("assets/FDM_HA.svg", fig) # hide
; # hide
```
![](assets/FDM_HA.svg)

## Example of Spherical Oscillator

Analytical solutions are implemented in [spherical oscillator](https://ohno.github.io/Antique.jl/stable/SphericalOscillator/).

```@example example
# solve
using TwoBody
H = Hamiltonian(Kinetic(1, 1), PowerLaw(coefficient=1/2, exponent=2))
FDM = FiniteDifferenceMethod(rₘₐₓ=10.0)
res = solve(H, FDM, info=0, nₘₐₓ=4)

# benchmark
import Antique
SO = Antique.SphericalOscillator(k=1.0, μ=1.0, ℏ=1.0)

# energy
using Printf
println("Total Energy Eₙ")
println("------------------------------")
println(" n     numerical    analytical")
println("------------------------------")
for n in 1:4
  @printf("%2d  %+.9f  %+.9f\n", n-1, res.E[n], Antique.E(SO; n=n-1))
end

# wave function
using CairoMakie
fig = Figure(
  size = (840,600),
  fontsize = 11,
  backgroundcolor = :transparent
)
for n in 1:4
  axis = Axis(
    fig[div(n-1,2)+1,rem(n-1,2)+1],
    xlabel = L"$r~/~a_0$",
    ylabel = L"$4\pi r^2|\psi(r)|^2~ /~{a_0}^{-1}$",
    xlabelsize = 16.5,
    ylabelsize = 16.5,
    limits=(
      0, [4.5, 5.0, 5.5, 6.0][n],
      0, [0.90, 0.75, 0.70, 0.65][n],
    )
  )
  X = res.method.R
  Y = 4π * X .^2 .* res.ψ[:,n] .^ 2
  scatter!(axis, X, Y, label="TwoBody.jl", markersize=6)
  lines!(axis, 0..50, r -> 4π * r^2 * abs(Antique.ψ(SO, r, 0, 0; n=n-1))^2, label="Antique.jl", color=:black)
  axislegend(axis, "n = $(n-1)", position=:rt, framevisible=false)
end
fig
save("assets/FDM_SO.svg", fig) # hide
; # hide
```
![](assets/FDM_SO.svg)

## API reference

```@docs; canonical=false
TwoBody.FiniteDifferenceMethod
TwoBody.solve(hamiltonian::Hamiltonian, method::FiniteDifferenceMethod)
TwoBody.matrix(o::Hamiltonian, method::FiniteDifferenceMethod)
TwoBody.matrix(o::RestEnergy, method::FiniteDifferenceMethod)
TwoBody.matrix(o::Kinetic, method::FiniteDifferenceMethod)
TwoBody.matrix(o::PotentialTerm, method::FiniteDifferenceMethod)
```
