```@meta
CurrentModule = TwoBody
```

# Rayleigh-Ritz Method

The Rayleigh-Ritz method is a variational method for approximating the eigenvalues and eigenfunctions of a Hamiltonian. The expectation value obtained from a trial wave function is an upper bound on the exact ground-state energy. Note that nonlinear parameters, such as the exponents of Gaussian basis functions, are not optimized by the linear procedure described below.

## Theory

In the Rayleigh-Ritz method, the trial wave function is written as a linear combination of basis functions ``\phi_1, \phi_2, \phi_3, \ldots``:

```math
\psi = \sum_i c_i \phi_i.
```

We then seek the expansion coefficients ``c_i`` that minimize the energy ``E[\psi]``. The coefficients are ultimately obtained by solving the generalized eigenvalue problem

```math
\boldsymbol{H}\boldsymbol{c} = E\boldsymbol{S}\boldsymbol{c}.
```

The derivation is given below. First, let us define the notation. ``\boldsymbol{H}`` is the Hamiltonian matrix, whose elements are

```math
H_{ij} = \langle \phi_i | \hat{H} | \phi_j \rangle,
```

and ``\boldsymbol{S}`` is the overlap matrix, whose elements are

```math
S_{ij} = \langle \phi_i | \phi_j \rangle.
```

Substituting ``\psi = \sum_i c_i \phi_i`` into ``E[\psi]`` gives

```math
\begin{aligned}
E[\psi]
&= \frac{\langle \psi | \hat{H} | \psi \rangle}
        {\langle \psi | \psi \rangle} \\
&= \frac{\left\langle \sum_i c_i \phi_i \middle| \hat{H} \middle| \sum_j c_j \phi_j \right\rangle}
        {\left\langle \sum_i c_i \phi_i \middle| \sum_j c_j \phi_j \right\rangle} \\
&= \frac{\sum_{i,j} c_i c_j \langle \phi_i | \hat{H} | \phi_j \rangle}
        {\sum_{i,j} c_i c_j \langle \phi_i | \phi_j \rangle} \\
&= \frac{\sum_{i,j} c_i c_j H_{ij}}
        {\sum_{i,j} c_i c_j S_{ij}}.
\end{aligned}
```

Here and below, the basis functions and coefficients are assumed to be real. For a complex-valued basis, the coefficient on the bra side must be replaced by its complex conjugate ``c_i^*``. The expansion above follows directly from the distributive law and can be seen explicitly by writing ``\sum_i c_i \phi_i = c_1 \phi_1 + c_2 \phi_2 + \cdots``.

We minimize this expression subject to the normalization constraint

```math
\sum_{i,j} c_i c_j S_{ij} = 1.
```

Using the method of Lagrange multipliers, define

```math
L(c_i,E)
= \sum_{i,j} c_i c_j H_{ij}
- E \sum_{i,j} c_i c_j S_{ij}
+ E.
```

When ``L`` is differentiated with respect to ``c_k``, only terms for which ``i=k`` or ``j=k`` remain. Therefore,

```math
\begin{aligned}
\frac{\partial L}{\partial c_k} &= 0, \\
\sum_i c_i H_{ik} + \sum_j c_j H_{kj}
- E \sum_i c_i S_{ik} - E \sum_j c_j S_{kj} &= 0, \\
2 \sum_j H_{kj} c_j - 2E \sum_j S_{kj} c_j &= 0, \\
\sum_j \left(H_{kj}c_j - E S_{kj}c_j\right) &= 0, \\
\boldsymbol{H}\boldsymbol{c} - E\boldsymbol{S}\boldsymbol{c} &= 0, \\
\boldsymbol{H}\boldsymbol{c} &= E\boldsymbol{S}\boldsymbol{c}.
\end{aligned}
```

In the third line, we used ``H_{ij}=H_{ji}`` and ``S_{ij}=S_{ji}``, which hold for the real-valued case considered here. Similar derivations can be found in the following references:

- J. M. Thijssen, *Computational Physics* (Japanese translation, Maruzen Publishing, 2012), Sec. 3.1, [Variational Calculations](https://www.maruzen-publishing.co.jp/item/b294596.html).
- A. Szabo and N. S. Ostlund, *Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory*, Vol. 1 (Japanese translation, University of Tokyo Press, 1987), Sec. 1.3.2, [The Linear Variational Problem](http://www.utp.or.jp/book/b302128.html).

In Julia, the generalized eigenvalue problem ``\boldsymbol{H}\boldsymbol{c} = E\boldsymbol{S}\boldsymbol{c}`` can be solved easily with the `LinearAlgebra` standard library:

```julia
using LinearAlgebra
E, c = eigen(H, S)
```

It is therefore not necessary to know the details of the eigensolver here. The practical challenge is instead to evaluate the matrix elements ``H_{ij} = \langle \phi_i | \hat{H} | \phi_j \rangle`` and ``S_{ij} = \langle \phi_i | \phi_j \rangle``, and to choose basis functions for which these elements can be calculated efficiently.

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

Define the basis set:

```math
\begin{aligned}
  \phi_1(r) &= \exp(-13.00773 ~r^2), \\
  \phi_2(r) &= \exp(-1.962079 ~r^2), \\
  \phi_3(r) &= \exp(-0.444529 ~r^2), \\
  \phi_4(r) &= \exp(-0.1219492 ~r^2).
\end{aligned}
```
```@example example
BS = BasisSet(
  SimpleGaussianBasis(13.00773),
  SimpleGaussianBasis(1.962079),
  SimpleGaussianBasis(0.444529),
  SimpleGaussianBasis(0.1219492),
)
nothing # hide
```

Solve the eigenvalue problem. You should find
```math
E_{n=1} = -0.499278~E_\mathrm{h},
```
which is amazingly good for only four basis functions according to [Thijssen(2007)](https://doi.org/10.1017/CBO9781139171397). The exact ground-state energy is ``-0.5~E_\mathrm{h}``.

```@repl example
solve(H, BS)
```

## Example of Hydrogen Atom

Analytical solutions are implemented in [Antique.jl](https://ohno.github.io/Antique.jl/stable/HydrogenAtom/).

```@example example
# solve
using TwoBody
H = Hamiltonian(Kinetic(1, 1), Coulomb(-1))
BS = GeometricBasisSet(SimpleGaussianBasis, 0.1, 80.0, 20)
res = solve(H, BS)

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
  @printf("%2d  %+.9f  %+.9f\n", n, res.E[n], Antique.energy(HA; n=n))
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
  lines!(axis, 0..50, r -> 4π * r^2 * abs(TwoBody.ψ(res,r,n=n))^2, label="TwoBody.jl")
  lines!(axis, 0..50, r -> 4π * r^2 * abs(Antique.wavefunction(HA, r, 0, 0; n=n))^2, label="Antique.jl", color=:black, linestyle=:dash)
  axislegend(axis, "n = $n", position=:rt, framevisible=false)
end
fig
save("assets/RR_HA.svg", fig) # hide
; # hide
```
![](assets/RR_HA.svg)

## Example of Spherical Oscillator

Analytical solutions are implemented in [spherical oscillator](https://ohno.github.io/Antique.jl/stable/SphericalOscillator/).

```@example example
# solve
using TwoBody
H = Hamiltonian(Kinetic(1, 1), PowerLaw(coefficient=1/2, exponent=2))
BS = GeometricBasisSet(SimpleGaussianBasis, 1.0, 10.0, 20)
res = solve(H, BS)

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
  @printf("%2d  %+.9f  %+.9f\n", n-1, res.E[n], Antique.energy(SO; n=n-1))
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
  lines!(axis, 0..50, r -> 4π * r^2 * abs(TwoBody.ψ(res,r,n=n))^2, label="TwoBody.jl")
  lines!(axis, 0..50, r -> 4π * r^2 * abs(Antique.wavefunction(SO, r, 0, 0; n=n-1))^2, label="Antique.jl", color=:black, linestyle=:dash)
  axislegend(axis, "n = $(n-1)", position=:rt, framevisible=false)
end
fig
save("assets/RR_SO.svg", fig) # hide
; # hide
```
![](assets/RR_SO.svg)

## Examples of Hadron Spectroscopy

Natural units are used below; masses are displayed in MeV.

### ``\Lambda_c(1/2^+)``

Parameters follow [Kim, Hiyama, Oka, and Suzuki (2020)](https://doi.org/10.1103/PhysRevD.102.014004).

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

### ``\eta_c(1S)``: Meng, Wang, and Oka

Parameters follow [Meng, Wang, and Oka (2024)](https://doi.org/10.48550/arXiv.2404.01238).

```@example meng_charmonium
using TwoBody

m₁ = 1.836
m₂ = 1.836
κ = 0.5069
κ′ = 1.8609
spin = -3
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
    coefficient=2π * κ′ * spin / (3m₁ * m₂ * (sqrt(π) * r₀)^3),
    exponent=inv(r₀^2),
  ),
)
BS = GeometricBasisSet(GaussianBasis, 0.1, 80.0, 20)

round(1000 * solve(H, BS; info=0).E[1]; digits=3)
```

### ``\eta_c(1S)``: Arifi, Happ, Ohno, and Oka

Parameters follow [Arifi, Happ, Ohno, and Oka (2024)](https://arxiv.org/abs/2401.07933).

```@example arifi_charmonium
using TwoBody

m₁ = 1.515
m₂ = 1.515
αs = 0.285
spin = -3/4
μ = inv(inv(m₁) + inv(m₂))
λ = 1.437 * sqrt(μ)

H = Hamiltonian(
  RestEnergy(m=m₁),
  RestEnergy(m=m₂),
  RelativisticKinetic(m=m₁),
  RelativisticKinetic(m=m₂),
  Constant(constant=-0.189),
  Linear(coefficient=0.092),
  Coulomb(coefficient=-4αs/3),
  Gaussian(
    coefficient=32π * αs * (λ / sqrt(π))^3 * spin / (9m₁ * m₂),
    exponent=λ^2,
  ),
)
BS = GeometricBasisSet(GaussianBasis, 0.358, 2.720, 10)

round(1000 * solve(H, BS; info=0).E[1]; digits=3)
```

## STO-3G

This example reproduces the STO-3G calculation for hydrogen reported by [Pérez-Torres (2019)](https://doi.org/10.1021/acs.jchemed.8b00959). In the contracted calculation, the published coefficients are held fixed, and the resulting contracted function is supplied to the solver. In the uncontracted calculation, the three primitive functions are supplied separately, allowing the Rayleigh–Ritz solver to optimize their linear coefficients.

```@example perez_sto3g
using TwoBody

# Hamiltonian
H = Hamiltonian(
  Kinetic(hbar=1.0, m=1.0),
  Coulomb(coefficient=-1.0),
)

# parameters
α = (0.109818, 0.405771, 2.227660)
C = (0.444635, 0.535328, 0.154329)

# basis set
primitives = ntuple(i -> SimpleGaussianBasis(α[i]), 3)
normalization = ntuple(i -> (2α[i] / π)^(3/4), 3)
coefficients = ntuple(i -> C[i] * normalization[i], 3)
contracted_basis = ContractedBasis(coefficients, primitives)

# solve
contracted = solve(H, BasisSet(contracted_basis))
uncontracted = solve(H, BasisSet(primitives...))

# results
println("Contracted STO-3G")
println("  This work: ", contracted.E[1])
println("  Reference: ", -0.494908)
println("Uncontracted STO-3G")
println("  This work: ", uncontracted.E[1])
println("  Reference: ", -0.495010)
```

The results agree with those in Table 1 of the Supporting Information. The small differences between the calculated and reference values arise from rounding in the published parameters.

## API reference

### Solver

```@docs; canonical=false
solve(hamiltonian::Hamiltonian, basisset::BasisSet; perturbation=Hamiltonian(), info=4)
solve(hamiltonian::Hamiltonian, basis::Basis; perturbation=Hamiltonian(), info=4)
solve(hamiltonian::Hamiltonian, basisset::GeometricBasisSet; perturbation=Hamiltonian(), info=4)
optimize(hamiltonian::Hamiltonian, basisset::BasisSet; perturbation=Hamiltonian(), info=4, progress=true, optimizer=Optim.NelderMead(), options...)
optimize(hamiltonian::Hamiltonian, basis::Basis; perturbation=Hamiltonian(), info=1, progress=true, optimizer=Optim.NelderMead(), options...)
optimize(hamiltonian::Hamiltonian, basisset::GeometricBasisSet; perturbation=Hamiltonian(), info=4, progress=true, optimizer=Optim.NelderMead(), options...)
```

### Basis Set

```@docs; canonical=false
TwoBody.BasisSet
TwoBody.GeometricBasisSet
TwoBody.geometric(r₁, rₙ, n::Int; nₘₐₓ::Int=n, nₘᵢₙ::Int=1)
```

### Basis Functions

```@docs; canonical=false
TwoBody.SimpleGaussianBasis
TwoBody.ContractedBasis
```

### Matrix

```@docs; canonical=false
matrix(basisset::BasisSet)
matrix(operator::Operator, basisset::BasisSet)
matrix(hamiltonian::Hamiltonian, basisset::BasisSet)
```

### Matrix Elements

```@docs; canonical=false
element(SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)
element(o::Hamiltonian, B1::Basis, B2::Basis)
element(o::RestEnergy, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)
element(o::Laplacian, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)
element(o::Kinetic, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)
element(o::Constant, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)
element(o::Linear, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)
element(o::Coulomb, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)
element(o::PowerLaw, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)
element(o::Gaussian, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)
element(o::Custom, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)
```
