export Operator, Hamiltonian, getindex, Kinetic, RestEnergy, RelativisticCorrection, RelativisticKinetic, Constant, Linear, Coulomb, PowerLaw, Gaussian, Exponential, Yukawa, Delta, Custom, Tabulated

# type

abstract type Operator end
abstract type PotentialTerm <: Operator end
abstract type KineticTerm <: Operator end

# struct

struct Hamiltonian
  terms::Array{Operator,1}
  Hamiltonian(args...) = new([args...])
end

Base.@kwdef struct Laplacian <: KineticTerm
  coefficient = 1
end

Base.@kwdef struct Kinetic <: KineticTerm
  hbar = 1
  m = 1
end

Base.@kwdef struct RestEnergy <: KineticTerm
  c = 1
  m = 1
end

Base.@kwdef struct RelativisticCorrection <: KineticTerm
  c = 1
  m = 1
  n = 2
end

Base.@kwdef struct RelativisticKinetic <: KineticTerm
  c = 1
  m = 1
end

Base.@kwdef struct Constant <: PotentialTerm
  constant = 1
end

Base.@kwdef struct Linear <: PotentialTerm
  coefficient = 1
end

Base.@kwdef struct Coulomb <: PotentialTerm
  coefficient = 1
end

Base.@kwdef struct PowerLaw <: PotentialTerm
  coefficient = 1
  exponent = 1
end

Base.@kwdef struct Gaussian <: PotentialTerm
  coefficient = 1
  exponent = 1
end

Base.@kwdef struct Exponential <: PotentialTerm
  coefficient = 1
  exponent = 1
end

Base.@kwdef struct Yukawa <: PotentialTerm
  coefficient = 1
  exponent = 1
end

Base.@kwdef struct Delta <: PotentialTerm
  coefficient = 1
end

Base.@kwdef struct Custom <: PotentialTerm
  f::Function
end

Base.@kwdef struct Tabulated <: PotentialTerm
  R::StepRangeLen
  V::Array{Number,1}
end

# utility

Base.string(t::Operator) = "$(typeof(t))(" * join(["$(symbol)=$(getproperty(t,symbol))" for symbol in fieldnames(typeof(t))], ", ") * ")"
Base.string(H::Hamiltonian) = "Hamiltonian(" * join(["$(term)" for term in H.terms], ", ") * ")"
Base.show(io::IO, t::Operator) = print(io, Base.string(t))
Base.show(io::IO, H::Hamiltonian) = print(io, Base.string(H))
Base.getindex(H::Hamiltonian, index) = H.terms[index]
Base.length(H::Hamiltonian) = length(H.terms)

# function

V(p::Constant       , r) = p.constant
V(p::Linear         , r) = p.coefficient * r
V(p::Coulomb        , r) = p.coefficient / r
V(p::PowerLaw       , r) = p.coefficient * r ^ p.exponent
V(p::Gaussian       , r) = p.coefficient * exp(- p.exponent * r ^ 2)
V(p::Exponential    , r) = p.coefficient * exp(- p.exponent * r)
V(p::Yukawa         , r) = p.coefficient * exp(- p.exponent * r) / r
# V(p::Delta          , r) =
V(p::Custom   , r) = p.f(r)
V(p::Tabulated, r) = p.V[findfirst(p.R, r)]

# docstring

@doc raw"""
`Operator` is an abstract type.
""" Operator

@doc raw"""
`PotentialTerm <: Operator` is an abstract type.
""" PotentialTerm

@doc raw"""
`KineticTerm <: Operator` is an abstract type.
""" KineticTerm

@doc raw"""
`Hamiltonian(operator1, operator2, ...)`
```math
\hat{H} = \sum_i \hat{o}_i
```

The Hamiltonian is the input for each solver. This is an example for the non-relativistic Hamiltonian of hydrogen atom in atomic units:

```math
\hat{H} = 
- \frac{1}{2} \nabla^2
- \frac{1}{r}
```

```@example
H = Hamiltonian(
  Kinetic(hbar = 1, m = 1),
  Coulomb(coefficient = -1),
)
```
""" Hamiltonian

@doc raw"""
`Laplacian(coefficient=1)`
```math
+ a\nabla^2
```
| Arguments | Symbol |
| :-- | :-- |
| `coefficient` | ``a`` |
""" Laplacian

@doc raw"""
`Kinetic(hbar=1, m=1)`
```math
-\frac{\hbar^2}{2m} \nabla^2
```
""" Kinetic

@doc raw"""
`RestEnergy(c=1, m=1)`
```math
m c^2
```
Use `c = 137.035999177` (from [2022 CODATA](https://physics.nist.gov/cgi-bin/cuu/Value?alphinv)) in the atomic units.
""" RestEnergy

@doc raw"""
`RelativisticCorrection(c=1, m=1, n=2)`
The p^{2n} term of the Taylor expansion:
```math
\begin{aligned}
  \sqrt{p^2 c^2 + m^2 c^4}
  =& m \times c^2 \\
  &+ 1 / 2   / m         \times p^2 (n=1) \\
  &- 1 / 8   / m^3 / c^2 \times p^4 (n=2) \\
  &+ 1 / 16  / m^5 / c^4 \times p^6 (n=3) \\
  &- 5 / 128 / m^7 / c^6 \times p^8 (n=4) \\
  &+ \cdots
\end{aligned}
```
Use `c = 137.035999177` (from [2022 CODATA](https://physics.nist.gov/cgi-bin/cuu/Value?alphinv)) in the atomic units.
""" RelativisticCorrection

@doc raw"""
`RelativisticKinetic(c=1, m=1)`
```math
\sqrt{p^2 c^2 + m^2 c^4} - m c^2
```
Use `c = 137.035999177` (from [2022 CODATA](https://physics.nist.gov/cgi-bin/cuu/Value?alphinv)) in the atomic units.
""" RelativisticKinetic

@doc raw"""
`Constant(constant=1)`
```math
+ c
```
| Arguments | Symbol |
| :-- | :-- |
| `constant` | ``c`` |
""" Constant

@doc raw"""
`Linear(coefficient=1)`
```math
+ ar
```
| Arguments | Symbol |
| :-- | :-- |
| `coefficient` | ``a`` |
""" Linear

@doc raw"""
`Coulomb(coefficient=1)`
```math
+ \frac{a}{r}
```
| Arguments | Symbol |
| :-- | :-- |
| `coefficient` | ``a`` |
""" Coulomb

@doc raw"""
`PowerLaw(coefficient=1, exponent=1)`
```math
+ ar^n
```
| Arguments | Symbol |
| :-- | :-- |
| `coefficient` | ``a`` |
| `exponent` | ``n`` |
""" PowerLaw

@doc raw"""
`Gaussian(coefficient=1, exponent=1)`
```math
+ a \exp(- b r^2)
```
| Arguments | Symbol |
| :-- | :-- |
| `coefficient` | ``a`` |
| `exponent`    | ``b`` |
""" Gaussian

@doc raw"""
`Exponential(coefficient=1, exponent=1)`
```math
+ a \exp(- b r)
```
| Arguments | Symbol |
| :-- | :-- |
| `coefficient` | ``a`` |
| `exponent`    | ``b`` |
""" Exponential

@doc raw"""
`Yukawa(coefficient=1, exponent=1)`
```math
+ \frac{a}{r} \exp(- b r)
```
| Arguments | Symbol |
| :-- | :-- |
| `coefficient` | ``a`` |
| `exponent`    | ``b`` |
""" Yukawa

@doc raw"""
`Delta(coefficient=1)`
```math
+ a δ(r)
```
| Arguments | Symbol |
| :-- | :-- |
| `coefficient` | ``a`` |
""" Delta

@doc raw"""
`Custom(f)`
```math
+ f(r)
```
""" Custom

@doc raw"""
`Tabulated(R, V)`
""" Tabulated
