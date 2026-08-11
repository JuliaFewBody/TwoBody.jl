export BasisSet, Basis, GeometricBasisSet, PrimitiveBasis, ContractedBasis,
       SimpleGaussianBasis, GaussianBasis, ComplexGaussianBasis,
       ComplexGaussianBasisSet

# type

abstract type Basis end
abstract type PrimitiveBasis <: Basis end

# struct

struct BasisSet
  basis::Vector{Basis}
  BasisSet(args...) = new([args...])
end

function geometric(r₁, rₙ, n::Int; nₘₐₓ::Int=n, nₘᵢₙ::Int=1)
  rgp = try; abs(rₙ/r₁)^(1/(n-nₘᵢₙ)); catch; 0; end # ratio of geometric sequence
  ν = abs(r₁)^-2 .* rgp.^(-2.0*(nₘᵢₙ-1:nₘₐₓ-1))
  return ν
end

Base.@kwdef struct GeometricBasisSet
  basistype
  r₁::Real
  rₙ::Real
  n::Int
  nₘᵢₙ::Int
  nₘₐₓ::Int
  basis::Vector{Basis}
  function GeometricBasisSet(basistype, r₁, rₙ, n::Int; nₘᵢₙ::Int=1, nₘₐₓ::Int=n)
    new(basistype, r₁, rₙ, n, nₘᵢₙ, nₘₐₓ, [basistype(α) for α in geometric(r₁, rₙ, n, nₘᵢₙ=nₘᵢₙ, nₘₐₓ=nₘₐₓ)])
  end
  function GeometricBasisSet(basistype; r₁=0.1, rₙ=80.0, n::Int=20, nₘᵢₙ::Int=1, nₘₐₓ::Int=n)
    new(basistype, r₁, rₙ, n, nₘᵢₙ, nₘₐₓ, [basistype(α) for α in geometric(r₁, rₙ, n, nₘᵢₙ=nₘᵢₙ, nₘₐₓ=nₘₐₓ)])
  end
end

Base.@kwdef struct SimpleGaussianBasis <: PrimitiveBasis
  a = 1
end

Base.@kwdef struct GaussianBasis <: PrimitiveBasis
  a = 1
  l = 0
  m = 0
end

Base.@kwdef struct ComplexGaussianBasis <: PrimitiveBasis
  a = 1
  ω = 1.0
  component::Symbol = :cos
  l = 0
  m = 0
end

# GeometricBasisSet constructs primitives from one exponent at a time.
GaussianBasis(a; l=0, m=0) = GaussianBasis(a, l, m)
ComplexGaussianBasis(a, ω, component; l=0, m=0) =
  ComplexGaussianBasis(a, ω, component, l, m)

function ComplexGaussianBasisSet(r₁, rₙ, n::Int; ω=1.0, l=0, m=0)
  n > 0 || throw(ArgumentError("n must be positive"))
  primitives = Basis[]
  for a in geometric(r₁, rₙ, n)
    push!(primitives, ComplexGaussianBasis(a, ω, :cos, l, m))
    push!(primitives, ComplexGaussianBasis(a, ω, :sin, l, m))
  end
  return BasisSet(primitives...)
end

@doc raw"""
`ComplexGaussianBasisSet(r₁, rₙ, n; ω=1.0, l=0, m=0)`

Construct the `2n` real, normalized complex-range Gaussian primitives

```math
r^l e^{-\nu_j r^2}\cos(\omega\nu_jr^2),\qquad
r^l e^{-\nu_j r^2}\sin(\omega\nu_jr^2),
```

where `νⱼ = 1/rⱼ²` and the ranges from `r₁` through `rₙ` form a
geometric progression. This is the real cos/sin form of the complex-range GEM
basis introduced by Hiyama, Kino, and Kamimura (2003) and used for the
highly excited hydrogen example of Hiyama and Kamimura (2018).
""" ComplexGaussianBasisSet

struct ContractedBasis <: Basis
  c::Vector
  φ::Vector
end

# utility

Base.string(b::Basis) = "$(typeof(b))(" * join(["$(symbol)=$(getproperty(b,symbol))" for symbol in fieldnames(typeof(b))], ", ") * ")"
Base.string(bs::BasisSet) = "BasisSet(" * join(["$(b)" for b in bs.basis], ", ") * ")"
Base.show(io::IO, b::Basis) = print(io, Base.string(b))
Base.show(io::IO, bs::BasisSet) = print(io, Base.string(bs))
Base.getindex(BS::BasisSet, index) = BS.basis[index]
Base.getindex(GBS::GeometricBasisSet, index) = GBS.basis[index]
Base.length(BS::BasisSet) = length(BS.basis)
Base.length(GBS::GeometricBasisSet) = length(GBS.basis)

# function

φ(b::SimpleGaussianBasis, r) = exp(-b.a*r^2)
φ(b::ContractedBasis, r, θ, φ) = sum(b.c[i] * φ(b.φ[i], r, θ, φ) for i in 1:length(b.c))

_with_exponent(b::Basis, a) = typeof(b)(a)
_with_exponent(b::GaussianBasis, a) = GaussianBasis(a, b.l, b.m)
_with_exponent(b::ComplexGaussianBasis, a) =
  ComplexGaussianBasis(a, b.ω, b.component, b.l, b.m)

# function for testing

function expikr(k,θk,φk,r,θr,φr)
  kx = k * sin(θk) * cos(φk)
  ky = k * sin(θk) * sin(φk)
  kz = k * cos(θk)
  rx = r * sin(θr) * cos(φr)
  ry = r * sin(θr) * sin(φr)
  rz = r * cos(θr)
  return exp(im * (kx*rx + ky*ry + kz*rz))
end

# docstring

@doc raw"""
`Basis` is an abstract type.
""" Basis

@doc raw"""
`PrimitiveBasis <: Basis` is an abstract type.
""" PrimitiveBasis

@doc raw"""
`BasisSet(basis1, basis2, ...)`
```math
\{ \phi_1, \phi_2, \phi_3, \cdots  \}
```
The basis set is the input for Rayleigh-Ritz method. You can define the basis set like this:
```math
\begin{aligned}
  \phi_1(r) &= \exp(-13.00773 ~r^2), \\
  \phi_2(r) &= \exp(-1.962079 ~r^2), \\
  \phi_3(r) &= \exp(-0.444529 ~r^2), \\
  \phi_4(r) &= \exp(-0.1219492 ~r^2).
\end{aligned}
```
```@example
BS = BasisSet(
  SimpleGaussianBasis(13.00773),
  SimpleGaussianBasis(1.962079),
  SimpleGaussianBasis(0.444529),
  SimpleGaussianBasis(0.1219492),
)
```
""" BasisSet

@doc raw"""
`geometric(r₁, rₙ, n::Int; nₘₐₓ::Int=n, nₘᵢₙ::Int=1)`

Exponents of Gaussian basis functions are given by geometric progression:
```math
\begin{aligned}
  & v_i = \frac{1}{r_i^2}, \\
  & r_i = r_1 a^{i-1}.
\end{aligned}
```

This function return array of $\nu_i$:
```math
(r_1, r_{n}, n, n_\mathrm{max}) \mapsto (\nu_1, \nu_2, \cdots, \nu_{n-1}, \nu_n, \nu_{n+1}, \cdots, \nu_{n_\mathrm{max}})
```

Usually $n = n_\mathrm{max}$. Set $n<n_\mathrm{max}$ if you want to extend the geometric progression.

Examples:
```jldoctest
julia> ν = TwoBody.geometric(0.1, 10.0, 5)
5-element Vector{Float64}:
 100.0
  10.0
   0.9999999999999997
   0.09999999999999996
   0.009999999999999995

julia> ν = TwoBody.geometric(0.1, 10.0, 5, nₘₐₓ = 10)
10-element Vector{Float64}:
 100.0
  10.0
   0.9999999999999997
   0.09999999999999996
   0.009999999999999995
   0.0009999999999999994
   9.999999999999994e-5
   9.999999999999992e-6
   9.999999999999991e-7
   9.999999999999988e-8
```
""" geometric(r₁, rₙ, n::Int; nₘₐₓ::Int=n, nₘᵢₙ::Int=1)

@doc raw"""
`GeometricBasisSet(basistype, r₁, rₙ, n; nₘᵢₙ=1, nₘₐₓ=n)`

This is a basis set with exponentials generated by `geometric(r₁, rₙ, n; nₘₐₓ=n, nₘᵢₙ=1)`. You can define the same basis set as Table A2 in [E. Hiyama, M. Kamimura, Front. Phys. 13, 132106 (2018)](https://doi.org/10.1007/s11467-018-0828-5) like this:
```math
  r_1 = 0.1,
  r_{n_\mathrm{max}} = 80.0,
  n_\mathrm{max} = 20.
```
```@example
BS = GeometricBasisSet(SimpleGaussianBasis, 0.1, 80.0, 20)
```
""" GeometricBasisSet

@doc raw"""
`SimpleGaussianBasis(a=1)`

!!! note
    This basis is not normalized and only for s-wave.

## Position-Space
```math
\phi_i(\pmb{r}) = \exp(-a_i r^2)
```

## Momentum-Space
```math
\phi_{i}(\pmb{k})  = \frac{1}{(2a_i)^{\frac{3}{2}}} \exp(-k^2/4a_i)
```

## Proof (Fourier Transform)
```math
\begin{aligned}
  \phi_{n}(\pmb{k})
  &= \frac{1}{\sqrt{2 \pi}^3}
     \int
     \phi_{n}(\pmb{r})
     \mathrm{e}^{\mathrm{i} \pmb{k} \cdot \pmb{r}}
     \mathrm{d}\pmb{r} \\
  &= \frac{1}{\sqrt{2 \pi}^3}
     \int
     \phi_{n}(\pmb{r})
     \mathrm{e}^{\mathrm{i} \pmb{k} \cdot \pmb{r}} 
     r^2 \sin (\theta)
     ~\mathrm{d}r
     \mathrm{d}\theta
     \mathrm{d} \varphi \\
  &= \frac{1}{\sqrt{2 \pi}^3}
     \iiint
     \mathrm{e}^{-\alpha_i r^2}
     \sqrt{4\pi} Y_{00}(\hat{\pmb{r}})
     \left[
       4 \pi \sum_{l'=0}^{\infty} \sum_{m=-l'}^{l'}
       \mathrm{i}^{l'}
       j_{l'}(pr)
       Y_{l'm'}(\hat{\pmb{k}})
       Y_{l'm'}^*(\hat{\pmb{r}})
     \right]
     r^2 \sin\theta~
     \mathrm{d} r
     \mathrm{d} \theta
     \mathrm{d} \varphi \\
  &= \frac{1}{\sqrt{2 \pi}^3}
     4 \pi \sqrt{4\pi} \sum_{l'=0}^{\infty} \sum_{m=-l'}^{l'} \left[
     \mathrm{i}^{l'}
     Y_{l'm'}(\hat{\pmb{k}})
     \int_0^{2 \pi} 
     \int_0^\pi
       Y_{00}(\hat{\pmb{r}})
       Y_{l'm'}^*(\hat{\pmb{r}})
       \sin (\theta)~
     \mathrm{d} \theta
     \mathrm{d} \varphi
     \int_0^{\infty}
       j_{l'}(pr)
       \mathrm{e}^{-\alpha_i r^2}
       r^{2}
       \mathrm{d}r
     \right]\\
  &=  \frac{1}{\sqrt{2 \pi}^3}
     4 \pi \sqrt{4\pi} \sum_{l'=0}^{\infty} \sum_{m=-l'}^{l'} \left[
     \mathrm{i}^{l'}
     Y_{l'm'}(\hat{\pmb{k}})
     \delta_{0l'}
     \delta_{0m'}
     \int_0^{\infty}
       j_{l'}(kr)
       \mathrm{e}^{-\alpha_i r^2}
       r^{2}
       \mathrm{d}r
     \right] \\
  &= \frac{1}{\sqrt{2 \pi}^3}
     4 \pi \sqrt{4\pi}
     \mathrm{i}^{0}
     Y_{00}(\hat{\pmb{k}})
     \int_0^{\infty}
       j_{0}(kr)
       \mathrm{e}^{-\alpha_i r^2}
       r^{2}
     \mathrm{d}r \\
  &= \frac{1}{2\pi\sqrt{2\pi}}
     4 \pi
     \frac{\sqrt{4\pi}}{\sqrt{4\pi}}
     \sqrt{\frac{\pi}{2}}
     \sqrt{\frac{2}{\pi}}
     \int_0^{\infty}
       j_{0}(kr)
       \mathrm{e}^{-\alpha_i r^2}
       r^{2}
     ~\mathrm{d}r \\
  &= \frac{1}{(2\alpha_i)^{\frac{3}{2}}} \mathrm{e}^{-\frac{k^2}{4 \alpha_i}}
\end{aligned}
```

## Formula

[plane-wave expansion in spherical harmonics](https://en.wikipedia.org/wiki/Plane-wave_expansion#Expansion_in_spherical_harmonics):
```math
\mathrm{e}^{\mathrm{i} \pmb{k} \cdot \pmb{r}}
= 
4 \pi \sum_{l=0}^{\infty} \sum_{m=-l}^{l}
\mathrm{i}^{l}
j_{l}(pr)
Y_{lm}(\hat{\pmb{k}})
Y_{lm}^*(\hat{\pmb{r}})
```

[special case of spherical harmonics](https://en.wikipedia.org/wiki/Spherical_harmonics#List_of_spherical_harmonics):
```math
Y_{00}(\hat{\pmb{r}}) = \frac{1}{\sqrt{4\pi}}
```

[orthonormality of spherical harmonics](https://en.wikipedia.org/wiki/Spherical_harmonics#Orthogonality_and_normalization):
```math
\int_0^{2\pi}
\int_0^\pi
    Y_{lm}(\hat{\pmb{r}})^*
    Y_{l'm'}(\hat{\pmb{r}})
\sin(\theta) ~
\mathrm{d} \theta
\mathrm{d} \varphi
=
\delta_{ll'}
\delta_{mm'}
```

citation needed:
```math
\sqrt{\frac{2}{\pi}}
\int
r^{l}
j_l(kr)
\mathrm{e}^{-\alpha r^2}
r^{2}
\mathrm{d} r
=
\frac{1}{(2\alpha)^{l+\frac{3}{2}}}
k^l e^{-\frac{k^2}{4\alpha}}
```
""" SimpleGaussianBasis

@doc raw"""
`GaussianBasis(a=1, l=0, m=0)`
```math
\phi_{ilm}(r, θ, φ) = N _{il} r^l \exp(-a_i r^2) Y_l^m(θ, φ)
```

This normalized primitive is used by the Gaussian expansion method (GEM).
Combine primitives with the same `l` and `m` and pass the basis set to the
existing Rayleigh–Ritz solver.
""" GaussianBasis

@doc raw"""
`ComplexGaussianBasis(a=1, ω=1, component=:cos, l=0, m=0)`

One normalized real component of a complex-range Gaussian pair:

```math
\phi^{\cos}_{nlm}=N^{\cos}_{nl}r^l e^{-a_n r^2}
\cos(\omega a_n r^2)Y_l^m(\hat{\boldsymbol r}),
```

or the corresponding sine function when `component=:sin`. Equivalently, the
pair is formed from exponents `(1±iω)aₙ`. Use
`ComplexGaussianBasisSet(r₁, rₙ, n; ω, l, m)` to construct both members
at every geometrically spaced range.

The analytic overlap, kinetic, constant, power-law, Coulomb, linear, and
Gaussian-potential matrix elements are supported by the ordinary
Rayleigh–Ritz `solve` interface.

References: E. Hiyama, Y. Kino, and M. Kamimura, *Prog. Part. Nucl. Phys.*
**51**, 223–307 (2003), and E. Hiyama and M. Kamimura, *Front. Phys.* **13**,
132106 (2018).
""" ComplexGaussianBasis

@doc raw"""
`ContractedBasis([c1, c2, ...], [basis1, basis2, ...])`
```math
\phi' = \sum_i c_i \phi_i
```
""" ContractedBasis
