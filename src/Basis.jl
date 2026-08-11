export BasisSet, Basis, GeometricBasisSet, PrimitiveBasis, ContractedBasis, SimpleGaussianBasis, GaussianBasis

# type

abstract type Basis end
abstract type PrimitiveBasis <: Basis end

# struct

struct BasisSet{B<:Basis}
  basis::Vector{B}

  BasisSet(basis::Vector{B}, ::Val{:validated}) where {B<:Basis} = new{B}(basis)
end

function BasisSet(args::Basis...)
  isempty(args) && return BasisSet(Basis[], Val(:validated))

  basis_type = foldl(typejoin, (typeof(basis) for basis in args))
  stored_basis = Vector{basis_type}(undef, length(args))
  for index in eachindex(args)
    stored_basis[index] = args[index]
  end
  return BasisSet(stored_basis, Val(:validated))
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

Base.@kwdef struct SimpleGaussianBasis{T<:Real} <: PrimitiveBasis
  a::T = 1
end

Base.@kwdef struct GaussianBasis{T<:Real} <: PrimitiveBasis
  a::T = 1
  l::Int = 0
  m::Int = 0
end

struct ContractedBasis{N, T<:Number, P<:NTuple{N,PrimitiveBasis}} <: Basis
  coefficients::NTuple{N,T}
  primitives::P

  function ContractedBasis(
    coefficients::NTuple{N,T},
    primitives::P,
    ::Val{:validated},
  ) where {N, T<:Number, P<:NTuple{N,PrimitiveBasis}}
    N > 0 || throw(ArgumentError("a contracted basis requires at least one coefficient and primitive basis"))
    new{N,T,P}(coefficients, primitives)
  end
end

function ContractedBasis(
  coefficients::C,
  primitives::P,
) where {N, C<:NTuple{N,Number}, P<:NTuple{N,PrimitiveBasis}}
  N > 0 || throw(ArgumentError("a contracted basis requires at least one coefficient and primitive basis"))
  return ContractedBasis(promote(coefficients...), primitives, Val(:validated))
end

function _contracted_basis(coefficients, primitives)
  isempty(coefficients) && throw(ArgumentError("a contracted basis requires at least one coefficient and primitive basis"))
  length(coefficients) == length(primitives) || throw(DimensionMismatch("the numbers of coefficients and primitive bases must match"))
  all(coefficient -> coefficient isa Number, coefficients) || throw(ArgumentError("all contraction coefficients must be numbers"))
  all(primitive -> primitive isa PrimitiveBasis, primitives) || throw(ArgumentError("all components of a contracted basis must be primitive bases"))

  coefficient_values = Tuple(coefficients)
  stored_primitives = Tuple(primitives)
  coefficient_type = foldl(promote_type, (typeof(coefficient) for coefficient in coefficient_values))
  stored_coefficients = ntuple(index -> convert(coefficient_type, coefficient_values[index]), length(coefficient_values))

  return ContractedBasis(stored_coefficients, stored_primitives, Val(:validated))
end

ContractedBasis(coefficients::AbstractVector, primitives::AbstractVector) = ContractedBasis(Tuple(coefficients), Tuple(primitives))
ContractedBasis(coefficients::Tuple, primitives::Tuple) = _contracted_basis(coefficients, primitives)

# utility

Base.string(b::Basis) = "$(typeof(b))(" * join(["$(symbol)=$(getproperty(b,symbol))" for symbol in fieldnames(typeof(b))], ", ") * ")"
Base.string(bs::BasisSet) = "BasisSet(" * join(["$(b)" for b in bs.basis], ", ") * ")"
Base.show(io::IO, b::Basis) = print(io, Base.string(b))
Base.show(io::IO, bs::BasisSet) = print(io, Base.string(bs))
@inline function Base.getproperty(b::ContractedBasis, name::Symbol)
  name === :c && return getfield(b, :coefficients)
  name === :φ && return getfield(b, :primitives)
  return getfield(b, name)
end
Base.propertynames(::ContractedBasis, ::Bool=false) = (:coefficients, :primitives, :c, :φ)
Base.getindex(BS::BasisSet, index) = BS.basis[index]
Base.getindex(GBS::GeometricBasisSet, index) = GBS.basis[index]
Base.length(BS::BasisSet) = length(BS.basis)
Base.length(GBS::GeometricBasisSet) = length(GBS.basis)
Base.length(::ContractedBasis{N}) where {N} = N

# function

φ(b::SimpleGaussianBasis, r) = exp(-b.a*r^2)
φ(b::GaussianBasis, r, θ, φ) = N(b.l) * r^b.l * exp(-b.a*r^2) * Y(b.l, b.m, θ, φ)
@inline φ(b::ContractedBasis, coordinates...) = _contracted_value(
  getfield(b, :coefficients),
  getfield(b, :primitives),
  coordinates,
)
@inline _contracted_value(coefficients::Tuple{T}, primitives::Tuple{P}, coordinates) where {T,P} =
  first(coefficients) * φ(first(primitives), coordinates...)
@inline _contracted_value(coefficients::Tuple, primitives::Tuple, coordinates) = muladd(
  first(coefficients),
  φ(first(primitives), coordinates...),
  _contracted_value(Base.tail(coefficients), Base.tail(primitives), coordinates),
)

_replace_exponent(b::SimpleGaussianBasis, a) = SimpleGaussianBasis(a)
_replace_exponent(b::GaussianBasis, a) = GaussianBasis(a=a, l=b.l, m=b.m)
_replace_exponent(b::Basis, a) = typeof(b)(a)

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

The concrete element type is preserved when all basis functions have a common
type, avoiding abstract `Basis` dispatch in matrix-construction loops.

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
\phi_i(r, θ, φ) = N _{il} r^l \exp(-a_i r^2) Y_l^m(θ, φ)
```
""" GaussianBasis

@doc raw"""
`ContractedBasis([c1, c2, ...], [primitive1, primitive2, ...])`
```math
\phi' = \sum_i c_i \phi_i
```

A contracted basis is a nonempty linear combination of `PrimitiveBasis`
objects. The numbers of coefficients and primitive bases must match. Numeric
coefficient types are promoted to a common type.

The components are stored in tuples, making their number part of the type and
preserving the concrete type of every primitive basis. This lets Julia
specialize and inline evaluation without dispatching through an abstract
`PrimitiveBasis` container. Tuple inputs are recommended when constructing a
basis in performance-sensitive code; vectors are also accepted and converted
once during construction.

The fields are named `coefficients` and `primitives`. The aliases `c` and `φ`
are retained for compatibility.
""" ContractedBasis
