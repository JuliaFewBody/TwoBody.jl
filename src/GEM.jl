export φp, ψp

import LinearAlgebra
import SpecialFunctions

# Generalized Gauss–Laguerre nodes and weights for x^α exp(-x). The only
# non-analytic GEM matrix elements reuse these angular-momentum-specific grids.
const _gausslaguerre_cache = Dict{Tuple{Float64,Int},Tuple{Vector{Float64},Vector{Float64}}}()
const _gem_quadrature_order = 128

function _gausslaguerre(α::Real, n::Int=_gem_quadrature_order)
  α > -1 || throw(ArgumentError("Gauss–Laguerre quadrature requires α > -1"))
  key = (Float64(α), n)
  return get!(_gausslaguerre_cache, key) do
    k = collect(1.0:n)
    diagonal = 2 .* k .- 1 .+ α
    indices = collect(1.0:n-1)
    offdiagonal = sqrt.(indices .* (indices .+ α))
    decomposition = LinearAlgebra.eigen(LinearAlgebra.SymTridiagonal(diagonal, offdiagonal))
    weights = SpecialFunctions.gamma(α + 1) .* abs2.(decomposition.vectors[1, :])
    (decomposition.values, weights)
  end
end

function _check_gaussian_basis(b::GaussianBasis)
  b.a > 0 || throw(ArgumentError("Gaussian exponent a must be positive"))
  b.l isa Integer && b.l ≥ 0 || throw(ArgumentError("l must be a nonnegative integer"))
  b.m isa Integer && abs(b.m) ≤ b.l || throw(ArgumentError("m must be an integer with |m| ≤ l"))
  return b
end

function _check_complex_gaussian_basis(b::ComplexGaussianBasis)
  b.a > 0 || throw(ArgumentError("Gaussian exponent a must be positive"))
  b.ω > 0 || throw(ArgumentError("complex-range parameter ω must be positive"))
  b.component in (:cos, :sin) ||
    throw(ArgumentError("component must be :cos or :sin"))
  b.l isa Integer && b.l ≥ 0 || throw(ArgumentError("l must be a nonnegative integer"))
  b.m isa Integer && abs(b.m) ≤ b.l ||
    throw(ArgumentError("m must be an integer with |m| ≤ l"))
  return b
end

function _gaussian_normalization(a::Real, l::Integer)
  a > 0 || throw(ArgumentError("Gaussian exponent a must be positive"))
  l ≥ 0 || throw(ArgumentError("l must be nonnegative"))
  q = l + 3/2
  return sqrt(2 * (2 * a)^q / SpecialFunctions.gamma(q))
end

function _associated_legendre(l::Integer, m::Integer, x::Real)
  0 ≤ m ≤ l || throw(ArgumentError("associated Legendre indices require 0 ≤ m ≤ l"))
  pmm = one(float(x))
  if m > 0
    root = sqrt(max(zero(float(x)), one(float(x)) - x*x))
    factor = one(float(x))
    for _ in 1:m
      pmm *= -factor * root
      factor += 2
    end
  end
  l == m && return pmm
  pmmp1 = x * (2 * m + 1) * pmm
  l == m + 1 && return pmmp1
  previous, current = pmm, pmmp1
  for degree in m+2:l
    following = ((2 * degree - 1) * x * current - (degree + m - 1) * previous) / (degree - m)
    previous, current = current, following
  end
  return current
end

function _spherical_harmonic(l::Integer, m::Integer, θ::Real, ϕ::Real)
  l ≥ 0 || throw(ArgumentError("l must be nonnegative"))
  abs(m) ≤ l || throw(ArgumentError("m must satisfy |m| ≤ l"))
  if m < 0
    return (-1)^(-m) * conj(_spherical_harmonic(l, -m, θ, ϕ))
  end
  ratio = SpecialFunctions.gamma(l - m + 1) / SpecialFunctions.gamma(l + m + 1)
  normalization = sqrt((2 * l + 1) / (4π) * ratio)
  return normalization * _associated_legendre(l, m, cos(θ)) * exp(im * m * ϕ)
end

"""Return the normalized radial part of a Gaussian GEM primitive."""
function φ(b::GaussianBasis, r)
  _check_gaussian_basis(b)
  return _gaussian_normalization(b.a, b.l) * r^b.l * exp(-b.a * r^2)
end

"""Return a normalized Gaussian GEM primitive in position space."""
φ(b::GaussianBasis, r, θ, ϕ) = φ(b, r) * _spherical_harmonic(b.l, b.m, θ, ϕ)

function _complex_terms(b::ComplexGaussianBasis)
  _check_complex_gaussian_basis(b)
  exponents = (b.a * (1 + im * b.ω), b.a * (1 - im * b.ω))
  coefficients = b.component === :cos ? (1/2, 1/2) : (im/2, -im/2)
  return zip(exponents, coefficients)
end

function _complex_pair_sum(f, b1::ComplexGaussianBasis, b2::ComplexGaussianBasis)
  return sum(
    conj(c1) * c2 * f(conj(η1) + η2, conj(η1), η2)
    for (η1, c1) in _complex_terms(b1), (η2, c2) in _complex_terms(b2)
  )
end

function _complex_raw_moment(b1::ComplexGaussianBasis, b2::ComplexGaussianBasis, power)
  q = b1.l + (power + 3) / 2
  q > 0 || throw(ArgumentError("radial matrix element does not converge"))
  return _complex_pair_sum((z, _, _) -> SpecialFunctions.gamma(q) / (2 * z^q), b1, b2)
end

function _complex_gaussian_normalization(b::ComplexGaussianBasis)
  norm² = real(_complex_raw_moment(b, b, 0))
  norm² > 0 || throw(ArgumentError("complex-range primitive has zero norm"))
  return inv(sqrt(norm²))
end

"""Return the normalized radial part of a real complex-range Gaussian primitive."""
function φ(b::ComplexGaussianBasis, r)
  normalization = _complex_gaussian_normalization(b)
  oscillation = b.component === :cos ? cos(b.ω * b.a * r^2) : sin(b.ω * b.a * r^2)
  return normalization * r^b.l * exp(-b.a * r^2) * oscillation
end

φ(b::ComplexGaussianBasis, r, θ, ϕ) =
  φ(b, r) * _spherical_harmonic(b.l, b.m, θ, ϕ)

"""Return the radial momentum-space Gaussian primitive (unitary Fourier convention)."""
function φp(b::GaussianBasis, p)
  _check_gaussian_basis(b)
  q = b.l + 3/2
  return im^b.l * _gaussian_normalization(b.a, b.l) * p^b.l * exp(-p^2 / (4 * b.a)) / (2 * b.a)^q
end

"""Return the full momentum-space Gaussian primitive."""
φp(b::GaussianBasis, p, θ, ϕ) = φp(b, p) * _spherical_harmonic(b.l, b.m, θ, ϕ)

"""Evaluate a Rayleigh–Ritz eigenfunction in momentum space."""
function ψp(result::ResultRayleighRitz, p; n::Int=1)
  return sum(result.C[i, n] * φp(result.basisset[i], p) for i in 1:result.nₘₐₓ)
end

function ψp(result::ResultRayleighRitz, p, θ, ϕ; n::Int=1)
  return sum(result.C[i, n] * φp(result.basisset[i], p, θ, ϕ) for i in 1:result.nₘₐₓ)
end

function ψ(result::ResultRayleighRitz, r, θ, ϕ; n::Int=1)
  return sum(result.C[i, n] * φ(result.basisset[i], r, θ, ϕ) for i in 1:result.nₘₐₓ)
end

_same_channel(b1::GaussianBasis, b2::GaussianBasis) = b1.l == b2.l && b1.m == b2.m
_same_channel(b1::ComplexGaussianBasis, b2::ComplexGaussianBasis) =
  b1.l == b2.l && b1.m == b2.m

function _normalization_product(b1::GaussianBasis, b2::GaussianBasis)
  _check_gaussian_basis(b1)
  _check_gaussian_basis(b2)
  return _gaussian_normalization(b1.a, b1.l) * _gaussian_normalization(b2.a, b2.l)
end

function element(b1::GaussianBasis, b2::GaussianBasis)
  _same_channel(b1, b2) || return 0.0
  q = b1.l + 3/2
  return _normalization_product(b1, b2) * SpecialFunctions.gamma(q) / (2 * (b1.a + b2.a)^q)
end

function _central_element(f, b1::GaussianBasis, b2::GaussianBasis)
  _same_channel(b1, b2) || return 0.0
  q = b1.l + 3/2
  a = b1.a + b2.a
  nodes, weights = _gausslaguerre(q - 1)
  integral = sum(weights .* f.(sqrt.(nodes ./ a))) / (2 * a^q)
  return _normalization_product(b1, b2) * integral
end

element(o::RestEnergy, b1::GaussianBasis, b2::GaussianBasis) = o.m * o.c^2 * element(b1, b2)

function element(o::Laplacian, b1::GaussianBasis, b2::GaussianBasis)
  _same_channel(b1, b2) || return 0.0
  return -2 * o.coefficient * (2 * b1.l + 3) * b1.a * b2.a / (b1.a + b2.a) * element(b1, b2)
end

function element(o::Kinetic, b1::GaussianBasis, b2::GaussianBasis)
  _same_channel(b1, b2) || return 0.0
  return o.hbar^2 / o.m * (2 * b1.l + 3) * b1.a * b2.a / (b1.a + b2.a) * element(b1, b2)
end

element(o::Constant, b1::GaussianBasis, b2::GaussianBasis) = o.constant * element(b1, b2)

function element(o::PowerLaw, b1::GaussianBasis, b2::GaussianBasis)
  _same_channel(b1, b2) || return 0.0
  q = b1.l + (o.exponent + 3) / 2
  return o.coefficient * _normalization_product(b1, b2) * SpecialFunctions.gamma(q) / (2 * (b1.a + b2.a)^q)
end

element(o::Linear, b1::GaussianBasis, b2::GaussianBasis) = element(PowerLaw(o.coefficient, 1), b1, b2)
element(o::Coulomb, b1::GaussianBasis, b2::GaussianBasis) = element(PowerLaw(o.coefficient, -1), b1, b2)

function element(o::Gaussian, b1::GaussianBasis, b2::GaussianBasis)
  _same_channel(b1, b2) || return 0.0
  q = b1.l + 3/2
  return o.coefficient * _normalization_product(b1, b2) * SpecialFunctions.gamma(q) / (2 * (b1.a + b2.a + o.exponent)^q)
end

function _exp_gaussian_integral(n::Integer, σ::Number, a::Real)
  n ≥ 0 || throw(ArgumentError("integral power must be nonnegative"))
  a > 0 || throw(ArgumentError("Gaussian exponent must be positive"))
  i₀ = sqrt(π) * SpecialFunctions.erfcx(σ / (2 * sqrt(a))) / (2 * sqrt(a))
  n == 0 && return i₀
  i₁ = (1 - σ * i₀) / (2 * a)
  n == 1 && return i₁
  previous, current = i₀, i₁
  for power in 2:n
    following = ((power - 1) * previous - σ * current) / (2 * a)
    previous, current = current, following
  end
  return current
end

function element(o::Exponential, b1::GaussianBasis, b2::GaussianBasis)
  _same_channel(b1, b2) || return 0.0
  radial = _exp_gaussian_integral(2 * b1.l + 2, o.exponent, b1.a + b2.a)
  return o.coefficient * _normalization_product(b1, b2) * radial
end


function element(o::Yukawa, b1::GaussianBasis, b2::GaussianBasis)
  _same_channel(b1, b2) || return 0.0
  radial = _exp_gaussian_integral(2 * b1.l + 1, o.exponent, b1.a + b2.a)
  return o.coefficient * _normalization_product(b1, b2) * radial
end

element(o::Custom, b1::GaussianBasis, b2::GaussianBasis) = _central_element(o.f, b1, b2)

function element(o::Delta, b1::GaussianBasis, b2::GaussianBasis)
  _same_channel(b1, b2) || return 0.0
  b1.l == 0 || return 0.0
  return o.coefficient * _normalization_product(b1, b2) / (4π)
end

function _momentum_moment(n::Integer, b1::GaussianBasis, b2::GaussianBasis)
  _same_channel(b1, b2) || return 0.0
  n ≥ 0 || throw(ArgumentError("moment order must be nonnegative"))
  q = b1.l + 3/2
  a = (inv(b1.a) + inv(b2.a)) / 4
  prefactor = _normalization_product(b1, b2) / (4 * b1.a * b2.a)^q
  return prefactor * SpecialFunctions.gamma(q + n) / (2 * a^(q + n))
end

function _sqrt_binomial(n::Integer)
  n ≥ 0 || throw(ArgumentError("expansion order must be nonnegative"))
  coefficient = 1.0
  for k in 1:n
    coefficient *= (3/2 - k) / k
  end
  return coefficient
end

function element(o::RelativisticCorrection, b1::GaussianBasis, b2::GaussianBasis)
  o.n ≥ 1 || throw(ArgumentError("RelativisticCorrection requires n ≥ 1"))
  coefficient = _sqrt_binomial(o.n) / (o.m^(2 * o.n - 1) * o.c^(2 * o.n - 2))
  return coefficient * _momentum_moment(o.n, b1, b2)
end

function element(o::RelativisticKinetic, b1::GaussianBasis, b2::GaussianBasis)
  _same_channel(b1, b2) || return 0.0
  q = b1.l + 3/2
  a = (inv(b1.a) + inv(b2.a)) / 4
  nodes, weights = _gausslaguerre(q - 1)
  p² = nodes ./ a
  mc² = o.m * o.c^2
  kinetic = p² .* o.c^2 ./ (sqrt.(o.m^2 * o.c^4 .+ p² .* o.c^2) .+ mc²)
  radial = sum(weights .* kinetic) / (2 * a^q)
  prefactor = _normalization_product(b1, b2) / (4 * b1.a * b2.a)^q
  return prefactor * radial
end

function _complex_normalization_product(
  b1::ComplexGaussianBasis,
  b2::ComplexGaussianBasis,
)
  return _complex_gaussian_normalization(b1) * _complex_gaussian_normalization(b2)
end

function element(b1::ComplexGaussianBasis, b2::ComplexGaussianBasis)
  _same_channel(b1, b2) || return 0.0
  value = _complex_normalization_product(b1, b2) * _complex_raw_moment(b1, b2, 0)
  return real(value)
end

element(o::RestEnergy, b1::ComplexGaussianBasis, b2::ComplexGaussianBasis) =
  o.m * o.c^2 * element(b1, b2)

function _complex_kinetic_element(b1::ComplexGaussianBasis, b2::ComplexGaussianBasis)
  _same_channel(b1, b2) || return 0.0
  q = b1.l + 3/2
  raw = _complex_pair_sum(
    (z, η1, η2) -> (2 * b1.l + 3) * η1 * η2 / z *
                    SpecialFunctions.gamma(q) / (2 * z^q),
    b1,
    b2,
  )
  return real(_complex_normalization_product(b1, b2) * raw)
end

element(o::Laplacian, b1::ComplexGaussianBasis, b2::ComplexGaussianBasis) =
  -2 * o.coefficient * _complex_kinetic_element(b1, b2)

element(o::Kinetic, b1::ComplexGaussianBasis, b2::ComplexGaussianBasis) =
  o.hbar^2 / o.m * _complex_kinetic_element(b1, b2)

element(o::Constant, b1::ComplexGaussianBasis, b2::ComplexGaussianBasis) =
  o.constant * element(b1, b2)

function element(o::PowerLaw, b1::ComplexGaussianBasis, b2::ComplexGaussianBasis)
  _same_channel(b1, b2) || return 0.0
  value = o.coefficient * _complex_normalization_product(b1, b2) *
          _complex_raw_moment(b1, b2, o.exponent)
  return real(value)
end

element(o::Linear, b1::ComplexGaussianBasis, b2::ComplexGaussianBasis) =
  element(PowerLaw(o.coefficient, 1), b1, b2)
element(o::Coulomb, b1::ComplexGaussianBasis, b2::ComplexGaussianBasis) =
  element(PowerLaw(o.coefficient, -1), b1, b2)

function element(o::Gaussian, b1::ComplexGaussianBasis, b2::ComplexGaussianBasis)
  _same_channel(b1, b2) || return 0.0
  q = b1.l + 3/2
  raw = _complex_pair_sum(
    (z, _, _) -> SpecialFunctions.gamma(q) / (2 * (z + o.exponent)^q),
    b1,
    b2,
  )
  return real(o.coefficient * _complex_normalization_product(b1, b2) * raw)
end

@doc raw"""
`φp(b::GaussianBasis, p)` returns the radial momentum-space primitive. With
the unitary Fourier convention

```math
\widetilde\phi(\boldsymbol p)=(2\pi)^{-3/2}\int
e^{i\boldsymbol p\cdot\boldsymbol r}\phi(\boldsymbol r)\,d^3r,
```

the normalized position-space primitive
``N_l r^l e^{-a r^2}Y_l^m(\hat{\boldsymbol r})`` becomes

```math
\widetilde\phi_{alm}(\boldsymbol p)=
i^l N_l\frac{p^l}{(2a)^{l+3/2}}e^{-p^2/(4a)}
Y_l^m(\hat{\boldsymbol p}).
```

The two-argument method returns only its radial factor; the four-argument
method `φp(b, p, θ, ϕ)` includes the spherical harmonic. Position and momentum
must use reciprocal units (for example, GeV⁻¹ and GeV when `ℏ=c=1`).
""" φp

@doc raw"""
`ψp(result, p; n=1)` evaluates a Gaussian-expansion eigenfunction in momentum
space by summing `result.C[i,n] * φp(result.basisset[i], p)`. It therefore
uses the same unitary Fourier convention as `φp`. The four-argument form
`ψp(result, p, θ, ϕ; n=1)` also includes the spherical harmonic.

This method currently applies to results built from real-range
`GaussianBasis` primitives. Complex-range cos/sin primitives are intended for
the position-space Rayleigh–Ritz calculation.
""" ψp
