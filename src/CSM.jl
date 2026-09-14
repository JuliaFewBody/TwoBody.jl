export ComplexScalingMethod, ResultComplexScaling

import LinearAlgebra
import Subscripts

struct ComplexScalingMethod{B<:Union{BasisSet,GeometricBasisSet},T<:Real}
  basisset::B
  θ::T
  function ComplexScalingMethod(basisset::B; θ::T=zero(Float64)) where {B<:Union{BasisSet,GeometricBasisSet},T<:Real}
    0 ≤ θ < π / 2 || throw(ArgumentError("θ must satisfy 0 ≤ θ < π/2"))
    new{B,T}(basisset, θ)
  end
end

ComplexScalingMethod(basis::Basis; θ=zero(Float64)) =
  ComplexScalingMethod(BasisSet(basis); θ=θ)

struct ResultComplexScaling
  data::Any
  ResultComplexScaling(; args...) = new(NamedTuple(Dict(args)))
end

Base.getproperty(result::ResultComplexScaling, symbol::Symbol) =
  Base.getproperty(getfield(result, :data), symbol)
Base.show(io::IO, method::ComplexScalingMethod) = print(io, Base.string(method))
Base.show(io::IO, result::ResultComplexScaling) = print(io, Base.string(result))

Base.string(method::ComplexScalingMethod) =
  "ComplexScalingMethod(θ=$(method.θ), basisset=$(method.basisset))"

function Base.string(result::ResultComplexScaling)
  text = "# method\n\n$(result.method)\n\n# eigenvalue\n\n"
  for n in eachindex(result.E)
    text *= "E$(Subscripts.sub(string(n))) = $(result.E[n])\n"
  end
  return text
end

function ψ(result::ResultComplexScaling, r; n::Int=1)
  return sum(result.C[i,n] * φ(result.basisset[i], r) for i in 1:result.nₘₐₓ)
end

_complex_scaled(o::RestEnergy, θ) = (one(ComplexF64), o)
_complex_scaled(o::Laplacian, θ) = (cis(-2θ), o)
_complex_scaled(o::Kinetic, θ) = (cis(-2θ), o)
_complex_scaled(o::RelativisticCorrection, θ) = (cis(-2o.n * θ), o)
_complex_scaled(o::Constant, θ) = (one(ComplexF64), o)
_complex_scaled(o::Linear, θ) = (cis(θ), o)
_complex_scaled(o::Coulomb, θ) = (cis(-θ), o)
_complex_scaled(o::PowerLaw, θ) = (cis(o.exponent * θ), o)
_complex_scaled(o::Gaussian, θ) =
  (one(ComplexF64), Gaussian(coefficient=o.coefficient, exponent=o.exponent * cis(2θ)))
_complex_scaled(o::Exponential, θ) =
  (one(ComplexF64), Exponential(coefficient=o.coefficient, exponent=o.exponent * cis(θ)))
_complex_scaled(o::Yukawa, θ) =
  (cis(-θ), Yukawa(coefficient=o.coefficient, exponent=o.exponent * cis(θ)))
_complex_scaled(o::Delta, θ) = (cis(-3θ), o)
_complex_scaled(o::Custom, θ) = (one(ComplexF64), Custom(f=r -> o.f(r * cis(θ))))

_complex_scaled(::Tabulated, θ) =
  throw(ArgumentError("complex scaling requires an analytic potential"))
_complex_scaled(o::RelativisticKinetic, θ) =
  throw(ArgumentError("complex scaling does not support $(typeof(o))"))

function _complex_scaled(o::Operator, θ)
  throw(ArgumentError("complex scaling does not support $(typeof(o))"))
end

_basis(method::ComplexScalingMethod) = method.basisset isa GeometricBasisSet ?
  BasisSet(method.basisset.basis...) : method.basisset

function matrix(operator::Operator, method::ComplexScalingMethod)
  basisset = _basis(method)
  factor, scaled = _complex_scaled(operator, method.θ)
  nₘₐₓ = length(basisset)
  M = Matrix{ComplexF64}(undef, nₘₐₓ, nₘₐₓ)
  for j in 1:nₘₐₓ
    for i in 1:j
      value = factor * element(scaled, basisset[i], basisset[j])
      M[i,j] = value
      M[j,i] = value
    end
  end
  return M
end

matrix(hamiltonian::Hamiltonian, method::ComplexScalingMethod) =
  sum(matrix(term, method) for term in hamiltonian.terms)

function solve(hamiltonian::Hamiltonian, method::ComplexScalingMethod)
  basisset = _basis(method)
  S = Matrix(matrix(basisset))
  H = matrix(hamiltonian, method)
  decomposition = LinearAlgebra.eigen(H, S)
  order = sortperm(decomposition.values; by=real)
  E = decomposition.values[order]
  C = decomposition.vectors[:,order]
  return ResultComplexScaling(;
    hamiltonian, method, basisset, nₘₐₓ=length(basisset), H, S, E, C,
  )
end

@doc raw"""
`ComplexScalingMethod(basisset; θ=0.0)`

Configure complex scaling by the angle ``θ``.
""" ComplexScalingMethod

@doc raw"""
`ResultComplexScaling`

Result of a complex-scaled generalized eigenvalue problem.
""" ResultComplexScaling

@doc raw"""
`solve(hamiltonian, method::ComplexScalingMethod)`

Solve the complex-scaled Hamiltonian in the configured basis.
""" solve(hamiltonian::Hamiltonian, method::ComplexScalingMethod)
