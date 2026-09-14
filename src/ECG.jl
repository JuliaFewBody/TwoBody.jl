export ExplicitlyCorrelatedGaussianBasis, ECGBasis, ECGKinetic, ECGCoulomb

import LinearAlgebra
import QuadGK

struct ExplicitlyCorrelatedGaussianBasis <: PrimitiveBasis
  A::Matrix{Float64}
  prefactors::Vector{Vector{Float64}}
  function ExplicitlyCorrelatedGaussianBasis(A::AbstractMatrix; prefactors=())
    size(A, 1) == size(A, 2) || throw(ArgumentError("A must be square"))
    matrix = Matrix{Float64}(A)
    LinearAlgebra.issymmetric(matrix) || throw(ArgumentError("A must be symmetric"))
    LinearAlgebra.isposdef(matrix) || throw(ArgumentError("A must be positive definite"))
    vectors = [Vector{Float64}(vector) for vector in prefactors]
    length(vectors) ≤ 2 || throw(ArgumentError("at most two prefactors are supported"))
    dimension = 3 * size(matrix, 1)
    all(length(vector) == dimension for vector in vectors) ||
      throw(DimensionMismatch("each prefactor must have length $dimension"))
    new(matrix, vectors)
  end
end

const ECGBasis = ExplicitlyCorrelatedGaussianBasis

struct ECGKinetic{T<:Real} <: KineticTerm
  K::Matrix{T}
  function ECGKinetic(K::AbstractMatrix{T}) where {T<:Real}
    size(K, 1) == size(K, 2) || throw(ArgumentError("K must be square"))
    LinearAlgebra.issymmetric(K) || throw(ArgumentError("K must be symmetric"))
    new{T}(Matrix(K))
  end
end

struct ECGCoulomb{T<:Real,U<:Real} <: PotentialTerm
  coefficient::T
  w::Vector{U}
end

ECGCoulomb(coefficient::Real, w::AbstractVector{<:Real}) =
  ECGCoulomb(coefficient, collect(w))

Base.string(basis::ExplicitlyCorrelatedGaussianBasis) =
  "ECGBasis(A=$(basis.A), rank=$(length(basis.prefactors)))"

function _ecg_coordinates(coordinates::AbstractMatrix, n::Int)
  size(coordinates) == (n, 3) || throw(DimensionMismatch("coordinates must have size ($n, 3)"))
  return vec(permutedims(coordinates))
end

function φ(basis::ExplicitlyCorrelatedGaussianBasis, coordinates::AbstractVector)
  n = size(basis.A, 1)
  length(coordinates) == 3 * n || throw(DimensionMismatch("coordinates must have length $(3 * n)"))
  A = LinearAlgebra.kron(basis.A, Matrix{Float64}(LinearAlgebra.I, 3, 3))
  return prod(LinearAlgebra.dot(vector, coordinates) for vector in basis.prefactors;
              init=1.0) * exp(-LinearAlgebra.dot(coordinates, A * coordinates))
end

φ(basis::ExplicitlyCorrelatedGaussianBasis, coordinates::AbstractMatrix) =
  φ(basis, _ecg_coordinates(coordinates, size(basis.A, 1)))

function _ecg_parameters(bra::ExplicitlyCorrelatedGaussianBasis,
                         ket::ExplicitlyCorrelatedGaussianBasis)
  size(bra.A) == size(ket.A) || throw(DimensionMismatch("correlation matrices must have the same size"))
  n = size(ket.A, 1)
  R = inv(bra.A + ket.A)
  covariance = LinearAlgebra.kron(R / 2, Matrix{Float64}(LinearAlgebra.I, 3, 3))
  M₀ = (pi^n / LinearAlgebra.det(bra.A + ket.A))^(3/2)
  return (; n, R, covariance, M₀)
end

function _ecg_moment(vectors, covariance)
  isempty(vectors) && return 1.0
  isodd(length(vectors)) && return 0.0
  first_vector = first(vectors)
  value = 0.0
  for index in 2:length(vectors)
    remaining = [vectors[i] for i in eachindex(vectors) if i != 1 && i != index]
    value += LinearAlgebra.dot(first_vector, covariance * vectors[index]) *
             _ecg_moment(remaining, covariance)
  end
  return value
end

function _ecg_quadratic_moment(matrix, vectors, covariance)
  value = LinearAlgebra.tr(matrix * covariance) * _ecg_moment(vectors, covariance)
  for i in eachindex(vectors), j in eachindex(vectors)
    i == j && continue
    remaining = [vectors[k] for k in eachindex(vectors) if k != i && k != j]
    value += LinearAlgebra.dot(covariance * vectors[i], matrix * covariance * vectors[j]) *
             _ecg_moment(remaining, covariance)
  end
  return value
end

function element(bra::ExplicitlyCorrelatedGaussianBasis,
                 ket::ExplicitlyCorrelatedGaussianBasis)
  parameters = _ecg_parameters(bra, ket)
  return parameters.M₀ * _ecg_moment([bra.prefactors; ket.prefactors], parameters.covariance)
end

function element(operator::ECGKinetic, bra::ExplicitlyCorrelatedGaussianBasis,
                 ket::ExplicitlyCorrelatedGaussianBasis)
  parameters = _ecg_parameters(bra, ket)
  size(operator.K) == (parameters.n, parameters.n) ||
    throw(DimensionMismatch("K must match the correlation matrices"))
  identity3 = Matrix{Float64}(LinearAlgebra.I, 3, 3)
  A = LinearAlgebra.kron(ket.A, identity3)
  B = LinearAlgebra.kron(bra.A, identity3)
  K = LinearAlgebra.kron(operator.K, identity3)
  value = 0.0

  for (i, a) in pairs(ket.prefactors), (j, b) in pairs(bra.prefactors)
    vectors = [bra.prefactors[k] for k in eachindex(bra.prefactors) if k != j]
    append!(vectors, [ket.prefactors[k] for k in eachindex(ket.prefactors) if k != i])
    value += LinearAlgebra.dot(b, K * a) * _ecg_moment(vectors, parameters.covariance)
  end
  for (j, b) in pairs(bra.prefactors)
    vectors = [bra.prefactors[k] for k in eachindex(bra.prefactors) if k != j]
    append!(vectors, ket.prefactors)
    value -= 2 * _ecg_moment([A * K * b, vectors...], parameters.covariance)
  end
  for (i, a) in pairs(ket.prefactors)
    vectors = copy(bra.prefactors)
    append!(vectors, [ket.prefactors[k] for k in eachindex(ket.prefactors) if k != i])
    value -= 2 * _ecg_moment([B * K * a, vectors...], parameters.covariance)
  end
  vectors = [bra.prefactors; ket.prefactors]
  value += 4 * _ecg_quadratic_moment(B * K * A, vectors, parameters.covariance)
  return parameters.M₀ * value
end

function element(operator::ECGCoulomb, bra::ExplicitlyCorrelatedGaussianBasis,
                 ket::ExplicitlyCorrelatedGaussianBasis)
  parameters = _ecg_parameters(bra, ket)
  length(operator.w) == parameters.n || throw(DimensionMismatch("w must match the correlation matrices"))
  vectors = [bra.prefactors; ket.prefactors]
  if isempty(vectors)
    β = inv(LinearAlgebra.dot(operator.w, parameters.R * operator.w))
    return operator.coefficient * 2 * sqrt(β / pi) * parameters.M₀
  end
  integral, _ = QuadGK.quadgk(0.0, Inf; rtol=1e-10) do t
    matrix = bra.A + ket.A + t^2 * operator.w * transpose(operator.w)
    covariance = LinearAlgebra.kron(inv(matrix) / 2,
      Matrix{Float64}(LinearAlgebra.I, 3, 3))
    M₀ = (pi^parameters.n / LinearAlgebra.det(matrix))^(3/2)
    return M₀ * _ecg_moment(vectors, covariance)
  end
  return operator.coefficient * 2 / sqrt(pi) * integral
end

@doc raw"""
`ExplicitlyCorrelatedGaussianBasis(A; prefactors=())`

Construct an ECG basis with up to two prefactor vectors. `ECGBasis` is an alias.
""" ExplicitlyCorrelatedGaussianBasis

@doc raw"""
`ECGKinetic(K)`

Construct ``-\partial_r K\partial_r^T`` for ECG matrix elements.
""" ECGKinetic

@doc raw"""
`ECGCoulomb(coefficient, w)`

Construct ``\mathrm{coefficient}/|w^T r|`` for ECG matrix elements.
""" ECGCoulomb
