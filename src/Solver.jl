export SolverMethod

import LinearAlgebra

# type

"""
    SolverMethod

Abstract supertype of the variational solver algorithms. A method is a small
struct holding the algorithm-level options, such as the basis size or the
number of candidates per step. Problem-level options (`perturbation`, `info`,
`rng`) belong to [`solve`](@ref).

A new algorithm is added by defining a subtype and the corresponding
`solve(hamiltonian::Hamiltonian, basis, method::NewMethod; ...)` method.
[`BayesianVariationalMethod`](@ref), [`StochasticVariationalMethod`](@ref),
and [`GradientVariationalMethod`](@ref) share this interface: every method
takes a Hamiltonian and a basis (a candidate set, an initial set, or a
prototype), and returns a `ResultRayleighRitz` extended with the search
history of the method.
"""
abstract type SolverMethod end

# utility

Base.string(method::SolverMethod) =
  "$(nameof(typeof(method)))(" *
  join(
    ("$(name)=$(getproperty(method, name))" for name in fieldnames(typeof(method))),
    ", ",
  ) *
  ")"
Base.show(io::IO, method::SolverMethod) = print(io, Base.string(method))

# The nonlinear parameter optimized by the variational methods. It is the
# counterpart of `_replace_exponent(basis, a)`.
_exponent(basis::Basis) = basis.a

# shared evaluation of a trial basis set

_invalid_evaluation(reason::Symbol, message::AbstractString) = (
  valid=false,
  energy=Inf,
  coefficients=Float64[],
  reason=reason,
  message=String(message),
)

function _is_expected_numerical_error(error)
  return error isa ArgumentError ||
         error isa DomainError ||
         error isa LinearAlgebra.PosDefException ||
         error isa LinearAlgebra.SingularException ||
         error isa LinearAlgebra.LAPACKException
end

"""
    TwoBody._variational_energy(hamiltonian, basisset; overlap_tol=1e-10, reference=nothing)

Evaluate the lowest Rayleigh-Ritz eigenpair of `basisset` and report whether it
is admissible. The named tuple `(valid, energy, coefficients, reason, message)`
carries the reason of the rejection instead of an exception, so that a search
algorithm can score many trial basis sets without aborting.

A basis set is rejected when a matrix element is non-finite
(`:nonfinite_matrix`), when an overlap diagonal entry is not positive
(`:nonpositive_overlap_diagonal`), when the smallest eigenvalue of the
normalized overlap matrix falls below `overlap_tol`
(`:ill_conditioned_overlap`), when the eigenvalue is non-finite
(`:nonfinite_energy`), when a `reference` energy is given and the energy
exceeds it (`:energy_increase`), or when the linear algebra fails
(`:numerical_failure`). The returned `coefficients` are normalized as
``\\pmb{c}^\\top \\pmb{S} \\pmb{c} = 1``.
"""
function _variational_energy(
  hamiltonian::Hamiltonian,
  basisset::BasisSet;
  overlap_tol::Real=1e-10,
  reference::Union{Nothing,Real}=nothing,
)
  try
    overlap = matrix(basisset)
    hamiltonian_matrix = matrix(hamiltonian, basisset)
    if !all(isfinite, overlap) || !all(isfinite, hamiltonian_matrix)
      return _invalid_evaluation(
        :nonfinite_matrix, "Hamiltonian or overlap matrix is non-finite",
      )
    end

    diagonal = LinearAlgebra.diag(overlap)
    if any(value -> !isfinite(value) || value <= 0, diagonal)
      return _invalid_evaluation(
        :nonpositive_overlap_diagonal,
        "overlap diagonal entries must be positive and finite",
      )
    end

    scale = LinearAlgebra.Diagonal(inv.(sqrt.(diagonal)))
    normalized_overlap = LinearAlgebra.Symmetric(scale * overlap * scale)
    minimum_overlap = LinearAlgebra.eigmin(normalized_overlap)
    if minimum_overlap < overlap_tol
      return _invalid_evaluation(
        :ill_conditioned_overlap,
        "normalized overlap eigenvalue $(minimum_overlap) is below $(overlap_tol)",
      )
    end

    decomposition = LinearAlgebra.eigen(hamiltonian_matrix, overlap)
    energy = first(decomposition.values)
    isfinite(energy) || return _invalid_evaluation(
      :nonfinite_energy, "generalized eigenvalue is non-finite",
    )

    coefficients = decomposition.vectors[:,1]
    coefficients ./= sqrt(coefficients' * overlap * coefficients)

    if !isnothing(reference)
      tolerance = 100eps(Float64) * max(1, abs(reference))
      energy <= reference + tolerance || return _invalid_evaluation(
        :energy_increase,
        "candidate energy $(energy) exceeds current energy $(reference)",
      )
    end

    return (
      valid=true,
      energy=energy,
      coefficients=coefficients,
      reason=nothing,
      message="",
    )
  catch error
    _is_expected_numerical_error(error) || rethrow()
    return _invalid_evaluation(:numerical_failure, sprint(showerror, error))
  end
end
