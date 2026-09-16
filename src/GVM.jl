export GradientVariationalMethod, GVM

import ForwardDiff
import Optim

# type

"""
    GradientVariationalMethod(; max_basis=nothing, amin=1e-4, amax=1e+4,
                              optimizer=Optim.LBFGS(), gradtol=1e-8,
                              maxiter=1000, overlap_tol=1e-10)

Configure the gradient variational method (GVM). All the exponents of the
basis set are optimized simultaneously with the analytic energy gradient

```math
\\frac{\\partial E}{\\partial a_k}
= 2 c_k \\sum_j c_j
  \\left(
    \\frac{\\partial H_{kj}}{\\partial a_k} - E \\frac{\\partial S_{kj}}{\\partial a_k}
  \\right),
```

the Hellmann-Feynman derivative of the lowest generalized eigenvalue for
``\\pmb{c}^\\top \\pmb{S} \\pmb{c} = 1``. The matrix-element derivatives are
evaluated with forward-mode automatic differentiation, so the `element`
methods of the basis functions have to accept a dual number as the exponent.
The optimization variables are ``\\log a_k``, which keeps every exponent
positive.

`max_basis` is the number of basis functions of a cold start, whose exponents
form a geometric progression from `amin` to `amax`. It must be omitted for a
warm start, where the size and the exponents come from the given basis set.
`optimizer` is any gradient-based `Optim` algorithm, `gradtol` is the
gradient-norm tolerance, `maxiter` limits the iterations, and `overlap_tol`
rejects nearly linearly dependent basis sets during the optimization.

Reference: K. Varga, Y. Suzuki, *Phys. Rev. C* **52**, 2885 (1995),
[arXiv:nucl-th/9508023](https://arxiv.org/abs/nucl-th/9508023).
"""
struct GradientVariationalMethod{T<:AbstractFloat,O} <: SolverMethod
  max_basis::Union{Nothing,Int}
  amin::T
  amax::T
  optimizer::O
  gradtol::T
  maxiter::Int
  overlap_tol::T
end

function GradientVariationalMethod(;
  max_basis::Union{Nothing,Int}=nothing,
  amin::Real=1e-4,
  amax::Real=1e+4,
  optimizer=Optim.LBFGS(),
  gradtol::Real=1e-8,
  maxiter::Int=1000,
  overlap_tol::Real=1e-10,
)
  isnothing(max_basis) || max_basis > 0 ||
    throw(ArgumentError("max_basis must be positive"))
  isfinite(amin) && amin > 0 ||
    throw(ArgumentError("amin must be positive and finite"))
  isfinite(amax) && amax > amin ||
    throw(ArgumentError("amax must be finite and greater than amin"))
  isfinite(gradtol) && gradtol > 0 ||
    throw(ArgumentError("gradtol must be positive and finite"))
  maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
  isfinite(overlap_tol) && overlap_tol > 0 ||
    throw(ArgumentError("overlap_tol must be positive and finite"))

  amin_value, amax_value, gradtol_value, overlap_tol_value =
    promote(float(amin), float(amax), float(gradtol), float(overlap_tol))
  return GradientVariationalMethod{typeof(amin_value),typeof(optimizer)}(
    max_basis,
    amin_value,
    amax_value,
    optimizer,
    gradtol_value,
    maxiter,
    overlap_tol_value,
  )
end

# Optim's algorithms print their whole configuration, so the summary of the
# optimizer is used instead of the object itself.
Base.string(method::GradientVariationalMethod) =
  "GradientVariationalMethod(" *
  join(
    (
      "$(name)=$(name === :optimizer ? summary(method.optimizer) : getproperty(method, name))"
      for name in fieldnames(typeof(method))
    ),
    ", ",
  ) *
  ")"

"""
    GVM(; kwargs...)

Alias for [`GradientVariationalMethod`](@ref).
"""
const GVM = GradientVariationalMethod

# The energy of a rejected basis set. A finite value keeps the line search of
# the optimizer working, while `Inf` is recorded in the optimization log.
const _gvm_penalty = 1e+10

# gradient

"""
    TwoBody._gvm_gradient!(gradient, hamiltonian, basis, exponents, energy, coefficients)

Fill `gradient` with the derivatives of the ground-state energy with respect to
``\\log a_k``. `basis` holds the basis functions with the current `exponents`,
and `coefficients` is the normalized eigenvector of `energy`. The matrix
elements are assumed to be real and symmetric, as in
[`matrix`](@ref TwoBody.matrix).
"""
function _gvm_gradient!(
  gradient::AbstractVector,
  hamiltonian::Hamiltonian,
  basis::AbstractVector{<:Basis},
  exponents::AbstractVector{<:Real},
  energy::Real,
  coefficients::AbstractVector{<:Real},
)
  for k in eachindex(basis)
    total = zero(eltype(gradient))
    for j in eachindex(basis)
      derivative_H = ForwardDiff.derivative(
        a -> element(hamiltonian, _replace_exponent(basis[k], a), basis[j]),
        exponents[k],
      )
      derivative_S = ForwardDiff.derivative(
        a -> element(_replace_exponent(basis[k], a), basis[j]),
        exponents[k],
      )
      total += coefficients[j] * (derivative_H - energy * derivative_S)
    end
    # the chain rule of a = exp(x) turns ∂E/∂a into ∂E/∂x
    gradient[k] = 2 * coefficients[k] * total * exponents[k]
  end
  return gradient
end

# initial exponents

function _gvm_initial_exponents(method::GradientVariationalMethod, n::Int)
  n == 1 && return [sqrt(method.amin * method.amax)]
  ratio = (method.amax / method.amin)^(1 / (n - 1))
  return [method.amin * ratio^(index - 1) for index in 1:n]
end

# solver

"""
    solve(
        hamiltonian::Hamiltonian,
        basisset::BasisSet,
        method::GradientVariationalMethod;
        perturbation=Hamiltonian(),
        info=4,
        progress=true,
    )
    solve(
        hamiltonian::Hamiltonian,
        prototype::Basis,
        method::GradientVariationalMethod;
        perturbation=Hamiltonian(),
        info=4,
        progress=true,
    )

Optimize every exponent of the basis set with the analytic energy gradient,
then solve the optimized basis set with the Rayleigh-Ritz method. The first
form warm-starts from `basisset`, which fixes the size, the types, and the
initial exponents. The second form is a cold start: `method.max_basis` copies
of `prototype` are used, and their exponents are initialized as a geometric
progression from `method.amin` to `method.amax`.

The result is a `ResultRayleighRitz` with additional `method`, `optimizer`,
`initialbasisset`, `history`, `n_evaluations`, `iterations`, and `converged`
properties. `progress=true` prints the optimization log with the result.
"""
function solve(
  hamiltonian::Hamiltonian,
  basisset::BasisSet,
  method::GradientVariationalMethod;
  perturbation::Hamiltonian=Hamiltonian(),
  info::Int=4,
  progress::Bool=true,
)
  # Forward `info` to Rayleigh-Ritz semantics (negative values are allowed there).
  isempty(basisset.basis) &&
    throw(ArgumentError("a warm start needs a nonempty BasisSet"))
  isnothing(method.max_basis) || method.max_basis == length(basisset) || throw(ArgumentError(
    "max_basis=$(method.max_basis) does not match the $(length(basisset)) " *
    "functions of the initial BasisSet; omit max_basis for a warm start",
  ))

  prototypes = Basis[basisset.basis...]
  initial = _variational_energy(
    hamiltonian, basisset; overlap_tol=method.overlap_tol,
  )
  initial.valid || error(
    "GVM cannot start from the given BasisSet: $(initial.message)",
  )

  history = NamedTuple[]

  function objective!(F, G, x)
    exponents = exp.(x)
    basis = Basis[
      _replace_exponent(prototypes[index], exponents[index])
      for index in eachindex(prototypes)
    ]
    evaluation = _variational_energy(
      hamiltonian, BasisSet(basis...); overlap_tol=method.overlap_tol,
    )
    if !evaluation.valid
      isnothing(G) || fill!(G, 0)
      push!(history, (energy=Inf, parameters=exponents))
      return isnothing(F) ? nothing : _gvm_penalty
    end
    isnothing(G) || _gvm_gradient!(
      G, hamiltonian, basis, exponents, evaluation.energy, evaluation.coefficients,
    )
    push!(history, (energy=evaluation.energy, parameters=exponents))
    return isnothing(F) ? nothing : evaluation.energy
  end

  optimization = Optim.optimize(
    Optim.only_fg!(objective!),
    [log(_exponent(basis)) for basis in prototypes],
    method.optimizer,
    Optim.Options(g_tol=method.gradtol, iterations=method.maxiter),
  )

  exponents = exp.(Optim.minimizer(optimization))
  optimized = BasisSet((
    _replace_exponent(prototypes[index], exponents[index])
    for index in eachindex(prototypes)
  )...)
  final = _variational_energy(
    hamiltonian, optimized; overlap_tol=method.overlap_tol,
  )
  final.valid || error(
    "GVM did not find a valid basis set: $(final.message); " *
    "narrow [amin, amax] or increase overlap_tol=$(method.overlap_tol)",
  )

  # result
  result = solve(hamiltonian, optimized; perturbation=perturbation, info=info)
  return ResultRayleighRitz(;
    getfield(result, :data)...,
    method=method,
    optimizer=method.optimizer,
    initialbasisset=basisset,
    progress=progress,
    history=history,
    n_evaluations=length(history),
    iterations=Optim.iterations(optimization),
    converged=Optim.converged(optimization),
  )
end

function solve(
  hamiltonian::Hamiltonian,
  prototype::Basis,
  method::GradientVariationalMethod;
  perturbation::Hamiltonian=Hamiltonian(),
  info::Int=4,
  progress::Bool=true,
)
  isnothing(method.max_basis) && throw(ArgumentError(
    "a cold start needs max_basis: " *
    "call solve(hamiltonian, basisset, method) to start from a BasisSet",
  ))
  basisset = BasisSet((
    _replace_exponent(prototype, exponent)
    for exponent in _gvm_initial_exponents(method, method.max_basis)
  )...)
  return solve(
    hamiltonian,
    basisset,
    method;
    perturbation=perturbation,
    info=info,
    progress=progress,
  )
end
