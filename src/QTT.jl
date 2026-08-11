export QuanticsTensorTrainMethod, QTTVector, QTTMatrix, order, ranks, qttvalue

import LinearAlgebra
import Logging
import TensorTrainNumerics

"A vector stored as a quantics tensor train."
const QTTVector = TensorTrainNumerics.TTvector

"A matrix stored as a quantics tensor train / matrix product operator."
const QTTMatrix = TensorTrainNumerics.TToperator

"Return the number of binary sites in a QTT."
order(tt::Union{QTTVector,QTTMatrix}) = tt.N

"Return the left boundary, internal, and right boundary ranks of a QTT."
ranks(tt::QTTVector) = copy(tt.ttv_rks)
ranks(tt::QTTMatrix) = copy(tt.tto_rks)

struct QuanticsTensorTrainMethod
  quantics::Int
  r₀::Float64
  rₘₐₓ::Float64
  R::LinRange{Float64,Int}
  Δr::Float64
  l::Int
  tolerance::Float64
  maxbonddim::Int
  maxoperatorbonddim::Int
  maxiter::Int
  sweeps::Int
  deflation_shift::Float64

  function QuanticsTensorTrainMethod(;
    quantics=10,
    r₀=0.0,
    rₘₐₓ=50.0,
    l=0,
    tolerance=1e-10,
    maxbonddim=64,
    maxoperatorbonddim=2 * maxbonddim,
    maxiter=100,
    sweeps=4,
    deflation_shift=10.0,
  )
    quantics >= 2 || throw(ArgumentError("quantics must be at least 2"))
    0 <= r₀ < rₘₐₓ || throw(ArgumentError("the radial interval must satisfy 0 <= r₀ < rₘₐₓ"))
    l >= 0 || throw(ArgumentError("l must be nonnegative"))
    tolerance > 0 || throw(ArgumentError("tolerance must be positive"))
    maxbonddim >= 1 || throw(ArgumentError("maxbonddim must be positive"))
    maxoperatorbonddim >= 1 || throw(ArgumentError("maxoperatorbonddim must be positive"))
    maxiter >= 1 || throw(ArgumentError("maxiter must be positive"))
    sweeps >= 1 || throw(ArgumentError("sweeps must be positive"))
    deflation_shift > 0 || throw(ArgumentError("deflation_shift must be positive"))

    npoints = 1 << quantics
    Δr = Float64(rₘₐₓ - r₀) / (npoints + 1)
    R = range(Float64(r₀) + Δr, Float64(rₘₐₓ) - Δr; length=npoints)
    new(
      quantics,
      Float64(r₀),
      Float64(rₘₐₓ),
      R,
      Δr,
      l,
      Float64(tolerance),
      maxbonddim,
      maxoperatorbonddim,
      maxiter,
      sweeps,
      Float64(deflation_shift),
    )
  end
end

Base.string(method::QuanticsTensorTrainMethod) =
  "QuanticsTensorTrainMethod(" *
  join(["$(symbol)=$(getproperty(method, symbol))" for symbol in
        (:quantics, :r₀, :rₘₐₓ, :l, :tolerance, :maxbonddim,
         :maxoperatorbonddim, :maxiter, :sweeps, :deflation_shift)], ", ") * ")"
Base.show(io::IO, method::QuanticsTensorTrainMethod) = print(io, string(method))

function qttvalue(tt::QTTVector{T}, index::Integer) where {T}
  n = 1 << order(tt)
  1 <= index <= n || throw(BoundsError(1:n, index))
  state = ones(T, 1)
  index0 = index - 1
  for site in 1:order(tt)
    physical = Int((index0 >> (order(tt) - site)) & 1) + 1
    state = vec(reshape(state, 1, :) * @view(tt.ttv_vec[site][physical, :, :]))
  end
  return only(state)
end

function qttvalue(tt::QTTMatrix{T}, row::Integer, column::Integer) where {T}
  n = 1 << order(tt)
  1 <= row <= n || throw(BoundsError(1:n, row))
  1 <= column <= n || throw(BoundsError(1:n, column))
  state = ones(T, 1)
  row0, column0 = row - 1, column - 1
  for site in 1:order(tt)
    rowbit = Int((row0 >> (order(tt) - site)) & 1) + 1
    columnbit = Int((column0 >> (order(tt) - site)) & 1) + 1
    state = vec(reshape(state, 1, :) * @view(tt.tto_vec[site][rowbit, columnbit, :, :]))
  end
  return only(state)
end

function _qttvector(f::Function, method::QuanticsTensorTrainMethod)
  q = method.quantics
  domain = [Float64[0, 1] for _ in 1:q]

  function evaluate(bits::AbstractMatrix)
    indices = zeros(Int, size(bits, 1))
    for site in 1:q
      indices .+= round.(Int, @view(bits[:, site])) .* (1 << (q - site))
    end
    return [Float64(f(method.R[index + 1])) for index in indices]
  end

  algorithm = TensorTrainNumerics.MaxVol(
    tol=method.tolerance,
    rmax=method.maxbonddim,
    maxiter=method.maxiter,
    verbose=false,
  )
  return TensorTrainNumerics.tt_cross(
    evaluate,
    domain,
    algorithm;
    ranks=min(2, method.maxbonddim),
    val_size=min(1000, 1 << q),
  )
end

function _compress_vector(vector::QTTVector, method::QuanticsTensorTrainMethod)
  compressed = copy(vector)
  TensorTrainNumerics.tt_compress!(
    compressed,
    method.maxbonddim;
    truncerr=method.tolerance,
  )
  return compressed
end

function _compress_operator(operator::QTTMatrix, method::QuanticsTensorTrainMethod)
  fused = TensorTrainNumerics.tto_to_ttv(operator)
  TensorTrainNumerics.tt_compress!(
    fused,
    method.maxoperatorbonddim;
    truncerr=method.tolerance,
  )
  return TensorTrainNumerics.ttv_to_tto(fused)
end

function matrix(term::Kinetic, method::QuanticsTensorTrainMethod)
  kinetic = (term.hbar^2 / (2 * term.m * method.Δr^2)) *
    TensorTrainNumerics.Δ(method.quantics)
  if method.l == 0
    return kinetic
  end
  centrifugal = _qttvector(
    r -> term.hbar^2 * method.l * (method.l + 1) / (2 * term.m * r^2),
    method,
  )
  return kinetic + TensorTrainNumerics.ttv_to_diag_tto(centrifugal)
end

matrix(term::RestEnergy, method::QuanticsTensorTrainMethod) =
  TensorTrainNumerics.ttv_to_diag_tto(_qttvector(_ -> term.m * term.c^2, method))

const QTTPotentialTerm = Union{Constant,Linear,Coulomb,PowerLaw,Gaussian,Exponential,Yukawa,Custom}

matrix(term::QTTPotentialTerm, method::QuanticsTensorTrainMethod) =
  TensorTrainNumerics.ttv_to_diag_tto(_qttvector(r -> V(term, r), method))

matrix(term::PotentialTerm, ::QuanticsTensorTrainMethod) =
  throw(ArgumentError("$(typeof(term)) is not supported by QuanticsTensorTrainMethod"))

function matrix(hamiltonian::Hamiltonian, method::QuanticsTensorTrainMethod)
  isempty(hamiltonian.terms) && throw(ArgumentError("the Hamiltonian must contain at least one term"))
  matrices = map(hamiltonian.terms) do term
    term isa Union{Kinetic,RestEnergy,QTTPotentialTerm} ||
      throw(ArgumentError("$(typeof(term)) is not supported by QuanticsTensorTrainMethod"))
    matrix(term, method)
  end
  return _compress_operator(reduce(+, matrices), method)
end

_ttnorm(vector::QTTVector) = TensorTrainNumerics.norm(vector)
_ttdot(left::QTTVector, right::QTTVector) = TensorTrainNumerics.dot(left, right)

function _dmrg_groundstate(operator::QTTMatrix, guess::QTTVector, method::QuanticsTensorTrainMethod)
  normalized_guess = guess / _ttnorm(guess)
  history, state, rank_history = Logging.with_logger(Logging.NullLogger()) do
    TensorTrainNumerics.dmrg_eigsolve(
      operator,
      normalized_guess;
      N=2,
      tol=method.tolerance,
      sweep_schedule=[method.sweeps],
      rmax_schedule=[method.maxbonddim],
      it_solver=true,
      linsolv_maxiter=method.maxiter,
      linsolv_tol=max(sqrt(method.tolerance), 1e-8),
    )
  end
  state = state / _ttnorm(state)
  return last(history), state, history, rank_history
end

function _state_guess(initial::Function, level::Int, method::QuanticsTensorTrainMethod)
  width = method.rₘₐₓ - method.r₀
  modulation(r) = level == 0 ? 1.0 : cos(level * π * (r - method.r₀) / width)
  return _qttvector(r -> r * initial(r) * modulation(r), method)
end

function _solve_qtt(
  hamiltonian::Hamiltonian,
  initial::Function,
  method::QuanticsTensorTrainMethod;
  info=4,
  nₘₐₓ=4,
)
  1 <= nₘₐₓ <= (1 << method.quantics) ||
    throw(ArgumentError("nₘₐₓ must lie between 1 and the number of grid points"))

  H = matrix(hamiltonian, method)
  deflated = H
  energies = Float64[]
  states = QTTVector[]
  histories = Vector{Float64}[]
  rank_histories = Vector{Int}[]
  residuals = Float64[]

  for level in 0:(nₘₐₓ - 1)
    guess = _state_guess(initial, level, method)
    _, state, history, rank_history = _dmrg_groundstate(deflated, guess, method)

    Hstate = H * state
    energy = real(_ttdot(state, Hstate) / _ttdot(state, state))
    residual = _ttnorm(Hstate - energy * state)
    push!(energies, energy)
    push!(states, state)
    push!(histories, history)
    push!(rank_histories, rank_history)
    push!(residuals, residual)

    if level < nₘₐₓ - 1
      projector = TensorTrainNumerics.outer_product(state, state)
      deflated = _compress_operator(deflated + method.deflation_shift * projector, method)
    end
  end

  overlaps = [_ttdot(states[i], states[j]) for i in eachindex(states), j in eachindex(states)]
  scale = inv(sqrt(4π * method.Δr))
  u = [scale * state for state in states]
  inverse_radius = _qttvector(r -> inv(r), method)
  ψ = [_compress_vector(TensorTrainNumerics.hadamard(state, inverse_radius), method) for state in u]

  if info > 0
    println("\n# method\n")
    println(method)
    println("\n# eigenvalues\n")
    for level in 1:min(nₘₐₓ, info)
      println("E[$level] = $(energies[level])")
    end
    println()
  end

  return info >= 0 ? (
    hamiltonian=hamiltonian,
    method=method,
    nₘₐₓ=nₘₐₓ,
    H=H,
    E=energies,
    C=states,
    ψ=ψ,
    u=u,
    overlaps=overlaps,
    residuals=residuals,
    histories=histories,
    rank_histories=rank_histories,
  ) : (E=energies,)
end

function solve(
  hamiltonian::Hamiltonian,
  method::QuanticsTensorTrainMethod;
  initial=r -> exp(-r),
  info=4,
  nₘₐₓ=4,
)
  return _solve_qtt(hamiltonian, initial, method; info=info, nₘₐₓ=nₘₐₓ)
end

function solve(
  hamiltonian::Hamiltonian,
  wavefunction::Function,
  method::QuanticsTensorTrainMethod;
  info=4,
  nₘₐₓ=4,
)
  return _solve_qtt(hamiltonian, wavefunction, method; info=info, nₘₐₓ=nₘₐₓ)
end

@doc raw"""
    QuanticsTensorTrainMethod(; quantics=10, r₀=0.0, rₘₐₓ=50.0, l=0,
                               tolerance=1e-10, maxbonddim=64,
                               maxoperatorbonddim=128, maxiter=100, sweeps=4,
                               deflation_shift=10.0)

Configure a radial QTT calculation on ``2^q`` uniformly spaced interior points,
where `q = quantics`. `r₀` and `rₘₐₓ` are zero-value (Dirichlet) boundaries and are
excluded from the grid. `maxbonddim` bounds DMRG state ranks;
`maxoperatorbonddim` bounds the compressed Hamiltonian and deflation-projector ranks.
`deflation_shift` controls the penalty applied to states already found.
""" QuanticsTensorTrainMethod

@doc raw"""
    solve(hamiltonian, method::QuanticsTensorTrainMethod;
          initial=r -> exp(-r), info=4, nₘₐₓ=4)
    solve(hamiltonian, initial::Function, method::QuanticsTensorTrainMethod;
          info=4, nₘₐₓ=4)

Find the lowest `nₘₐₓ` eigenstates of the reduced radial Hamiltonian with the
TensorTrainNumerics.jl two-site DMRG eigensolver. Excited states are obtained by
successive projector deflation,

```math
H^{(n)} = H + \mu \sum_{k=0}^{n-1} |u_k\rangle\langle u_k|.
```

The supplied `initial` function seeds the first state; node-modulated versions seed
subsequent states. The returned energies are Rayleigh quotients of the original,
undeflated Hamiltonian. Inspect `overlaps` and `residuals` to assess convergence.
""" solve(hamiltonian::Hamiltonian, method::QuanticsTensorTrainMethod; initial=r -> exp(-r), info=4, nₘₐₓ=4)

@doc raw"""
    qttvalue(tt::QTTVector, index)
    qttvalue(tt::QTTMatrix, row, column)

Contract a QTT at one vector or matrix index without materializing its ``2^q`` entries.
""" qttvalue
