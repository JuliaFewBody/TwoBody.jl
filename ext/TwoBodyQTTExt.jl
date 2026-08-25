module TwoBodyQTTExt

using TwoBody
import Logging
import TensorTrainNumerics
import TwoBody: PotentialTerm, V, matrix

const QTTVector = TensorTrainNumerics.TTvector
const QTTMatrix = TensorTrainNumerics.TToperator

TwoBody.order(tt::Union{QTTVector,QTTMatrix}) = tt.N
TwoBody.ranks(tt::QTTVector) = copy(tt.ttv_rks)
TwoBody.ranks(tt::QTTMatrix) = copy(tt.tto_rks)

function TwoBody.qttvalue(tt::QTTVector{T}, index::Integer) where {T}
  n = 1 << TwoBody.order(tt)
  1 <= index <= n || throw(BoundsError(1:n, index))
  state = ones(T, 1)
  index0 = index - 1
  for site in 1:TwoBody.order(tt)
    physical = Int((index0 >> (TwoBody.order(tt) - site)) & 1) + 1
    state = vec(reshape(state, 1, :) * @view(tt.ttv_vec[site][physical, :, :]))
  end
  return only(state)
end

function TwoBody.qttvalue(tt::QTTMatrix{T}, row::Integer, column::Integer) where {T}
  n = 1 << TwoBody.order(tt)
  1 <= row <= n || throw(BoundsError(1:n, row))
  1 <= column <= n || throw(BoundsError(1:n, column))
  state = ones(T, 1)
  row0, column0 = row - 1, column - 1
  for site in 1:TwoBody.order(tt)
    rowbit = Int((row0 >> (TwoBody.order(tt) - site)) & 1) + 1
    columnbit = Int((column0 >> (TwoBody.order(tt) - site)) & 1) + 1
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

function solve_qtt(
  hamiltonian::Hamiltonian,
  method::QuanticsTensorTrainMethod;
  initial=r -> exp(-r),
  info=4,
  nₘₐₓ=4,
)
  return _solve_qtt(hamiltonian, initial, method; info=info, nₘₐₓ=nₘₐₓ)
end

function solve_qtt(
  hamiltonian::Hamiltonian,
  wavefunction::Function,
  method::QuanticsTensorTrainMethod;
  info=4,
  nₘₐₓ=4,
)
  return _solve_qtt(hamiltonian, wavefunction, method; info=info, nₘₐₓ=nₘₐₓ)
end

end
