export QuanticsTensorTrainMethod, QTTVector, QTTMatrix, order, ranks, qttvalue

import LinearAlgebra
import QuanticsTCI
import TensorCrossInterpolation

"A vector stored as a quantics tensor train with cores `(left rank, 2, right rank)`."
struct QTTVector{T}
  cores::Vector{Array{T,3}}
end

"A matrix stored as a quantics tensor train with cores `(left rank, 2, 2, right rank)`."
struct QTTMatrix{T}
  cores::Vector{Array{T,4}}
end

"Return the number of binary sites in a QTT."
order(tt::Union{QTTVector,QTTMatrix}) = length(tt.cores)

"Return the left boundary, internal, and right boundary ranks of a QTT."
ranks(tt::Union{QTTVector,QTTMatrix}) = [size(tt.cores[1], 1); [size(core, ndims(core)) for core in tt.cores]]

struct QuanticsTensorTrainMethod
  quantics::Int
  r₀::Float64
  rₘₐₓ::Float64
  R::LinRange{Float64,Int}
  Δr::Float64
  l::Int
  tolerance::Float64
  maxbonddim::Int
  maxiter::Int

  function QuanticsTensorTrainMethod(;
    quantics=10,
    r₀=0.0,
    rₘₐₓ=50.0,
    l=0,
    tolerance=1e-10,
    maxbonddim=64,
    maxiter=100,
  )
    quantics >= 2 || throw(ArgumentError("quantics must be at least 2"))
    0 <= r₀ < rₘₐₓ || throw(ArgumentError("the radial interval must satisfy 0 <= r₀ < rₘₐₓ"))
    l >= 0 || throw(ArgumentError("l must be nonnegative"))
    tolerance > 0 || throw(ArgumentError("tolerance must be positive"))
    maxbonddim >= 1 || throw(ArgumentError("maxbonddim must be positive"))
    maxiter >= 1 || throw(ArgumentError("maxiter must be positive"))

    npoints = 1 << quantics
    Δr = Float64(rₘₐₓ - r₀) / (npoints + 1)
    R = range(Float64(r₀) + Δr, Float64(rₘₐₓ) - Δr; length=npoints)
    new(quantics, Float64(r₀), Float64(rₘₐₓ), R, Δr, l,
        Float64(tolerance), maxbonddim, maxiter)
  end
end

Base.string(method::QuanticsTensorTrainMethod) =
  "QuanticsTensorTrainMethod(" *
  join(["$(symbol)=$(getproperty(method, symbol))" for symbol in
        (:quantics, :r₀, :rₘₐₓ, :l, :tolerance, :maxbonddim, :maxiter)], ", ") * ")"
Base.show(io::IO, method::QuanticsTensorTrainMethod) = print(io, string(method))

function _check_cores(tt::QTTVector)
  isempty(tt.cores) && throw(ArgumentError("a QTT must contain at least one core"))
  size(first(tt.cores), 1) == 1 || throw(ArgumentError("the first left rank must be one"))
  size(last(tt.cores), 3) == 1 || throw(ArgumentError("the last right rank must be one"))
  all(size(core, 2) == 2 for core in tt.cores) || throw(ArgumentError("every physical dimension must be two"))
  all(size(tt.cores[k], 3) == size(tt.cores[k + 1], 1) for k in 1:(order(tt) - 1)) ||
    throw(ArgumentError("adjacent QTT ranks do not match"))
  return tt
end

function _check_cores(tt::QTTMatrix)
  isempty(tt.cores) && throw(ArgumentError("a QTT must contain at least one core"))
  size(first(tt.cores), 1) == 1 || throw(ArgumentError("the first left rank must be one"))
  size(last(tt.cores), 4) == 1 || throw(ArgumentError("the last right rank must be one"))
  all(size(core, 2) == size(core, 3) == 2 for core in tt.cores) ||
    throw(ArgumentError("every physical dimension must be two by two"))
  all(size(tt.cores[k], 4) == size(tt.cores[k + 1], 1) for k in 1:(order(tt) - 1)) ||
    throw(ArgumentError("adjacent QTT ranks do not match"))
  return tt
end

function qttvalue(tt::QTTVector{T}, index::Integer) where {T}
  n = 1 << order(tt)
  1 <= index <= n || throw(BoundsError(1:n, index))
  state = ones(T, 1)
  index0 = index - 1
  for site in 1:order(tt)
    core = tt.cores[site]
    physical = Int((index0 >> (order(tt) - site)) & 1) + 1
    state = vec(reshape(state, 1, :) * @view(core[:, physical, :]))
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
    state = vec(reshape(state, 1, :) * @view(tt.cores[site][:, rowbit, columnbit, :]))
  end
  return only(state)
end

function Base.:*(factor::Number, tt::QTTVector)
  cores = copy(tt.cores)
  cores[1] = factor .* cores[1]
  return QTTVector(cores)
end

function Base.:*(factor::Number, tt::QTTMatrix)
  cores = copy(tt.cores)
  cores[1] = factor .* cores[1]
  return QTTMatrix(cores)
end

Base.:*(tt::Union{QTTVector,QTTMatrix}, factor::Number) = factor * tt
Base.:-(tt::QTTMatrix) = -1 * tt
Base.:-(A::QTTMatrix, B::QTTMatrix) = A + (-B)

function Base.:+(A::QTTMatrix{TA}, B::QTTMatrix{TB}) where {TA,TB}
  order(A) == order(B) || throw(DimensionMismatch("QTT orders must match"))
  nsites = order(A)
  T = promote_type(TA, TB)
  cores = Vector{Array{T,4}}(undef, nsites)

  rankA, rankB = size(A.cores[1], 4), size(B.cores[1], 4)
  cores[1] = zeros(T, 1, 2, 2, rankA + rankB)
  cores[1][1, :, :, 1:rankA] = A.cores[1]
  cores[1][1, :, :, (rankA + 1):end] = B.cores[1]

  for site in 2:(nsites - 1)
    a, b = A.cores[site], B.cores[site]
    aleft, aright = size(a, 1), size(a, 4)
    bleft, bright = size(b, 1), size(b, 4)
    core = zeros(T, aleft + bleft, 2, 2, aright + bright)
    core[1:aleft, :, :, 1:aright] = a
    core[(aleft + 1):end, :, :, (aright + 1):end] = b
    cores[site] = core
  end

  a, b = A.cores[end], B.cores[end]
  aleft, bleft = size(a, 1), size(b, 1)
  cores[end] = zeros(T, aleft + bleft, 2, 2, 1)
  cores[end][1:aleft, :, :, 1] = a[:, :, :, 1]
  cores[end][(aleft + 1):end, :, :, 1] = b[:, :, :, 1]
  return QTTMatrix(cores)
end

function _tridiagonal_qtt(quantics::Int, lower::Real, diagonal::Real, upper::Real)
  identity2 = Matrix{Float64}(LinearAlgebra.I, 2, 2)
  shift = [0.0 1.0; 0.0 0.0]

  firstcore = zeros(Float64, 1, 2, 2, 3)
  firstcore[1, :, :, 1] = identity2
  firstcore[1, :, :, 2] = transpose(shift)
  firstcore[1, :, :, 3] = shift

  middlecore = zeros(Float64, 3, 2, 2, 3)
  middlecore[1, :, :, 1] = identity2
  middlecore[1, :, :, 2] = transpose(shift)
  middlecore[1, :, :, 3] = shift
  middlecore[2, :, :, 2] = shift
  middlecore[3, :, :, 3] = transpose(shift)

  lastcore = zeros(Float64, 3, 2, 2, 1)
  lastcore[1, :, :, 1] = diagonal * identity2 + upper * shift + lower * transpose(shift)
  lastcore[2, :, :, 1] = lower * shift
  lastcore[3, :, :, 1] = upper * transpose(shift)

  cores = Vector{Array{Float64,4}}(undef, quantics)
  cores[1] = firstcore
  for site in 2:(quantics - 1)
    cores[site] = middlecore
  end
  cores[end] = lastcore
  return QTTMatrix(cores)
end

function _diagonal(tt::QTTVector{T}) where {T}
  cores = Vector{Array{T,4}}(undef, order(tt))
  for site in 1:order(tt)
    vectorcore = tt.cores[site]
    core = zeros(T, size(vectorcore, 1), 2, 2, size(vectorcore, 3))
    core[:, 1, 1, :] = vectorcore[:, 1, :]
    core[:, 2, 2, :] = vectorcore[:, 2, :]
    cores[site] = core
  end
  return QTTMatrix(cores)
end

function _qttvector_from_tci(tci, method::QuanticsTensorTrainMethod, f::Function)
  tensortrain = TensorCrossInterpolation.TensorTrain(tci.tci)
  tensortrain = TensorCrossInterpolation.reverse(tensortrain)
  sitetensors = collect(TensorCrossInterpolation.sitetensors(tensortrain))
  length(sitetensors) == method.quantics || throw(ArgumentError("unexpected tensor-train order"))
  qtt = _check_cores(QTTVector([Float64.(Array(core)) for core in sitetensors]))

  reversed = QTTVector([permutedims(core, (3, 2, 1)) for core in reverse(qtt.cores)])
  probes = unique(clamp.([1, 2, length(method.R) ÷ 2, length(method.R) - 1, length(method.R)], 1, length(method.R)))
  normalerror = sum(abs(qttvalue(qtt, i) - f(method.R[i])) for i in probes)
  reversederror = sum(abs(qttvalue(reversed, i) - f(method.R[i])) for i in probes)
  return reversederror < normalerror ? reversed : qtt
end

function _qttvector(f::Function, method::QuanticsTensorTrainMethod)
  tci, _, _ = QuanticsTCI.quanticscrossinterpolate(
    Float64,
    r -> Float64(f(r)),
    method.R;
    tolerance=method.tolerance,
    maxbonddim=method.maxbonddim,
    maxiter=method.maxiter,
  )
  return _qttvector_from_tci(tci, method, f)
end

function _inner(left::QTTVector{TL}, right::QTTVector{TR}) where {TL,TR}
  order(left) == order(right) || throw(DimensionMismatch("QTT orders must match"))
  T = promote_type(TL, TR)
  environment = ones(T, 1, 1)
  for site in 1:order(left)
    lcore, rcore = left.cores[site], right.cores[site]
    next = zeros(T, size(lcore, 3), size(rcore, 3))
    for lleft in axes(lcore, 1), rleft in axes(rcore, 1), physical in 1:2,
        lright in axes(lcore, 3), rright in axes(rcore, 3)
      next[lright, rright] += environment[lleft, rleft] *
        conj(lcore[lleft, physical, lright]) * rcore[rleft, physical, rright]
    end
    environment = next
  end
  return only(environment)
end

function _expectation(left::QTTVector{TL}, operator::QTTMatrix{TO}, right::QTTVector{TR}) where {TL,TO,TR}
  order(left) == order(operator) == order(right) || throw(DimensionMismatch("QTT orders must match"))
  T = promote_type(TL, TO, TR)
  environment = ones(T, 1, 1, 1)
  for site in 1:order(left)
    lcore, opcore, rcore = left.cores[site], operator.cores[site], right.cores[site]
    next = zeros(T, size(lcore, 3), size(opcore, 4), size(rcore, 3))
    for lleft in axes(lcore, 1), oleft in axes(opcore, 1), rleft in axes(rcore, 1),
        row in 1:2, column in 1:2, lright in axes(lcore, 3),
        oright in axes(opcore, 4), rright in axes(rcore, 3)
      next[lright, oright, rright] += environment[lleft, oleft, rleft] *
        conj(lcore[lleft, row, lright]) * opcore[oleft, row, column, oright] *
        rcore[rleft, column, rright]
    end
    environment = next
  end
  return only(environment)
end

function matrix(term::Kinetic, method::QuanticsTensorTrainMethod)
  h = method.Δr
  secondderivative = _tridiagonal_qtt(method.quantics, 1 / h^2, -2 / h^2, 1 / h^2)
  radialoperator = if method.l == 0
    secondderivative
  else
    centrifugal = method.l * (method.l + 1) * _diagonal(_qttvector(r -> 1 / r^2, method))
    secondderivative - centrifugal
  end
  return (-term.hbar^2 / (2 * term.m)) * radialoperator
end

matrix(term::RestEnergy, method::QuanticsTensorTrainMethod) =
  _diagonal(_qttvector(_ -> term.m * term.c^2, method))

const QTTPotentialTerm = Union{Constant,Linear,Coulomb,PowerLaw,Gaussian,Exponential,Yukawa,Custom}

matrix(term::QTTPotentialTerm, method::QuanticsTensorTrainMethod) =
  _diagonal(_qttvector(r -> V(term, r), method))

matrix(term::PotentialTerm, ::QuanticsTensorTrainMethod) =
  throw(ArgumentError("$(typeof(term)) is not supported by QuanticsTensorTrainMethod"))

function matrix(hamiltonian::Hamiltonian, method::QuanticsTensorTrainMethod)
  isempty(hamiltonian.terms) && throw(ArgumentError("the Hamiltonian must contain at least one term"))
  matrices = map(hamiltonian.terms) do term
    term isa Union{Kinetic,RestEnergy,QTTPotentialTerm} ||
      throw(ArgumentError("$(typeof(term)) is not supported by QuanticsTensorTrainMethod"))
    matrix(term, method)
  end
  return reduce(+, matrices)
end

function solve(
  hamiltonian::Hamiltonian,
  wavefunction::Function,
  method::QuanticsTensorTrainMethod;
  info=1,
)
  H = matrix(hamiltonian, method)
  unnormalizedψ = _qttvector(wavefunction, method)
  unnormalizedu = _qttvector(r -> r * wavefunction(r), method)
  denominator = _inner(unnormalizedu, unnormalizedu)
  numerator = _expectation(unnormalizedu, H, unnormalizedu)
  energy = real(numerator / denominator)
  normalization = sqrt(real(4π * method.Δr * denominator))
  ψ = (1 / normalization) * unnormalizedψ
  u = (1 / normalization) * unnormalizedu

  if info > 0
    println("\n# method\n")
    println(method)
    println("\n# energy\n")
    println("E = $energy\n")
  end

  return info >= 0 ? (
    hamiltonian=hamiltonian,
    method=method,
    H=H,
    E=energy,
    ψ=ψ,
    u=u,
  ) : (E=energy,)
end

@doc raw"""
    QuanticsTensorTrainMethod(; quantics=10, r₀=0.0, rₘₐₓ=50.0, l=0,
                               tolerance=1e-10, maxbonddim=64, maxiter=100)

Configure a radial quantics tensor train (QTT) calculation on ``2^q`` uniformly spaced
interior points, where `q = quantics`. `r₀` and `rₘₐₓ` are zero-value (Dirichlet)
boundaries and are excluded from the grid, so Coulomb and centrifugal potentials are
never evaluated at the origin. `tolerance`, `maxbonddim`, and `maxiter` are forwarded
to tensor cross interpolation.
""" QuanticsTensorTrainMethod

@doc raw"""
    solve(hamiltonian, wavefunction, method::QuanticsTensorTrainMethod; info=1)

Compress the Hamiltonian and trial radial wave function as QTTs and evaluate the
Rayleigh quotient for the reduced radial wave function ``u(r)=r\psi(r)``,

```math
E = \frac{\langle u | H_u | u \rangle}{\langle u | u \rangle},
\qquad
H_u = -\frac{\hbar^2}{2m}
\left[\frac{\mathrm{d}^2}{\mathrm{d}r^2}-\frac{l(l+1)}{r^2}\right]+V(r).
```

The finite-difference kinetic operator is represented directly as a rank-three QTT
matrix; functions of ``r`` are compressed with tensor cross interpolation.
""" solve(hamiltonian::Hamiltonian, wavefunction::Function, method::QuanticsTensorTrainMethod; info=1)

@doc raw"""
    qttvalue(tt::QTTVector, index)
    qttvalue(tt::QTTMatrix, row, column)

Contract a QTT at one vector or matrix index without materializing its ``2^q`` entries.
""" qttvalue
