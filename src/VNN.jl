export VariationalNeuralNetwork, VNN

import Random

_softplus(x) = max(x, zero(x)) + log1p(exp(-abs(x)))

struct VariationalNeuralNetwork{F<:FiniteDifferenceMethod,A,I,O,T<:AbstractFloat}
  fdm::F
  architecture::Vector{Int}
  activation::A
  init::I
  optimizer::O
  maxiters::Int
  abstol::T
  patience::Int
  every::Int

  function VariationalNeuralNetwork(;
    fdm::Union{Nothing,FiniteDifferenceMethod}=nothing,
    Δr::Union{Nothing,Real}=nothing,
    rₘₐₓ::Union{Nothing,Real}=nothing,
    R::Union{Nothing,AbstractRange}=nothing,
    l::Union{Nothing,Int}=nothing,
    direction::Union{Nothing,Symbol}=nothing,
    solver::Union{Nothing,Symbol}=nothing,
    architecture::AbstractVector{<:Integer}=[2],
    activation=_softplus,
    init=nothing,
    optimizer=nothing,
    maxiters::Int=1_000,
    abstol::Real=1e-8,
    patience::Int=10,
    every::Int=100,
  )
    if isnothing(fdm)
      spacing = something(Δr, 0.1)
      maximum_radius = something(rₘₐₓ, 50.0)
      grid = isnothing(R) ? (spacing:spacing:maximum_radius) : R
      fdm = FiniteDifferenceMethod(
        Δr=spacing,
        rₘₐₓ=maximum_radius,
        R=grid,
        l=something(l, 0),
        direction=something(direction, :c),
        solver=something(solver, :LinearAlgebra),
      )
    elseif any(value -> !isnothing(value), (Δr, rₘₐₓ, R, l, direction, solver))
      throw(ArgumentError("pass either fdm or finite-difference keywords, not both"))
    end

    _validate_vnn_fdm(fdm)
    isempty(architecture) && throw(ArgumentError("architecture must not be empty"))
    all(0 .< architecture) ||
      throw(ArgumentError("all hidden-layer widths must be positive"))
    0 <= maxiters || throw(ArgumentError("maxiters must be nonnegative"))
    isfinite(abstol) && 0 <= abstol ||
      throw(ArgumentError("abstol must be nonnegative and finite"))
    0 < patience || throw(ArgumentError("patience must be positive"))
    0 < every || throw(ArgumentError("every must be positive"))

    tolerance = float(abstol)
    new{
      typeof(fdm),
      typeof(activation),
      typeof(init),
      typeof(optimizer),
      typeof(tolerance),
    }(
      fdm,
      Int.(architecture),
      activation,
      init,
      optimizer,
      maxiters,
      tolerance,
      patience,
      every,
    )
  end
end

const VNN = VariationalNeuralNetwork

function _validate_vnn_fdm(fdm::FiniteDifferenceMethod)
  isfinite(fdm.Δr) && 0 < fdm.Δr ||
    throw(ArgumentError("fdm.Δr must be positive and finite"))
  isfinite(fdm.rₘₐₓ) && 0 < fdm.rₘₐₓ ||
    throw(ArgumentError("fdm.rₘₐₓ must be positive and finite"))
  isempty(fdm.R) && throw(ArgumentError("fdm.R must not be empty"))
  all(isfinite, fdm.R) || throw(ArgumentError("fdm.R must contain only finite values"))
  all(0 .< fdm.R) || throw(ArgumentError("fdm.R must contain only positive values"))
  0 <= fdm.l || throw(ArgumentError("fdm.l must be nonnegative"))
  fdm.direction in (:c, :f, :b) ||
    throw(ArgumentError("fdm.direction must be :c, :f, or :b"))
  return nothing
end

Base.string(method::VariationalNeuralNetwork) =
  "VariationalNeuralNetwork(" *
  join(["$(symbol)=$(getproperty(method, symbol))" for symbol in fieldnames(typeof(method))], ", ") *
  ")"
Base.show(io::IO, method::VariationalNeuralNetwork) = print(io, Base.string(method))

function _vnn_unavailable()
  throw(ArgumentError(
    "VNN support requires Lux.jl, Optimisers.jl, and Zygote.jl. Run " *
    "`Pkg.add([\"Lux\", \"Optimisers\", \"Zygote\"])`, then load them with " *
    "`using Lux, Optimisers, Zygote`.",
  ))
end

function _vnn_extension()
  extension = Base.get_extension(@__MODULE__, :TwoBodyVNNExt)
  isnothing(extension) && _vnn_unavailable()
  return extension
end

function solve(
  hamiltonian::Hamiltonian,
  method::VariationalNeuralNetwork;
  kwargs...,
)
  return _vnn_extension().solve_vnn(hamiltonian, method; kwargs...)
end

function solve(
  hamiltonian::Hamiltonian,
  model,
  method::VariationalNeuralNetwork;
  rng::Random.AbstractRNG=Random.MersenneTwister(123),
  parameters=nothing,
  states=nothing,
  optimizer_state=nothing,
  trial=(r, value) -> value,
  info::Int=0,
)
  return _vnn_extension().solve_vnn(
    hamiltonian,
    model,
    method;
    rng=rng,
    parameters=parameters,
    states=states,
    optimizer_state=optimizer_state,
    trial=trial,
    info=info,
  )
end

@doc raw"""
    VariationalNeuralNetwork(; fdm=nothing, Δr=nothing, rₘₐₓ=nothing,
      R=nothing, l=nothing, direction=nothing, solver=nothing,
      architecture=[2], activation=softplus, init=nothing, optimizer=nothing,
      maxiters=1000, abstol=1e-8, patience=10, every=100)

Options for optimizing a Lux neural network as a radial trial wavefunction.
`VNN` is an abbreviation for `VariationalNeuralNetwork`. VNN support is activated
by loading Lux.jl, Optimisers.jl, and Zygote.jl. If `init` or `optimizer` is
`nothing`, the extension uses `Lux.glorot_normal` or `Optimisers.Adam(0.01)`,
respectively.
""" VariationalNeuralNetwork

@doc raw"""
    solve(hamiltonian, method::VariationalNeuralNetwork; kwargs...)

Build the standard Lux model specified by `method.architecture` and minimize its
finite-difference Rayleigh quotient. See the three-argument overload to supply a
custom Lux model.
""" solve(hamiltonian::Hamiltonian, method::VariationalNeuralNetwork; kwargs...)

@doc raw"""
    solve(hamiltonian, model, method::VariationalNeuralNetwork;
          rng=Random.MersenneTwister(123), parameters=nothing, states=nothing,
          optimizer_state=nothing, trial=(r, value) -> value, info=0)

Minimize the finite-difference Rayleigh quotient with respect to the parameters of
a Lux model. The model receives the complete radial grid as a batch and must return
one real value per grid point. Pass returned `parameters`, `states`, and optionally
`optimizer_state` to continue training.
""" solve(
  hamiltonian::Hamiltonian,
  model,
  method::VariationalNeuralNetwork;
  rng::Random.AbstractRNG=Random.MersenneTwister(123),
  parameters=nothing,
  states=nothing,
  optimizer_state=nothing,
  trial=(r, value) -> value,
  info::Int=0,
)
