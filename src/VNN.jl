export VariationalNeuralNetwork, VNN

import Lux
import Optimisers
import Printf
import Random
import Zygote

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
    init=Lux.glorot_normal,
    optimizer=Optimisers.Adam(0.01),
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

function _neural_network(method::VariationalNeuralNetwork)
  dimensions = [1; method.architecture; 1]
  layers = [
    Lux.Dense(
      dimensions[index],
      dimensions[index + 1],
      method.activation;
      init_weight=method.init,
    ) for index in 1:(length(dimensions) - 1)
  ]
  return Lux.Chain(layers...)
end

_parameter_eltype(x::AbstractArray) = eltype(x)
_parameter_eltype(x::Number) = typeof(x)

function _parameter_eltype(x::Union{NamedTuple,Tuple})
  for value in values(x)
    type = _parameter_eltype(value)
    isnothing(type) || return type
  end
  return nothing
end

_parameter_eltype(x) = nothing

function _network_input(parameters, R)
  type = _parameter_eltype(parameters)
  isnothing(type) && throw(ArgumentError("the Lux model must contain at least one parameter"))
  type <: Real || throw(ArgumentError("the Lux model parameters must be real-valued"))
  return reshape(type.(R), 1, :)
end

function _trial_values(model, parameters, states, input, trial)
  output, updated_states = Lux.apply(model, input, parameters, states)
  length(output) == length(input) || throw(DimensionMismatch(
    "the Lux model must return one wavefunction value for each radial-grid point",
  ))
  values = trial.(vec(input), vec(output))
  eltype(values) <: Real ||
    throw(ArgumentError("the trial wavefunction must be real-valued"))
  return values, updated_states
end

function _validate_trial(values, energy)
  all(isfinite, values) ||
    throw(ArgumentError("the trial wavefunction must contain only finite values"))
  any(value -> !iszero(value), values) ||
    throw(ArgumentError("the trial wavefunction must not be identically zero"))
  isfinite(energy) || throw(ArgumentError("the variational energy must be finite"))
  return nothing
end

function solve(
  hamiltonian::Hamiltonian,
  method::VariationalNeuralNetwork;
  kwargs...,
)
  return solve(hamiltonian, _neural_network(method), method; kwargs...)
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
  !isnothing(optimizer_state) && isnothing(parameters) &&
    throw(ArgumentError("parameters must be supplied with optimizer_state"))

  if isnothing(parameters) || isnothing(states)
    initialized_parameters, initialized_states = Lux.setup(rng, model)
    parameters = isnothing(parameters) ? initialized_parameters : parameters
    states = isnothing(states) ? initialized_states : states
  end

  fdm = method.fdm
  H = matrix(hamiltonian, fdm)
  J = _jacobian(fdm)
  input = _network_input(parameters, fdm.R)

  values, states = _trial_values(model, parameters, states, input, trial)
  energy = _rayleigh_quotient(values, H, J)
  _validate_trial(values, energy)

  history = [energy]
  optimizer_state = isnothing(optimizer_state) ?
    Optimisers.setup(method.optimizer, parameters) : optimizer_state
  converged = false
  stable_steps = 0
  n_iterations = 0

  if 0 < info
    println("\n# method\n")
    println(method)
    println("\n# optimization\n")
    Printf.@printf("%9s\t%14s\n", "iteration", "energy")
    Printf.@printf("%9d\t%+.12e\n", 0, energy)
  end

  for iteration in 1:method.maxiters
    gradients = first(Zygote.gradient(parameters) do candidate_parameters
      candidate_values, _ =
        _trial_values(model, candidate_parameters, states, input, trial)
      _rayleigh_quotient(candidate_values, H, J)
    end)
    optimizer_state, parameters =
      Optimisers.update(optimizer_state, parameters, gradients)

    previous_energy = energy
    values, states = _trial_values(model, parameters, states, input, trial)
    energy = _rayleigh_quotient(values, H, J)
    _validate_trial(values, energy)
    push!(history, energy)
    n_iterations = iteration

    if abs(energy - previous_energy) <= method.abstol
      stable_steps += 1
      converged = method.patience <= stable_steps
    else
      stable_steps = 0
    end

    if 0 < info && (iteration % method.every == 0 || converged || iteration == method.maxiters)
      Printf.@printf("%9d\t%+.12e\n", iteration, energy)
    end
    converged && break
  end

  normalization = _normalization(values, fdm, J)
  normalized_values = normalization * values
  parameter_type = _parameter_eltype(parameters)
  wavefunction = function (r::Real)
    scalar_input = reshape(parameter_type[r], 1, 1)
    output, _ = Lux.apply(model, scalar_input, parameters, states)
    return normalization * trial(r, only(output))
  end

  return (
    hamiltonian=hamiltonian,
    method=method,
    model=model,
    parameters=parameters,
    states=states,
    optimizer_state=optimizer_state,
    H=H,
    J=J,
    E=energy,
    ψ=normalized_values,
    raw_ψ=values,
    wavefunction=wavefunction,
    history=history,
    n_iterations=n_iterations,
    converged=converged,
  )
end

@doc raw"""
`VariationalNeuralNetwork(; fdm=nothing, Δr=nothing, rₘₐₓ=nothing, R=nothing, l=nothing, direction=nothing, solver=nothing, architecture=[2], activation=softplus, init=Lux.glorot_normal, optimizer=Optimisers.Adam(0.01), maxiters=1000, abstol=1e-8, patience=10, every=100)`

Options for optimizing a Lux neural network as a radial trial wavefunction.
`VNN` is an abbreviation for `VariationalNeuralNetwork`.

Pass an existing `FiniteDifferenceMethod` as `fdm`, or use the finite-difference
keywords directly. `architecture` defines the hidden-layer widths of the
standard Lux model used by `solve(hamiltonian, method)`. A custom Lux model can
instead be supplied with `solve(hamiltonian, model, method)`.
""" VariationalNeuralNetwork

@doc raw"""
`solve(hamiltonian, method::VariationalNeuralNetwork; kwargs...)`

Build the standard Lux model specified by `method.architecture` and minimize its
finite-difference Rayleigh quotient. See the three-argument overload to supply a
custom Lux model.
""" solve(hamiltonian::Hamiltonian, method::VariationalNeuralNetwork; kwargs...)

@doc raw"""
`solve(hamiltonian, model, method::VariationalNeuralNetwork; rng=Random.MersenneTwister(123), parameters=nothing, states=nothing, optimizer_state=nothing, trial=(r, value) -> value, info=0)`

Minimize the finite-difference Rayleigh quotient

```math
E[\psi_\theta] =
\frac{\pmb{\psi}_\theta^\mathsf{T}\pmb{J}\pmb{H}\pmb{\psi}_\theta}
     {\pmb{\psi}_\theta^\mathsf{T}\pmb{J}\pmb{\psi}_\theta}
```

with respect to the parameters of a Lux model. The model receives the complete
radial grid as a batch and must return one real value per grid point. `trial`
can impose an envelope or boundary condition on the raw output. Pass returned
`parameters`, `states`, and optionally `optimizer_state` to continue training.

The result contains the energy `E`, normalized grid values `ψ`, raw values
`raw_ψ`, a callable `wavefunction`, Lux variables, optimizer state, energy
`history`, `n_iterations`, and `converged`.
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
