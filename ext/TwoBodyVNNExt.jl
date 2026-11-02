module TwoBodyVNNExt

using TwoBody
import Lux
import Optimisers
import Printf
import Random
import Zygote
import TwoBody: _jacobian, _normalization, _rayleigh_quotient

_resolved_init(method::VariationalNeuralNetwork) =
  isnothing(method.init) ? Lux.glorot_normal : method.init

_resolved_optimizer(method::VariationalNeuralNetwork) =
  isnothing(method.optimizer) ? Optimisers.Adam(0.01) : method.optimizer

function _neural_network(method::VariationalNeuralNetwork)
  dimensions = [1; method.architecture; 1]
  init = _resolved_init(method)
  layers = [
    Lux.Dense(
      dimensions[index],
      dimensions[index + 1],
      method.activation;
      init_weight=init,
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

function solve_vnn(
  hamiltonian::Hamiltonian,
  method::VariationalNeuralNetwork;
  kwargs...,
)
  return solve_vnn(hamiltonian, _neural_network(method), method; kwargs...)
end

function solve_vnn(
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
  H = TwoBody.matrix(hamiltonian, fdm)
  J = _jacobian(fdm)
  input = _network_input(parameters, fdm.R)

  values, states = _trial_values(model, parameters, states, input, trial)
  energy = _rayleigh_quotient(values, H, J)
  _validate_trial(values, energy)

  history = [energy]
  optimizer_state = isnothing(optimizer_state) ?
    Optimisers.setup(_resolved_optimizer(method), parameters) : optimizer_state
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

end
