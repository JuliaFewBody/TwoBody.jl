export DiffusionMonteCarlo, ResultDiffusionMonteCarlo

import LinearAlgebra
import Random

struct DiffusionMonteCarlo{T<:AbstractFloat}
  n_steps::Int
  equilibration::Int
  n_walkers::Int
  Δt::T
  initial_scale::T
  feedback::T

  function DiffusionMonteCarlo(;
    n_steps::Int=2_000,
    equilibration::Int=500,
    n_walkers::Int=1_000,
    Δt::Real=0.01,
    initial_scale::Real=1.0,
    feedback::Real=0.1,
  )
    0 < n_steps || throw(ArgumentError("n_steps must be positive"))
    0 ≤ equilibration < n_steps ||
      throw(ArgumentError("equilibration must satisfy 0 ≤ equilibration < n_steps"))
    0 < n_walkers || throw(ArgumentError("n_walkers must be positive"))
    isfinite(Δt) && 0 < Δt || throw(ArgumentError("Δt must be positive and finite"))
    isfinite(initial_scale) && 0 < initial_scale ||
      throw(ArgumentError("initial_scale must be positive and finite"))
    isfinite(feedback) && 0 < feedback ≤ 1 ||
      throw(ArgumentError("feedback must satisfy 0 < feedback ≤ 1"))
    type = promote_type(typeof(float(Δt)), typeof(float(initial_scale)), typeof(float(feedback)))
    new{type}(n_steps, equilibration, n_walkers, Δt, initial_scale, feedback)
  end
end

struct ResultDiffusionMonteCarlo
  data::Any
  ResultDiffusionMonteCarlo(; args...) = new(NamedTuple(Dict(args)))
end

Base.getproperty(result::ResultDiffusionMonteCarlo, symbol::Symbol) =
  Base.getproperty(getfield(result, :data), symbol)

Base.string(method::DiffusionMonteCarlo) =
  "DiffusionMonteCarlo(" *
  join(["$(symbol)=$(getproperty(method, symbol))" for symbol in fieldnames(typeof(method))], ", ") *
  ")"

function Base.string(result::ResultDiffusionMonteCarlo)
  return "# method\n\n$(result.method)\n\n# energy\n\nE = $(result.E)\n"
end

Base.show(io::IO, method::DiffusionMonteCarlo) = print(io, Base.string(method))
Base.show(io::IO, result::ResultDiffusionMonteCarlo) = print(io, Base.string(result))

_dmc_diffusion(term::Kinetic) = term.hbar^2 / (2 * term.m)
_dmc_diffusion(term::Laplacian) = -term.coefficient
_dmc_diffusion(::RestEnergy) = 0.0

function _dmc_diffusion(term::KineticTerm)
  throw(ArgumentError("$(typeof(term)) is not supported by DiffusionMonteCarlo"))
end

_dmc_potential(term::RestEnergy, radius) = term.m * term.c^2
_dmc_potential(term::PotentialTerm, radius) = V(term, radius)

function _dmc_potential(term::Union{Delta,Tabulated}, radius)
  throw(ArgumentError("$(typeof(term)) is not supported by DiffusionMonteCarlo"))
end

_dmc_potential(::KineticTerm, radius) = 0.0

function _dmc_potential(hamiltonian::Hamiltonian, walkers::AbstractMatrix)
  potential = zeros(eltype(walkers), size(walkers, 2))
  for term in hamiltonian.terms
    for index in axes(walkers, 2)
      potential[index] += _dmc_potential(term, LinearAlgebra.norm(@view walkers[:,index]))
    end
  end
  all(isfinite, potential) ||
    throw(ArgumentError("the potential energy must be finite at every walker"))
  return potential
end

function _dmc_resample(weights, rng::Random.AbstractRNG)
  total = sum(weights)
  isfinite(total) && 0 < total || throw(ArgumentError("branching weights must have a finite positive sum"))
  cumulative = cumsum(weights ./ total)
  n_walkers = length(weights)
  offset = rand(rng) / n_walkers
  indices = Vector{Int}(undef, n_walkers)
  source = 1
  for target in 1:n_walkers
    position = offset + (target - 1) / n_walkers
    while cumulative[source] < position
      source += 1
    end
    indices[target] = source
  end
  return indices
end

function solve(
  hamiltonian::Hamiltonian,
  method::DiffusionMonteCarlo;
  rng::Random.AbstractRNG=Random.MersenneTwister(123),
  initial=nothing,
)
  diffusion = 0.0
  for term in hamiltonian.terms
    term isa KineticTerm && (diffusion += _dmc_diffusion(term))
  end
  isfinite(diffusion) && 0 < diffusion ||
    throw(ArgumentError("the Hamiltonian must have a positive nonrelativistic diffusion coefficient"))

  walkers = if isnothing(initial)
    method.initial_scale .* randn(rng, typeof(method.Δt), 3, method.n_walkers)
  else
    positions = Matrix{typeof(method.Δt)}(initial)
    size(positions) == (3, method.n_walkers) ||
      throw(DimensionMismatch("initial must have size (3, $(method.n_walkers))"))
    all(isfinite, positions) || throw(ArgumentError("initial positions must be finite"))
    positions
  end

  potential = _dmc_potential(hamiltonian, walkers)
  reference_energy = sum(potential) / length(potential)
  history = Vector{typeof(reference_energy)}(undef, method.n_steps)
  σ = sqrt(2 * diffusion * method.Δt)

  for step in 1:method.n_steps
    proposed = walkers .+ σ .* randn(rng, typeof(method.Δt), size(walkers))
    proposed_potential = _dmc_potential(hamiltonian, proposed)
    branching_potential = (potential .+ proposed_potential) ./ 2
    log_weights = -method.Δt .* (branching_potential .- reference_energy)
    shift = maximum(log_weights)
    weights = exp.(log_weights .- shift)
    log_mean_weight = shift + log(sum(weights) / length(weights))
    growth_energy = reference_energy - log_mean_weight / method.Δt
    reference_energy += method.feedback * (growth_energy - reference_energy)

    indices = _dmc_resample(weights, rng)
    walkers = proposed[:,indices]
    potential = proposed_potential[indices]
    history[step] = reference_energy
  end

  retained = @view history[method.equilibration + 1:end]
  energy = sum(retained) / length(retained)
  variance = length(retained) == 1 ? zero(energy) :
    sum((value - energy)^2 for value in retained) / (length(retained) - 1)
  standard_error = sqrt(variance / length(retained))
  return ResultDiffusionMonteCarlo(;
    hamiltonian, method, E=energy, variance, standard_error,
    reference_energies=history, walkers,
  )
end

@doc raw"""
`DiffusionMonteCarlo(; n_steps=2000, equilibration=500, n_walkers=1000, Δt=0.01, initial_scale=1.0, feedback=0.1)`

Configure DMC sampling parameters.
""" DiffusionMonteCarlo

@doc raw"""
`ResultDiffusionMonteCarlo`

Result of a diffusion Monte Carlo calculation.
""" ResultDiffusionMonteCarlo

@doc raw"""
`solve(hamiltonian, method::DiffusionMonteCarlo; rng=Random.MersenneTwister(123), initial=nothing)`

Run the calculation and return a `ResultDiffusionMonteCarlo`.
""" solve(
  hamiltonian::Hamiltonian,
  method::DiffusionMonteCarlo;
  rng::Random.AbstractRNG=Random.MersenneTwister(123),
  initial=nothing,
)
