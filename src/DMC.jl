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
  log_weight_cap::T

  function DiffusionMonteCarlo(;
    n_steps::Int=2_000,
    equilibration::Int=500,
    n_walkers::Int=1_000,
    Δt::Real=0.01,
    initial_scale::Real=1.0,
    feedback::Real=0.1,
    log_weight_cap::Real=1.0,
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
    0 < log_weight_cap ||
      throw(ArgumentError("log_weight_cap must be positive; use Inf to disable the cap"))
    type = promote_type(
      typeof(float(Δt)), typeof(float(initial_scale)), typeof(float(feedback)), typeof(float(log_weight_cap)),
    )
    new{type}(n_steps, equilibration, n_walkers, Δt, initial_scale, feedback, log_weight_cap)
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
  cumulative[end] = one(eltype(cumulative))  # guard against round-off below one
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
  growth = similar(history)
  σ = sqrt(2 * diffusion * method.Δt)

  for step in 1:method.n_steps
    proposed = walkers .+ σ .* randn(rng, typeof(method.Δt), size(walkers))
    proposed_potential = _dmc_potential(hamiltonian, proposed)
    branching_potential = (potential .+ proposed_potential) ./ 2
    # The cap stops a single walker close to a singularity of the potential
    # from taking over the population in one step.
    log_weights = min.(-method.Δt .* (branching_potential .- reference_energy), method.log_weight_cap)
    shift = maximum(log_weights)
    weights = exp.(log_weights .- shift)
    log_mean_weight = shift + log(sum(weights) / length(weights))
    growth_energy = reference_energy - log_mean_weight / method.Δt
    # The reference energy cancels in the normalized resampling and only sets
    # the scale of the weights; the feedback keeps it near the growth estimate.
    reference_energy += method.feedback * (growth_energy - reference_energy)

    indices = _dmc_resample(weights, rng)
    walkers = proposed[:,indices]
    potential = proposed_potential[indices]
    history[step] = reference_energy
    growth[step] = growth_energy
  end

  # The energy is the mean of the growth estimator after equilibration, and the
  # standard error comes from block averages, which absorb part of the
  # autocorrelation of the walk.
  retained = @view growth[method.equilibration + 1:end]
  energy = sum(retained) / length(retained)
  variance = length(retained) == 1 ? zero(energy) :
    sum((value - energy)^2 for value in retained) / (length(retained) - 1)
  n_blocks = min(20, length(retained))
  block = length(retained) ÷ n_blocks
  block_means = [sum(@view retained[(b - 1) * block + 1:b * block]) / block for b in 1:n_blocks]
  block_mean = sum(block_means) / n_blocks
  standard_error = n_blocks == 1 ? zero(energy) :
    sqrt(sum((value - block_mean)^2 for value in block_means) / (n_blocks - 1) / n_blocks)
  return ResultDiffusionMonteCarlo(;
    hamiltonian, method, E=energy, variance, standard_error,
    growth_energies=growth, reference_energies=history, walkers,
  )
end

@doc raw"""
`DiffusionMonteCarlo(; n_steps=2000, equilibration=500, n_walkers=1000, Δt=0.01, initial_scale=1.0, feedback=0.1, log_weight_cap=1.0)`

| Arguments | Default | Description |
| :-------- | :------ | :---------- |
| `n_steps`        | `2000`  | Number of imaginary-time steps. |
| `equilibration`  | `500`   | Steps discarded before the energy is averaged. |
| `n_walkers`      | `1000`  | Fixed population size. |
| `Δt`             | `0.01`  | Imaginary-time step; the diffusion and branching errors are of first order in ``\Delta t``. |
| `initial_scale`  | `1.0`   | Standard deviation of the initial Gaussian cloud of walkers when `initial` is not passed to `solve`. |
| `feedback`       | `0.1`   | Relaxation of the reference energy towards the growth estimate per step. |
| `log_weight_cap` | `1.0`   | Upper bound of the branching exponent ``-\Delta t(\bar V-E_\mathrm{ref})`` per step. It prevents a walker near a singularity of the potential from taking over the population; `Inf` disables the cap. |
""" DiffusionMonteCarlo

@doc raw"""
`ResultDiffusionMonteCarlo`

Result of a diffusion Monte Carlo calculation with the properties

| Property | Description |
| :------- | :---------- |
| `E`                  | Mean of the growth estimator over the steps after equilibration. |
| `variance`           | Sample variance of the growth estimator per step. |
| `standard_error`     | Standard error of `E` from 20 block averages. It still underestimates the spread between independent runs at the default settings. |
| `growth_energies`    | Growth estimator ``E_\mathrm{ref}-\log\langle w\rangle/\Delta t`` at every step. |
| `reference_energies` | Reference energy at every step. |
| `walkers`            | Final positions, a ``3\times n_\mathrm{walkers}`` matrix distributed as ``\psi_0`` (not ``|\psi_0|^2``). |
| `hamiltonian`, `method` | The inputs. |
""" ResultDiffusionMonteCarlo

@doc raw"""
`solve(hamiltonian, method::DiffusionMonteCarlo; rng=Random.MersenneTwister(123), initial=nothing)`

Run the calculation and return a `ResultDiffusionMonteCarlo`. `rng` seeds the
walk, which is reproducible within one Julia version. `initial` is an optional
``3\times n_\mathrm{walkers}`` matrix of starting positions. The Hamiltonian
must contain a nonrelativistic kinetic term (`Kinetic` or `Laplacian`), may
contain `RestEnergy` and potential terms with a finite `V(o, r)`; `Delta`,
`Tabulated`, and the relativistic kinetic terms are rejected.
""" solve(
  hamiltonian::Hamiltonian,
  method::DiffusionMonteCarlo;
  rng::Random.AbstractRNG=Random.MersenneTwister(123),
  initial=nothing,
)
