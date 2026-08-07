export VariationalMonteCarlo, local_energy

import ForwardDiff
import LinearAlgebra
import Random

struct VariationalMonteCarlo{T<:AbstractFloat,V<:AbstractVector{T}}
  n_steps::Int
  burn_in::Int
  thinning::Int
  δ::T
  r₀::V

  function VariationalMonteCarlo(;
    n_steps::Int=10^5,
    burn_in::Int=10^3,
    thinning::Int=1,
    δ::Real=0.5,
    r₀::AbstractVector{<:Real}=[1.0, 0.0, 0.0],
  )
    0 < n_steps || throw(ArgumentError("n_steps must be positive"))
    0 ≤ burn_in || throw(ArgumentError("burn_in must be nonnegative"))
    0 < thinning || throw(ArgumentError("thinning must be positive"))
    isfinite(δ) && 0 < δ || throw(ArgumentError("δ must be positive and finite"))
    isempty(r₀) && throw(ArgumentError("r₀ must contain at least one coordinate"))
    all(isfinite, r₀) || throw(ArgumentError("r₀ must contain only finite coordinates"))

    position = float.(collect(r₀))
    step_size = convert(eltype(position), δ)
    new{eltype(position),typeof(position)}(n_steps, burn_in, thinning, step_size, position)
  end
end

Base.string(method::VariationalMonteCarlo) =
  "VariationalMonteCarlo(" *
  join(["$(symbol)=$(getproperty(method, symbol))" for symbol in fieldnames(typeof(method))], ", ") *
  ")"
Base.show(io::IO, method::VariationalMonteCarlo) = print(io, Base.string(method))

function _laplacian(wavefunction::Function, position::AbstractVector{<:Real})
  return LinearAlgebra.tr(ForwardDiff.hessian(wavefunction, position))
end

function _local_energy(term::Laplacian, wavefunction::Function, position, ψ)
  return term.coefficient * _laplacian(wavefunction, position) / ψ
end

function _local_energy(term::NonRelativisticKinetic, wavefunction::Function, position, ψ)
  return -term.ℏ^2 / (2 * term.m) * _laplacian(wavefunction, position) / ψ
end

_local_energy(term::RestEnergy, wavefunction::Function, position, ψ) = term.m * term.c^2
_local_energy(term::PotentialTerm, wavefunction::Function, position, ψ) =
  V(term, LinearAlgebra.norm(position))

function _local_energy(term::KineticTerm, wavefunction::Function, position, ψ)
  throw(ArgumentError("$(typeof(term)) is not supported by VariationalMonteCarlo"))
end

function local_energy(
  hamiltonian::Hamiltonian,
  wavefunction::Function,
  position::AbstractVector{<:Real},
)
  ψ = wavefunction(position)
  return sum(_local_energy(term, wavefunction, position, ψ) for term in hamiltonian.terms)
end

function _probability_density(wavefunction::Function, position)
  probability = abs2(wavefunction(position))
  probability isa Real || throw(ArgumentError("abs2(wavefunction(r)) must be real"))
  return probability
end

function _metropolis_samples(
  wavefunction::Function,
  method::VariationalMonteCarlo,
  rng::Random.AbstractRNG,
)
  position = copy(method.r₀)
  probability = _probability_density(wavefunction, position)
  isfinite(probability) && 0 < probability ||
    throw(ArgumentError("the probability density at r₀ must be positive and finite"))

  samples = Matrix{eltype(position)}(undef, length(position), method.n_steps)
  accepted = 0
  attempted = 0
  stored = 0
  n_transitions = method.burn_in + method.n_steps * method.thinning
  half = eltype(position)(1 // 2)

  for step in 1:n_transitions
    proposal = position .+ method.δ .* (rand(rng, eltype(position), length(position)) .- half)
    proposed_probability = _probability_density(wavefunction, proposal)
    attempted += 1

    if isfinite(proposed_probability) && 0 ≤ proposed_probability
      acceptance_probability = min(one(probability), proposed_probability / probability)
      if rand(rng) < acceptance_probability
        position = proposal
        probability = proposed_probability
        accepted += 1
      end
    end

    if method.burn_in < step && (step - method.burn_in) % method.thinning == 0
      stored += 1
      samples[:, stored] = position
    end
  end

  return samples, accepted / attempted
end

function _isfinite_number(value::Number)
  return isfinite(real(value)) && isfinite(imag(value))
end

function solve(
  hamiltonian::Hamiltonian,
  wavefunction::Function,
  method::VariationalMonteCarlo;
  rng::Random.AbstractRNG=Random.default_rng(),
  info::Int=1,
)
  samples, acceptance_rate = _metropolis_samples(wavefunction, method, rng)
  energies = [local_energy(hamiltonian, wavefunction, samples[:, i]) for i in axes(samples, 2)]
  finite_energies = filter(_isfinite_number, energies)
  isempty(finite_energies) &&
    throw(ArgumentError("all local energies are non-finite; check the wavefunction and initial position"))

  energy = sum(finite_energies) / length(finite_energies)
  variance = if length(finite_energies) == 1
    zero(abs2(first(finite_energies)))
  else
    sum(abs2(value - energy) for value in finite_energies) / (length(finite_energies) - 1)
  end
  standard_error = sqrt(variance / length(finite_energies))

  result = (
    hamiltonian=hamiltonian,
    method=method,
    E=energy,
    variance=variance,
    standard_error=standard_error,
    acceptance_rate=acceptance_rate,
    n_samples=length(finite_energies),
    n_discarded=length(energies) - length(finite_energies),
    local_energies=energies,
    samples=samples,
  )

  if 0 < info
    println("\n# method\n")
    println(method)
    println("\n# energy\n")
    println("E = $(result.E) ± $(result.standard_error)")
    println("\n# sampling\n")
    println("acceptance rate = $(result.acceptance_rate)")
    println("finite local energies = $(result.n_samples) / $(method.n_steps)")
  end

  return result
end

@doc raw"""
`VariationalMonteCarlo(n_steps=10^5, burn_in=10^3, thinning=1, δ=0.5, r₀=[1.0, 0.0, 0.0])`

Options for variational Monte Carlo with a symmetric, uniform Metropolis proposal.
The sampler targets ``|\psi(\mathbf{r})|^2``. `n_steps` is the number of retained
samples, `burn_in` is the number of discarded initial transitions, `thinning` is
the number of transitions between retained samples, `δ` is the proposal-box
width, and `r₀` is the initial position.
""" VariationalMonteCarlo

@doc raw"""
`local_energy(hamiltonian, wavefunction, position)`

Evaluate

```math
E_\mathrm{loc}(\mathbf{r}) =
\frac{\hat{H}\psi(\mathbf{r})}{\psi(\mathbf{r})}.
```

The Laplacian of a real-valued trial wavefunction is evaluated with
`ForwardDiff.hessian`. Non-relativistic kinetic, Laplacian, rest-energy, and
potential terms with a defined `V` method are supported.
""" local_energy

@doc raw"""
`solve(hamiltonian, wavefunction, method::VariationalMonteCarlo; rng=Random.default_rng(), info=1)`

Estimate the variational energy by averaging the local energy over Metropolis
samples from ``|\psi|^2``. A seeded random-number generator can be supplied with
`rng`. Non-finite local energies, which can occur at a measure-zero singularity
such as the origin of a Coulomb potential, are excluded and reported as
`n_discarded` in the result.
""" solve(
  hamiltonian::Hamiltonian,
  wavefunction::Function,
  method::VariationalMonteCarlo;
  rng::Random.AbstractRNG=Random.default_rng(),
  info::Int=1,
)
