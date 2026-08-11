```@meta
CurrentModule = TwoBody
```

# Variational Neural Network

`VariationalNeuralNetwork` (`VNN`) uses a neural network defined with
[Lux.jl](https://lux.csail.mit.edu/) as a radial trial wavefunction. It minimizes
the finite-difference Rayleigh quotient

```math
E[\psi_\theta] =
\frac{\pmb{\psi}_\theta^\mathsf{T}\pmb{J}\pmb{H}\pmb{\psi}_\theta}
     {\pmb{\psi}_\theta^\mathsf{T}\pmb{J}\pmb{\psi}_\theta},
```

where the grid, Hamiltonian matrix ``\pmb{H}``, and radial Jacobian ``\pmb{J}``
are provided by `FiniteDifferenceMethod`.

## Standard model

The two-argument `solve` method constructs a Lux network from `architecture`.

```@example vnn
using TwoBody

H = Hamiltonian(
  Kinetic(hbar=1, m=1),
  Coulomb(coefficient=-1),
)

method = VNN(
  Δr=0.2,
  rₘₐₓ=4.0,
  architecture=[4],
  maxiters=100,
  every=25,
  abstol=0,
)

result = solve(H, method; info=1)
result.E
```

The returned normalized radial wavefunction is callable.

```@example vnn
result.wavefunction(1.0)
```

## Custom Lux model

An arbitrary Lux model can be passed explicitly. The model receives radii as a
`1 × number_of_grid_points` batch and must return one real value per radius.

```julia
using Lux
using Optimisers
using Random

model = Lux.Chain(
  Lux.Dense(1 => 8, tanh),
  Lux.Dense(8 => 1),
)

method = VNN(
  fdm=FiniteDifferenceMethod(Δr=0.1, rₘₐₓ=20.0),
  optimizer=Optimisers.Adam(0.01),
  maxiters=2_000,
)

result = solve(
  H,
  model,
  method;
  rng=Random.MersenneTwister(123),
  trial=(r, value) -> exp(-r) * value,
  info=1,
)
```

`trial` can impose an envelope or boundary condition. To continue training,
pass `result.parameters`, `result.states`, and optionally
`result.optimizer_state` to another call. The result also contains normalized
grid values `ψ`, raw values `raw_ψ`, `history`, `n_iterations`, and `converged`.

## API reference

```@docs; canonical=false
TwoBody.VariationalNeuralNetwork
TwoBody.solve(hamiltonian::Hamiltonian, method::VariationalNeuralNetwork)
TwoBody.solve(
  hamiltonian::Hamiltonian,
  model,
  method::VariationalNeuralNetwork;
  rng::Random.AbstractRNG,
  parameters,
  states,
  optimizer_state,
  trial,
  info::Int,
)
```
