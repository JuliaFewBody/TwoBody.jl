```@meta
CurrentModule = TwoBody
```

# Database

The internal database provides benchmark Hamiltonians and reference energies
for testing solvers. Each Hamiltonian can be passed to a solver and its result
compared with the reference energy. The database is not exported because it is
intended for package development.

## Usage

Retrieve a benchmark using its key:

```julia-repl
julia> entry = TwoBody.db(:hydrogen)

julia> entry.hamiltonian

julia> entry.energy
```

Add a benchmark using `TwoBody.put!`:

```julia-repl
julia> hamiltonian = Hamiltonian(
           NonRelativisticKinetic(ℏ = 1.0, m = 1.0),
           CoulombPotential(coefficient = -1.0),
       )

julia> TwoBody.put!(:example, hamiltonian, -0.5)
```

Duplicate keys are rejected, and Hamiltonians are copied on registration and
lookup to protect stored benchmarks.

## Data

| Key | System | Reference energy |
|:--|:--|--:|
| `:hydrogen` | Hydrogen ground state | `-0.5` |
| `:positronium` | Positronium ground state | `-0.25` |
| `:harmonic_oscillator` | Three-dimensional harmonic oscillator ground state | `1.5` |

## API

```@docs; canonical=false
TwoBody.DatabaseEntry
TwoBody.put!
TwoBody.db
TwoBody.dbkeys
```
