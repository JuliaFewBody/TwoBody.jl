```@meta
CurrentModule = TwoBody
```

# Database

The internal database provides benchmark Hamiltonians and their reference
energies for developing and testing solvers. Database functionality is not
exported because it is intended for package development.

The returned Hamiltonian can be passed to a solver, and the calculated energy
can then be compared with the reference value.

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

Duplicate keys are rejected. Registration and lookup both copy the Hamiltonian
so that modifying a returned entry does not alter the stored benchmark.

## Data

| Key | System | Reference energy |
|:--|:--|--:|
| `:hydrogen` | Hydrogen ground state | `-0.5` |
| `:positronium` | Positronium ground state | `-0.25` |
| `:harmonic_oscillator` | Three-dimensional harmonic oscillator ground state | `1.5` |

## API

```@docs
TwoBody.DatabaseEntry
TwoBody.put!
TwoBody.db
TwoBody.dbkeys
```
