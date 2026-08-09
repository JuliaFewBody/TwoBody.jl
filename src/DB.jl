"""
    DatabaseEntry(hamiltonian, energy)

A benchmark problem stored in the database. `hamiltonian` is ready to be
passed to a solver and `energy` is its reference energy.
"""
struct DatabaseEntry{T<:Real}
    hamiltonian::Hamiltonian
    energy::T
end

import Base: put!

# Keeping the registry private makes `db` the single lookup boundary and leaves
# room for validation, lazy loading, and provenance.
const _DATABASE = Dict{Symbol,DatabaseEntry}()

"""
    put!(key, hamiltonian, energy)

Add a benchmark problem to the database. `key` may be a `Symbol` or string,
`hamiltonian` must be a [`Hamiltonian`](@ref), and `energy` must be real.
Registering the same key twice throws an `ArgumentError`.
"""
function put!(key::Symbol, hamiltonian::Hamiltonian, energy::T) where {T<:Real}
    haskey(_DATABASE, key) &&
        throw(ArgumentError("database key $(repr(key)) is already registered"))

    entry = DatabaseEntry(deepcopy(hamiltonian), energy)
    _DATABASE[key] = entry
    return entry
end

put!(key::AbstractString, hamiltonian::Hamiltonian, energy::Real) =
    put!(Symbol(key), hamiltonian, energy)

# PoC data in atomic units.
put!(
    :hydrogen,
    Hamiltonian(
        NonRelativisticKinetic(ℏ = 1.0, m = 1.0),
        CoulombPotential(coefficient = -1.0),
    ),
    -0.5,
)

put!(
    :positronium,
    Hamiltonian(
        NonRelativisticKinetic(ℏ = 1.0, m = 0.5),
        CoulombPotential(coefficient = -1.0),
    ),
    -0.25,
)

put!(
    :harmonic_oscillator,
    Hamiltonian(
        NonRelativisticKinetic(ℏ = 1.0, m = 1.0),
        PowerLawPotential(coefficient = 0.5, exponent = 2.0),
    ),
    1.5,
)

"""
    db(key::Union{Symbol,AbstractString}) -> DatabaseEntry

Return the benchmark Hamiltonian and reference energy associated with `key`.
The returned Hamiltonian is independent of the stored value and can safely be
modified by callers.

# Examples

```julia
entry = db(:hydrogen)
result = solve(entry.hamiltonian, method)
isapprox(result.values[1], entry.energy)
```
"""
function db(key::Symbol)
    haskey(_DATABASE, key) || throw(
        ArgumentError(
            "unknown database key $(repr(key)); available keys: " *
            join(repr.(dbkeys()), ", "),
        ),
    )

    entry = _DATABASE[key]
    return DatabaseEntry(deepcopy(entry.hamiltonian), entry.energy)
end

db(key::AbstractString) = db(Symbol(key))

"""
    dbkeys() -> Vector{Symbol}

Return the available database keys in deterministic order.
"""
dbkeys() = sort!(collect(keys(_DATABASE)); by = string)
