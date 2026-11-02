export FC

_fc_candidates(o::Laplacian, basis::PowerSlaterBasis) =
  iszero(o.coefficient) ? PowerSlaterBasis[] : _laplacian_candidates(basis)
_fc_candidates(o::Kinetic, basis::PowerSlaterBasis) =
  iszero(o.hbar) ? PowerSlaterBasis[] : _laplacian_candidates(basis)
_fc_candidates(o::RestEnergy, basis::PowerSlaterBasis) =
  iszero(o.m * o.c^2) ? PowerSlaterBasis[] : PowerSlaterBasis[basis]
_fc_candidates(o::Constant, basis::PowerSlaterBasis) =
  iszero(o.constant) ? PowerSlaterBasis[] : PowerSlaterBasis[basis]
_fc_candidates(o::Linear, basis::PowerSlaterBasis) =
  iszero(o.coefficient) ? PowerSlaterBasis[] : PowerSlaterBasis[PowerSlaterBasis(basis.n+1, basis.a)]
_fc_candidates(o::Coulomb, basis::PowerSlaterBasis) =
  iszero(o.coefficient) ? PowerSlaterBasis[] : PowerSlaterBasis[PowerSlaterBasis(basis.n-1, basis.a)]

function _fc_candidates(o::PowerLaw, basis::PowerSlaterBasis)
  iszero(o.coefficient) && return PowerSlaterBasis[]
  isinteger(o.exponent) || throw(ArgumentError(
    "FC requires an integer PowerLaw exponent for PowerSlaterBasis, got $(o.exponent)",
  ))
  return PowerSlaterBasis[PowerSlaterBasis(basis.n + Int(o.exponent), basis.a)]
end

_fc_candidates(o::Exponential, basis::PowerSlaterBasis) =
  iszero(o.coefficient) ? PowerSlaterBasis[] : PowerSlaterBasis[PowerSlaterBasis(basis.n, basis.a + o.exponent)]
_fc_candidates(o::Yukawa, basis::PowerSlaterBasis) =
  iszero(o.coefficient) ? PowerSlaterBasis[] : PowerSlaterBasis[PowerSlaterBasis(basis.n-1, basis.a + o.exponent)]

function _fc_candidates(o::Operator, ::PowerSlaterBasis)
  throw(ArgumentError("FC does not support $(typeof(o)) with PowerSlaterBasis"))
end

function FC(hamiltonian::Hamiltonian, basis::PowerSlaterBasis; g::PowerSlaterBasis=PowerSlaterBasis(1, 0))
  candidates = PowerSlaterBasis[basis] # the -Eₙ term in g(H-Eₙ)φ
  for operator in hamiltonian.terms
    append!(candidates, _fc_candidates(operator, basis))
  end
  complements = unique(_multiply(candidate, g) for candidate in candidates)
  return BasisSet(filter(candidate -> isfinite(φ(candidate, 0.0)), complements)...)
end

function FC(hamiltonian::Hamiltonian, basisset::BasisSet; g::PowerSlaterBasis=PowerSlaterBasis(1, 0))
  complements = PowerSlaterBasis[]
  for basis in basisset.basis
    basis isa PowerSlaterBasis || throw(ArgumentError(
      "FC only supports BasisSet entries of type PowerSlaterBasis, got $(typeof(basis))",
    ))
    append!(complements, FC(hamiltonian, basis; g=g).basis)
  end
  return BasisSet(unique(complements)...)
end

@doc raw"""
`FC(hamiltonian, basis; g=PowerSlaterBasis(1, 0))`

Generate the next Free Complement basis from a `PowerSlaterBasis` or a
`BasisSet` of power-Slater functions by collecting the basis-function forms in
``g(H-E_n)\phi``. Numerical coefficients are ignored. The default scaling
function is ``g(r)=r``. Duplicate functions and functions singular at the
origin are removed.

The Hamiltonian may contain `Kinetic`, `Laplacian`, `RestEnergy`, `Constant`,
`Linear`, `Coulomb`, integer-exponent `PowerLaw`, `Exponential`, and `Yukawa`
terms. An `ArgumentError` is thrown when an operator does not map a
`PowerSlaterBasis` to power-Slater functions.

For the hydrogen Hamiltonian, repeated application starting from
`PowerSlaterBasis(0, 1.5)` adds one power of ``r`` at each iteration.
""" FC