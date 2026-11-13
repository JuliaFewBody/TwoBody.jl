# Shared physical terms are lowered to the existing relative-coordinate kernels.
function _shared_pair(i, j)
  (i == 1 && j == 2 || i == 2 && j == 1) || throw(ArgumentError(
    "TwoBody requires the particle pair (1, 2), got ($i, $j)",
  ))
end

function _shared_term(t::FBH.Kinetic)
  length(t.masses) == 2 || throw(ArgumentError("TwoBody requires exactly two particle masses"))
  return Kinetic(hbar=1, m=inv(sum(inv, t.masses)))
end

function _shared_term(t::FBH.Coulomb)
  _shared_pair(t.i, t.j)
  t.n_charges in (0, 2) || throw(ArgumentError("TwoBody requires exactly two charges"))
  return Coulomb(coefficient=t.coefficient)
end

function _shared_term(t::FBH.Gaussian)
  for (i, j, _) in t.factors
    _shared_pair(i, j)
  end
  return Gaussian(coefficient=t.coefficient, exponent=sum(factor[3] for factor in t.factors))
end

function _shared_term(t::FBH.ManyBodyGaussian)
  size(t.G) == (2, 2) || throw(ArgumentError("TwoBody requires a 2×2 Gaussian exponent matrix"))
  return Gaussian(coefficient=t.coefficient, exponent=t.G[1, 1])
end

function _shared_term(t::FBH.Oscillator)
  _shared_pair(t.i, t.j)
  return PowerLaw(coefficient=t.coefficient, exponent=2)
end

function _shared_term(t::FBH.Custom)
  _shared_pair(t.i, t.j)
  return Custom(r -> t.coefficient * t.f(r))
end

_shared_term(t::FBH.Operator) = throw(ArgumentError("TwoBody does not support shared term $(typeof(t))"))

"""
    Hamiltonian(H::FewBodyHamiltonians.Hamiltonian)

Lower a shared two-particle Hamiltonian to TwoBody's relative-coordinate terms.
Masses use the shared convention ℏ=1, including masses supplied with nuclear unit
labels. Potential-only sums are accepted for perturbations. `Custom` quadrature
keywords are solver hints; TwoBody uses its backend's quadrature settings.
"""
function Hamiltonian(H::FBH.Hamiltonian)
  if any(t -> t isa FBH.Kinetic, H.terms)
    FBH.validate(H)
  end
  return Hamiltonian((_shared_term(t) for t in H.terms)...)
end

_native_hamiltonian(H::Hamiltonian) = H
_native_hamiltonian(H::FBH.Hamiltonian) = Hamiltonian(H)

function _shared_perturbation(H, perturbation)
  if perturbation isa FBH.Hamiltonian
    if any(t -> t isa FBH.Kinetic, perturbation.terms)
      FBH.kinetic(perturbation).units == FBH.kinetic(H).units || throw(ArgumentError(
        "the Hamiltonian and its perturbation must use the same units",
      ))
    else
      FBH.validate(FBH.Hamiltonian(FBH.kinetic(H), perturbation.terms...))
    end
  end
  return _native_hamiltonian(perturbation)
end

function solve(H::FBH.Hamiltonian, args...; perturbation=nothing, kwargs...)
  FBH.kinetic(H)
  native = Hamiltonian(H)
  return isnothing(perturbation) ? solve(native, args...; kwargs...) :
    solve(native, args...; perturbation=_shared_perturbation(H, perturbation), kwargs...)
end

function optimize(H::FBH.Hamiltonian, args...; perturbation=Hamiltonian(), kwargs...)
  FBH.kinetic(H)
  return optimize(Hamiltonian(H), args...; perturbation=_shared_perturbation(H, perturbation), kwargs...)
end

for Method in (BasisSet, FiniteDifferenceMethod, QuanticsTensorTrainMethod)
  @eval matrix(H::FBH.Hamiltonian, method::$Method) = matrix(Hamiltonian(H), method)
end
element(H::FBH.Hamiltonian, bra::Basis, ket::Basis) = element(Hamiltonian(H), bra, ket)
FC(H::FBH.Hamiltonian, basis; kwargs...) = FC(Hamiltonian(H), basis; kwargs...)
local_energy(H::FBH.Hamiltonian, wavefunction::Function, position::AbstractVector{<:Real}) =
  local_energy(Hamiltonian(H), wavefunction, position)
