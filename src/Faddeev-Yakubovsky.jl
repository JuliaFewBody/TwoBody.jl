export FaddeevYakubovsky

import LinearAlgebra
import Printf

# method

struct FaddeevYakubovsky
  basisset::BasisSet
  Eₘᵢₙ::Float64
  Eₘₐₓ::Float64
  tolerance::Float64
  maxiters::Int
  function FaddeevYakubovsky(basisset::BasisSet; Eₘᵢₙ=-1.0, Eₘₐₓ=0.0, tolerance=1e-10, maxiters=100)
    isempty(basisset.basis) && throw(ArgumentError("the basis set must not be empty"))
    isfinite(Eₘᵢₙ) && isfinite(Eₘₐₓ) && Eₘᵢₙ < Eₘₐₓ ||
      throw(ArgumentError("the energy interval must be finite and satisfy Eₘᵢₙ < Eₘₐₓ"))
    isfinite(tolerance) && tolerance > 0 ||
      throw(ArgumentError("tolerance must be positive and finite"))
    maxiters > 0 || throw(ArgumentError("maxiters must be positive"))
    new(basisset, Eₘᵢₙ, Eₘₐₓ, tolerance, maxiters)
  end
end

FaddeevYakubovsky(basis::Basis; options...) = FaddeevYakubovsky(BasisSet(basis); options...)
FaddeevYakubovsky(basisset::GeometricBasisSet; options...) =
  FaddeevYakubovsky(BasisSet(basisset.basis...); options...)

# result

struct ResultFaddeevYakubovsky
  data::Any
  ResultFaddeevYakubovsky(; args...) = new((; args...))
end

Base.getproperty(result::ResultFaddeevYakubovsky, symbol::Symbol) = getproperty(getfield(result, :data), symbol)
Base.haskey(result::ResultFaddeevYakubovsky, symbol::Symbol) = haskey(getfield(result, :data), symbol)
Base.show(io::IO, result::ResultFaddeevYakubovsky) = print(io, string(result))

function Base.string(result::ResultFaddeevYakubovsky)
  text = "ResultFaddeevYakubovsky:\n  E: $(result.E)\n"
  if result.info >= 0
    text *= "  kernel eigenvalue: $(result.λ)\n"
    text *= "  iterations: $(result.iterations)\n"
  end
  return text
end

function ψ(result::ResultFaddeevYakubovsky, r; n::Int=1)
  return sum(result.C[i,n] * φ(result.basisset[i], r) for i in eachindex(result.basisset.basis))
end

# kernel

function _faddeev_yakubovsky_kernel(T, V, S, E)
  # (T - E*S) = L*L'; K = -L⁻¹*V*L⁻ᵀ; c = L⁻ᵀ*y.
  L = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(T - E * S)).L
  K = LinearAlgebra.Symmetric(-(L \ V) / L')
  λ, Y = LinearAlgebra.eigen(K)
  return (λ=λ[end], c=L' \ Y[:,end])
end

# solver

function solve(hamiltonian::Hamiltonian, method::FaddeevYakubovsky; perturbation=Hamiltonian(), info=4)
  # matrix element
  basisset = method.basisset
  S = matrix(basisset)
  LinearAlgebra.isposdef(S) || throw(ArgumentError("the overlap matrix must be positive definite"))
  T = zeros(size(S))
  V = zeros(size(S))
  any(term -> term isa KineticTerm, hamiltonian.terms) ||
    throw(ArgumentError("the Hamiltonian must contain a kinetic term"))
  for term in hamiltonian.terms
    if term isa KineticTerm
      T += matrix(term, basisset)
    elseif term isa PotentialTerm
      V += matrix(term, basisset)
    else
      throw(ArgumentError("unsupported Hamiltonian term: $(typeof(term))"))
    end
  end
  H = T + V

  # energy interval
  Eₘᵢₙ, Eₘₐₓ = method.Eₘᵢₙ, method.Eₘₐₓ
  LinearAlgebra.isposdef(LinearAlgebra.Symmetric(T - Eₘₐₓ * S)) ||
    throw(ArgumentError("Eₘₐₓ must be below the lowest free eigenvalue of (T, S)"))
  lower = _faddeev_yakubovsky_kernel(T, V, S, Eₘᵢₙ)
  upper = _faddeev_yakubovsky_kernel(T, V, S, Eₘₐₓ)
  lower.λ <= 1 + method.tolerance && upper.λ >= 1 - method.tolerance || throw(ArgumentError(
    "the interval must bracket the ground state: λ(Eₘᵢₙ) <= 1 <= λ(Eₘₐₓ); " *
    "got $(lower.λ) and $(upper.λ). Change the interval or check for a bound state.",
  ))

  # bisection: the largest kernel eigenvalue equals one at the ground state
  E, state = abs(lower.λ - 1) <= abs(upper.λ - 1) ? (Eₘᵢₙ, lower) : (Eₘₐₓ, upper)
  iterations = 0
  while abs(state.λ - 1) > method.tolerance &&
        Eₘₐₓ - Eₘᵢₙ > method.tolerance * max(1, abs(E))
    iterations < method.maxiters || error("FaddeevYakubovsky did not converge within $(method.maxiters) iterations")
    E = Eₘᵢₙ + (Eₘₐₓ - Eₘᵢₙ) / 2
    state = _faddeev_yakubovsky_kernel(T, V, S, E)
    if state.λ < 1
      Eₘᵢₙ = E
    else
      Eₘₐₓ = E
    end
    iterations += 1
  end

  # normalization
  c = state.c / sqrt(real(state.c' * S * state.c))
  C = reshape(c, :, 1)

  # expectation value
  expectation = Dict()
  if info > 0
    expectation[:S] = [c' * S * c]
    expectation[:H] = [c' * H * c]
    expectation[:0] = [expectation[:H][1] - E]
    expectation[:perturbation] = [sum((c' * matrix(term, basisset) * c for term in perturbation.terms); init=0.0)]
    for term in [hamiltonian.terms..., perturbation.terms...]
      expectation[term] = [c' * matrix(term, basisset) * c]
    end
    Printf.@printf("Faddeev-Yakubovsky (two-body): E₁ = %.12f, λ = %.12f\n", E, state.λ)
  end

  # return
  if info >= 0
    return ResultFaddeevYakubovsky(;
      info, hamiltonian, perturbation, method, basisset, nₘₐₓ=1,
      H, T, V, S, E=[E], C, λ=state.λ, iterations, expectation,
    )
  else
    return ResultFaddeevYakubovsky(; info, E=[E])
  end
end

# docstring

@doc raw"""
    FaddeevYakubovsky(basisset; Eₘᵢₙ=-1.0, Eₘₐₓ=0.0, tolerance=1e-10, maxiters=100)

Two-body, single-component proof of concept for the Faddeev-Yakubovsky method.
Accepts a `BasisSet`, `GeometricBasisSet`, or a single `Basis`, using the same
matrix elements as the Rayleigh-Ritz solver.

The finite interval `[Eₘᵢₙ, Eₘₐₓ]` must bracket the lowest state in the supplied
basis sector and lie below the lowest eigenvalue of the free problem `(T, S)`.
All `KineticTerm`s (including rest energies) form `T`; all `PotentialTerm`s form
the single pair interaction `V`. The overlap matrix must be positive definite.
Bisection stops when `abs(λ - 1) <= tolerance` or the interval width is at most
`tolerance * max(1, abs(E))`. An invalid bracket or exhausted `maxiters` raises
an error. Scattering and coupled three- or four-body equations are not implemented.
""" FaddeevYakubovsky

@doc raw"""
    solve(hamiltonian::Hamiltonian, method::FaddeevYakubovsky; perturbation=Hamiltonian(), info=4)

Find the lowest state from ``c=(ES-T)^{-1}Vc`` by searching for a kernel
eigenvalue of one. Only the largest algebraic kernel eigenvalue is followed.

The result contains one energy `E[1]`, normalized coefficients `C[:,1]`
(`c' * S * c = 1`), the kernel eigenvalue `λ`, and the iteration count.
Evaluate the wavefunction with `TwoBody.ψ(result, r)`. For `info > 0`,
expectation values, including first-order `perturbation`, are calculated;
the perturbation does not change `E`. `info=0` skips these diagnostics and
printing; `info < 0` returns only `info` and `E`.
""" solve(hamiltonian::Hamiltonian, method::FaddeevYakubovsky)
