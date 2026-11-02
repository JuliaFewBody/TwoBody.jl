export RayleighRitzMethod, ResultRayleighRitz, ResultOptimizationRayleighRitz,
       solve, optimize, expectation, verification

import LinearAlgebra
import Optim
import Printf
import QuadGK
import SpecialFunctions
import Subscripts

# method

struct RayleighRitzMethod{B<:Union{BasisSet,GeometricBasisSet}}
  basisset::B
end

RayleighRitzMethod(basis::Basis) = RayleighRitzMethod(BasisSet(basis))

_basisset(basisset::BasisSet) = basisset
_basisset(basisset::GeometricBasisSet) = BasisSet(basisset.basis...)

_string_basisset(basisset::BasisSet) = string(basisset)
_string_basisset(basisset::GeometricBasisSet) =
  "GeometricBasisSet($(basisset.basistype), $(basisset.r₁), $(basisset.rₙ), $(basisset.n); nₘᵢₙ=$(basisset.nₘᵢₙ), nₘₐₓ=$(basisset.nₘₐₓ))"

Base.string(method::RayleighRitzMethod) = "RayleighRitzMethod(" * _string_basisset(method.basisset) * ")"
Base.show(io::IO, method::RayleighRitzMethod) = print(io, Base.string(method))

# result

struct ResultRayleighRitz
  data::Any
  ResultRayleighRitz(; args...) = new(NamedTuple(Dict(args)))
end

struct ResultOptimizationRayleighRitz
  data::Any
  ResultOptimizationRayleighRitz(; args...) = new(NamedTuple(Dict(args)))
end

function ψ(result::ResultRayleighRitz, r; n::Int=1)
  return sum(result.C[i,n] * TwoBody.φ(result.basisset[i], r) for i in 1:result.nₘₐₓ)
end

ψ(result::ResultOptimizationRayleighRitz, r; n::Int=1) = ψ(result.result, r; n=n)

Base.getproperty(result::ResultRayleighRitz, symbol::Symbol) = Base.getproperty(getfield(result,:data), symbol)
Base.haskey(result::ResultRayleighRitz, symbol::Symbol) = Base.haskey(getfield(result,:data), symbol)
Base.show(io::IO, result::ResultRayleighRitz) = print(io, Base.string(result))

function Base.getproperty(result::ResultOptimizationRayleighRitz, symbol::Symbol)
  data = getfield(result, :data)
  return Base.haskey(data, symbol) ? Base.getproperty(data, symbol) : Base.getproperty(data.result, symbol)
end
Base.haskey(result::ResultOptimizationRayleighRitz, symbol::Symbol) =
  Base.haskey(getfield(result, :data), symbol) || Base.haskey(getfield(result, :data).result, symbol)
Base.show(io::IO, result::ResultOptimizationRayleighRitz) = print(io, Base.string(result))

# expectation value and verification

_expectation(C::AbstractMatrix, M::AbstractMatrix, states) = [C[:,n]' * M * C[:,n] for n in states]

function _eigenvectors(result::ResultRayleighRitz)
  haskey(result, :C) || throw(ArgumentError(
    "the result does not contain the eigenvectors; call `solve` with `info > 0` " *
    "to keep the matrices and eigenvectors needed for expectation values"
  ))
  return result.C
end

function expectation(result::ResultRayleighRitz, M::AbstractMatrix)
  C = _eigenvectors(result)
  size(M) == (size(C, 1), size(C, 1)) || throw(DimensionMismatch(
    "the matrix size $(size(M)) does not match the number of basis functions $(size(C, 1))"
  ))
  return _expectation(C, M, axes(C, 2))
end

function expectation(result::ResultRayleighRitz, operator::Operator)
  _eigenvectors(result)
  return expectation(result, matrix(operator, result.basisset))
end

function expectation(result::ResultRayleighRitz, hamiltonian::Hamiltonian)
  _eigenvectors(result)
  return expectation(result, matrix(hamiltonian, result.basisset))
end

expectation(result::ResultOptimizationRayleighRitz, operator) = expectation(result.result, operator)

function verification(result::ResultRayleighRitz)
  _eigenvectors(result)
  norm = expectation(result, result.S)
  residual = abs.(expectation(result, result.H) .- result.E)
  return (norm=norm, residual=residual)
end

verification(result::ResultOptimizationRayleighRitz) = verification(result.result)

# string

function Base.string(result::ResultRayleighRitz)
  if haskey(result, :C) && haskey(result, :basisset)
    # results extended by other solvers may carry the optimizer data themselves
    return _string_rayleigh_ritz(result, haskey(result, :optimizer) ? result : nothing)
  else
    return _string_compact("ResultRayleighRitz", result)
  end
end

function Base.string(result::ResultOptimizationRayleighRitz)
  solver = result.result
  if haskey(solver, :C) && haskey(solver, :basisset)
    return _string_rayleigh_ritz(solver, result)
  else
    return _string_compact("ResultOptimizationRayleighRitz", solver)
  end
end

function _string_compact(name::AbstractString, result::ResultRayleighRitz)
  # info ≤ 0
  text = "$(name):\n"
  text *= "  E: $(typeof(result.E))\n"
  for e in result.E
    text *= "    $(e)\n"
  end
  return text
end

function _string_geometric(geometricbasisset)
  text = "type \t$(geometricbasisset.basistype)\n"
  text *= string("range\tr", Subscripts.sub("$(geometricbasisset.nₘᵢₙ)"), " - r", Subscripts.sub("$(geometricbasisset.nₘₐₓ)"), "\n")
  text *= string("r", Subscripts.sub("$(geometricbasisset.nₘᵢₙ)"), " \t", geometricbasisset.r₁, "\n")
  text *= string("r", Subscripts.sub("$(geometricbasisset.n)"  ), " \t", geometricbasisset.rₙ, "\n")
  return text
end

function _string_optimizer(optimization)
  # optimizer
  text = "\n# optimizer\n\n"
  text *= string(optimization.optimizer) * "\n"
  text *= "P. K. Mogensen, A. N. Riseth, J. Open Source Softw., 3(24), 615 (2018)\n"
  text *= "https://doi.org/10.21105/joss.00615\n"
  # optimization log
  if haskey(optimization, :progress) && optimization.progress && haskey(optimization, :history)
    text *= "\n# optimization log\n\n"
    for (E, x) in optimization.history
      if isinf(E)
        text *= Printf.@sprintf("%+.9e              %s\n", E, "[" * join([Printf.@sprintf("%+.3e", x[i]) for i in keys(x)], ", ") *"]")
      else
        text *= Printf.@sprintf("%+.9e  %s\n", E, "[" * join([Printf.@sprintf("%+.3e", x[i]) for i in keys(x)], ", ") *"]")
      end
    end
  end
  # initial basis set
  if haskey(optimization, :initialgeometricbasisset)
    # initial geometric progression
    text *= "\n# initial geometric progression\n\n"
    text *= _string_geometric(optimization.initialgeometricbasisset)
  elseif haskey(optimization, :initialbasisset)
    # initial basis function
    text *= "\n# initial basis function\n\n"
    for n in 1:length(optimization.initialbasisset)
      text *= Printf.@sprintf("φ%s(r) = TwoBody.φ(%s, r)\n", Subscripts.sub("$n"), optimization.initialbasisset[n])
    end
  end
  return text
end

function _string_rayleigh_ritz(result::ResultRayleighRitz, optimization)
  optimized = !isnothing(optimization)
  # method
  text = "# method\n\n"
  text *= "Rayleigh-Ritz method with $(typeof(result.basisset[1]))\n"
  text *= "J. Thijssen, Computational Physics 2nd Edition (2013)\n"
  text *= "https://doi.org/10.1017/CBO9781139171397\n"
  # optimization
  if optimized
    text *= _string_optimizer(optimization)
  end
  # geometric progression
  if haskey(result, :geometricbasisset) && 0 < result.info
    if optimized
      text *= "\n# optimized geometric progression\n\n"
    else
      text *= "\n# geometric progression\n\n"
    end
    text *= _string_geometric(result.geometricbasisset)
  end
  # basis set
  if optimized && !haskey(result, :geometricbasisset)
    text *= "\n# optimized basis function\n\n"
  else
    text *= "\n# basis function\n\n"
  end
  for n in 1:result.nₘₐₓ
    text *= Printf.@sprintf("φ%s(r) = TwoBody.φ(%s, r)\n", Subscripts.sub("$n"), result.basisset[n])
  end
  # eigenfunction
  text *= "\n# eigenfunction\n\n"
  for n in 1:min(result.nₘₐₓ, result.info)
    text *= Printf.@sprintf("ψ%s(r) = ", Subscripts.sub("$n"))
    if result.nₘₐₓ > 5
      text *= "( "
    end
    ncol = 5
    if mod(result.nₘₐₓ,5) == 0
      ncol = 5
    elseif mod(result.nₘₐₓ,4) == 0
      ncol = 4
    elseif mod(result.nₘₐₓ,3) == 0
      ncol = 3
    end
    for i in 1:result.nₘₐₓ
      text *= Printf.@sprintf("%s %.6fφ%s(r) ", result.C[i,n]<0 ? "-" : "+", abs(result.C[i,n]), Subscripts.sub("$i"))
      if result.nₘₐₓ > 5
        if result.nₘₐₓ == i
          text *= ")"
        elseif mod(i,ncol) == 0
          text *= "\n           "
        end
      end
    end
    text *= "\n"
  end
  # eigenvalue
  text *= "\n# eigenvalue\n\n"
  for n in 1:min(result.nₘₐₓ, result.info)
    text *= Printf.@sprintf("E%s = %s\n", Subscripts.sub("$n"), "$(result.E[n])")
  end
  if 0 < result.info
    # verification
    text *= "\n# verification\n"
    # norm
    text *= "\nn \tnorm, <ψₙ|ψₙ> = cₙ' * S * cₙ = 1\n"
    for n in 1:min(result.nₘₐₓ, result.info)
      text *= string("$n\t", result.expectation[:S][n], "\n")
    end
    # ill-conditioned
    text *= "\nn \till-conditioned, |<ψₙ|H|ψₙ> - E| = |cₙ' * H * cₙ - E| = 0\n"
    for n in 1:min(result.nₘₐₓ, result.info)
      text *= string("$n\t", abs(result.expectation[:0][n]), "\n")
    end
    # expectation value
    text *= "\n# expectation value\n"
    # hamiltonian
    text *= "\nn \thamiltonian, <ψₙ|H|ψₙ> = cₙ' * H * cₙ\n"
    for n in 1:min(result.nₘₐₓ, result.info)
      text *= string("$n\t", result.expectation[:H][n], "\n")
    end
    # perturbation
    if !isempty(result.perturbation.terms)
      text *= "\nn \tperturbation\n"
      for n in 1:min(result.nₘₐₓ, result.info)
        text *= string("$n\t", result.expectation[:perturbation][n], "\n")
      end
      text *= "\nn \thamiltonian + perturbation\n"
      for n in 1:min(result.nₘₐₓ, result.info)
        text *= string("$n\t", result.expectation[:H][n] + result.expectation[:perturbation][n], "\n")
      end
    end
    # each term
    for term in [result.hamiltonian.terms..., result.perturbation.terms...]
      text *= "\nn \texpectation value of $(term)\n"
      for n in 1:min(result.nₘₐₓ, result.info)
        text *= string("$n\t", result.expectation[term][n], "\n")
      end
    end
  end
  # end
  return text
end

# solver

function _solve_rayleigh_ritz(hamiltonian::Hamiltonian, basisset::BasisSet; perturbation=Hamiltonian(), info=4)

  # initialization
  nₘₐₓ = length(basisset.basis)

  # matrix element
  S = matrix(basisset)
  H = matrix(hamiltonian, basisset)

  # calculation
  E, C = LinearAlgebra.eigen(H, S)

  # return
  if 0 < info
    # expectation value
    states = 1:min(nₘₐₓ, info)
    expectations = Dict()
    expectations[:S] = _expectation(C, S, states)
    expectations[:H] = _expectation(C, H, states)
    expectations[:0] = [expectations[:H][n] - E[n] for n in states]
    if !isempty(perturbation.terms)
      expectations[:perturbation] = _expectation(C, matrix(perturbation, basisset), states)
    end
    for term in [hamiltonian.terms..., perturbation.terms...]
      expectations[term] = _expectation(C, matrix(term, basisset), states)
    end
    # full information
    return (
      info = info,
      hamiltonian = hamiltonian,
      perturbation = perturbation,
      basisset = basisset,
      nₘₐₓ = nₘₐₓ,
      H = H,
      S = S,
      E = E,
      C = C,
      expectation = expectations,
    )
  else
    # for optimization
    return (
      info = info,
      E = E,
    )
  end

end

function solve(hamiltonian::Hamiltonian, method::RayleighRitzMethod; perturbation=Hamiltonian(), info=4)
  data = _solve_rayleigh_ritz(hamiltonian, _basisset(method.basisset); perturbation=perturbation, info=info)
  if info < 0
    # for optimization
    return ResultRayleighRitz(; data...)
  elseif method.basisset isa GeometricBasisSet
    return ResultRayleighRitz(; data..., geometricbasisset=method.basisset, method=method)
  else
    return ResultRayleighRitz(; data..., method=method)
  end
end

function solve(hamiltonian::Hamiltonian, basisset::BasisSet; perturbation=Hamiltonian(), info=4)
  return solve(hamiltonian, RayleighRitzMethod(basisset); perturbation=perturbation, info=info)
end

function solve(hamiltonian::Hamiltonian, basis::Basis; perturbation=Hamiltonian(), info=4)
  return solve(hamiltonian, RayleighRitzMethod(basis); perturbation=perturbation, info=info)
end

function solve(hamiltonian::Hamiltonian, basisset::GeometricBasisSet; perturbation=Hamiltonian(), info=4)
  return solve(hamiltonian, RayleighRitzMethod(basisset); perturbation=perturbation, info=info)
end

# optimizer

function optimize(hamiltonian::Hamiltonian, basisset::BasisSet; perturbation=Hamiltonian(), info=4, progress=true, optimizer=Optim.NelderMead(), options...)

  # optimize
  history = []
  res = Optim.optimize(
    x -> try
      E = solve(
        hamiltonian,
        BasisSet([_replace_exponent(basisset.basis[i], x[i]) for i in keys(basisset.basis)]...),
        perturbation = perturbation,
        info = -1
      ).E[1]
      if 0 ≤ info
        push!(history, (energy=E, parameters=copy(x)))
      end
      E
    catch
      if 0 ≤ info
        push!(history, (energy=Inf, parameters=copy(x)))
      end
      Inf
    end,
    [float(basisset.basis[i].a) for i in keys(basisset.basis)];
    method = optimizer,
    options...
  )

  # result
  result = solve(hamiltonian, BasisSet([_replace_exponent(basisset.basis[i], res.minimizer[i]) for i in keys(basisset.basis)]...), perturbation=perturbation, info=info)
  return ResultOptimizationRayleighRitz(;
    result = result,
    optimizer = optimizer,
    initialbasisset = basisset,
    options = options,
    progress = progress,
    history = history,
  )

end

function optimize(hamiltonian::Hamiltonian, basis::Basis; perturbation=Hamiltonian(), info=1, progress=true, optimizer=Optim.NelderMead(), options...)
  return optimize(hamiltonian, BasisSet(basis); perturbation=perturbation, info=info, progress=progress, optimizer=optimizer, options...)
end

function optimize(hamiltonian::Hamiltonian, basisset::GeometricBasisSet; perturbation=Hamiltonian(), info=4, progress=true, optimizer=Optim.NelderMead(), options...)

  # optimize
  history = []
  res = Optim.optimize(
    x -> try
      E = solve(
        hamiltonian,
        GeometricBasisSet(basisset.basistype, x..., basisset.n, nₘₐₓ=basisset.nₘₐₓ, nₘᵢₙ=basisset.nₘᵢₙ),
        perturbation = perturbation,
        info = -1
      ).E[1]
      if 0 ≤ info
        push!(history, (energy=E, parameters=copy(x)))
      end
      E
    catch
      if 0 ≤ info
        push!(history, (energy=Inf, parameters=copy(x)))
      end
      Inf
    end,
    [float(basisset.r₁), float(basisset.rₙ)];
    method = optimizer,
    options...
  )

  # result
  result = solve(hamiltonian, GeometricBasisSet(basisset.basistype, res.minimizer..., basisset.n, nₘₐₓ=basisset.nₘₐₓ, nₘᵢₙ=basisset.nₘᵢₙ); perturbation=perturbation, info=info)
  return ResultOptimizationRayleighRitz(;
    result = result,
    optimizer = optimizer,
    initialbasisset = BasisSet(basisset.basis...),
    initialgeometricbasisset = basisset,
    options = options,
    progress = progress,
    history = history,
  )

end

function optimize(hamiltonian::Hamiltonian, method::RayleighRitzMethod; kwargs...)
  return optimize(hamiltonian, method.basisset; kwargs...)
end

# matrix

function matrix(basisset::BasisSet)
  nₘₐₓ = length(basisset.basis)
  # S = [element(basisset.basis[i], basisset.basis[j]) for i=1:nₘₐₓ, j=1:nₘₐₓ]
  S = Array{Float64}(undef, nₘₐₓ, nₘₐₓ)
  for j in 1:nₘₐₓ
    for i in 1:j
      S[i,j] = element(basisset.basis[i], basisset.basis[j])
    end
  end
  return LinearAlgebra.Symmetric(S)
end

function matrix(operator::Operator, basisset::BasisSet)
  nₘₐₓ = length(basisset.basis)
  M = Array{Float64}(undef, nₘₐₓ, nₘₐₓ)
  for j in 1:nₘₐₓ
    for i in 1:j
      M[i,j] = element(operator, basisset.basis[i], basisset.basis[j])
    end
  end
  return LinearAlgebra.Symmetric(M)
end

function matrix(hamiltonian::Hamiltonian, basisset::BasisSet)
  nₘₐₓ = length(basisset.basis)
  # H = [element(hamiltonian, basisset.basis[i], basisset.basis[j]) for i=1:nₘₐₓ, j=1:nₘₐₓ]
  H = Array{Float64}(undef, nₘₐₓ, nₘₐₓ)
  for j in 1:nₘₐₓ
    for i in 1:j
      H[i,j] = element(hamiltonian, basisset.basis[i], basisset.basis[j])
    end
  end
  return LinearAlgebra.Symmetric(H)
end

# element

function element(SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)
  return (π/(SGB1.a+SGB2.a))^(3/2)
end

function element(PSB1::PowerSlaterBasis, PSB2::PowerSlaterBasis)
  return 4 * π * SpecialFunctions.gamma(PSB1.n+PSB2.n+3) / (PSB1.a+PSB2.a)^(PSB1.n+PSB2.n+3)
end

function element(o::RestEnergy, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)
  return o.m * o.c^2 * (π/(SGB1.a+SGB2.a))^(3/2)
end

function element(o::Laplacian, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)
  return - 6*π^(3/2)*SGB1.a*SGB2.a/(SGB1.a+SGB2.a)^(5/2)
end

function element(o::Kinetic, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)
  return o.hbar^2/(2*o.m) * 6*π^(3/2)*SGB1.a*SGB2.a/(SGB1.a+SGB2.a)^(5/2)
end

function element(o::Kinetic, PSB1::PowerSlaterBasis, PSB2::PowerSlaterBasis)
  return - o.hbar^2/(2*o.m) * 4 * π * (
        PSB2.n*(PSB2.n+1) * SpecialFunctions.gamma(PSB1.n+PSB2.n+1) / (PSB1.a+PSB2.a)^(PSB1.n+PSB2.n+1)
    - 2*PSB2.a*(PSB2.n+1) * SpecialFunctions.gamma(PSB1.n+PSB2.n+2) / (PSB1.a+PSB2.a)^(PSB1.n+PSB2.n+2)
    +            PSB2.a^2 * SpecialFunctions.gamma(PSB1.n+PSB2.n+3) / (PSB1.a+PSB2.a)^(PSB1.n+PSB2.n+3)
    )
end

function element(o::Constant, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)
  return o.constant * (π/(SGB1.a+SGB2.a))^(3/2)
end

function element(o::Linear, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)
  return o.coefficient * 2*π/(SGB1.a+SGB2.a)^2
end

function element(o::Coulomb, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)
  return o.coefficient * 2*π/(SGB1.a+SGB2.a)
end

function element(o::Coulomb, PSB1::PowerSlaterBasis, PSB2::PowerSlaterBasis)
  return o.coefficient * 4 * π * SpecialFunctions.gamma(PSB1.n+PSB2.n+2) / (PSB1.a+PSB2.a)^(PSB1.n+PSB2.n+2)
end

function element(o::PowerLaw, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)
  return o.coefficient * 2 * π * SpecialFunctions.gamma((o.exponent+3)/2) / (SGB1.a+SGB2.a)^((o.exponent+3)/2)
end

function element(o::Gaussian, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)
  return o.coefficient * (π/(o.exponent+SGB1.a+SGB2.a))^(3/2)
end

function element(o::Custom, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)
  value, _ = QuadGK.quadgk(
    r -> 4π * r^2 * φ(SGB1, r) * V(o, r) * φ(SGB2, r),
    0.0,
    Inf,
    rtol=1e-10,
    atol=1e-12,
  )
  return value
end

@inline function element(B1::ContractedBasis, B2::Basis)
  return _contracted_bra_element(
    getfield(B1, :coefficients),
    getfield(B1, :primitives),
    B2,
  )
end

@inline function element(B1::PrimitiveBasis, B2::ContractedBasis)
  return _contracted_ket_element(
    B1,
    getfield(B2, :coefficients),
    getfield(B2, :primitives),
  )
end

@inline function element(o::Operator, B1::ContractedBasis, B2::Basis)
  return _contracted_bra_element(
    o,
    getfield(B1, :coefficients),
    getfield(B1, :primitives),
    B2,
  )
end

@inline function element(o::Operator, B1::PrimitiveBasis, B2::ContractedBasis)
  return _contracted_ket_element(
    o,
    B1,
    getfield(B2, :coefficients),
    getfield(B2, :primitives),
  )
end

@inline _contracted_bra_element(coefficients::Tuple{T}, primitives::Tuple{P}, B2::Basis) where {T,P} =
  conj(first(coefficients)) * element(first(primitives), B2)
@inline _contracted_bra_element(coefficients::Tuple, primitives::Tuple, B2::Basis) = muladd(
  conj(first(coefficients)),
  element(first(primitives), B2),
  _contracted_bra_element(Base.tail(coefficients), Base.tail(primitives), B2),
)

@inline _contracted_ket_element(B1::PrimitiveBasis, coefficients::Tuple{T}, primitives::Tuple{P}) where {T,P} =
  first(coefficients) * element(B1, first(primitives))
@inline _contracted_ket_element(B1::PrimitiveBasis, coefficients::Tuple, primitives::Tuple) = muladd(
  first(coefficients),
  element(B1, first(primitives)),
  _contracted_ket_element(B1, Base.tail(coefficients), Base.tail(primitives)),
)

@inline _contracted_bra_element(o::Operator, coefficients::Tuple{T}, primitives::Tuple{P}, B2::Basis) where {T,P} =
  conj(first(coefficients)) * element(o, first(primitives), B2)
@inline _contracted_bra_element(o::Operator, coefficients::Tuple, primitives::Tuple, B2::Basis) = muladd(
  conj(first(coefficients)),
  element(o, first(primitives), B2),
  _contracted_bra_element(o, Base.tail(coefficients), Base.tail(primitives), B2),
)

@inline _contracted_ket_element(o::Operator, B1::PrimitiveBasis, coefficients::Tuple{T}, primitives::Tuple{P}) where {T,P} =
  first(coefficients) * element(o, B1, first(primitives))
@inline _contracted_ket_element(o::Operator, B1::PrimitiveBasis, coefficients::Tuple, primitives::Tuple) = muladd(
  first(coefficients),
  element(o, B1, first(primitives)),
  _contracted_ket_element(o, B1, Base.tail(coefficients), Base.tail(primitives)),
)

function element(o::Hamiltonian, B1::Basis, B2::Basis)
  return sum(element(term, B1, B2) for term in o.terms)
end

# docstring

@doc raw"""
`RayleighRitzMethod(basisset)`

| Arguments | Description |
| :-------- | :---------- |
| `basisset::Union{BasisSet,GeometricBasisSet}` | Basis functions ``\phi_1, \phi_2, \ldots`` spanning the trial space. A single `Basis` is wrapped in a one-element `BasisSet`. |

Configuration of the Rayleigh-Ritz method in the same style as `FiniteDifferenceMethod`. Pass it to `solve` to solve the generalized eigenvalue problem ``\pmb{H} \pmb{c} = E \pmb{S} \pmb{c}`` in the given basis set, or to `optimize` to optimize the nonlinear parameters of the basis functions.

```julia
method = RayleighRitzMethod(BasisSet(SimpleGaussianBasis(1.0), SimpleGaussianBasis(0.1)))
result = solve(hamiltonian, method)
```
""" RayleighRitzMethod

@doc raw"""
`ResultRayleighRitz`

Result of `solve` with a `RayleighRitzMethod` (or a basis set). The data are accessed as properties, e.g. `result.E` and `result.C`. With `info > 0`, the result contains

| Property | Description |
| :------- | :---------- |
| `method`            | The `RayleighRitzMethod`. |
| `hamiltonian`       | The Hamiltonian. |
| `perturbation`      | The perturbation, of which only the expectation values are computed. |
| `basisset`          | The `BasisSet`. |
| `geometricbasisset` | The `GeometricBasisSet` (only for geometric progressions). |
| `nₘₐₓ`              | The number of basis functions. |
| `H`                 | The Hamiltonian matrix ``\pmb{H}``. |
| `S`                 | The overlap matrix ``\pmb{S}``. |
| `E`                 | The eigenvalues ``E_n``. |
| `C`                 | The eigenvectors ``\pmb{c}_n`` as columns. |
| `expectation`       | A `Dict` of the expectation values for the first `min(nₘₐₓ, info)` states. |
| `info`              | The `info` keyword passed to `solve`. |

With `info ≤ 0`, only `info` and `E` are kept (and `method` and `geometricbasisset` for `info = 0`). The expectation values and the verification for all states can be recomputed with `expectation` and `verification`. `haskey(result, key)` checks whether a property is available.
""" ResultRayleighRitz

@doc raw"""
`ResultOptimizationRayleighRitz`

Result of `optimize`. The final `ResultRayleighRitz` calculated with the optimized basis set is stored as `result.result`, and the optimization data are stored as

| Property | Description |
| :------- | :---------- |
| `optimizer`                | The optimizer of Optim.jl. |
| `options`                  | The other keyword arguments passed to `Optim.optimize`. |
| `progress`                 | Whether the optimization log is displayed. |
| `history`                  | The energies and parameters evaluated during the optimization (empty for `info < 0`). |
| `initialbasisset`          | The initial `BasisSet`. |
| `initialgeometricbasisset` | The initial `GeometricBasisSet` (only for geometric progressions). |

Any other property is forwarded to `result.result`, so that `result.E`, `result.C`, `result.basisset`, `result.nₘₐₓ`, `result.expectation`, etc. are available as for `ResultRayleighRitz`.
""" ResultOptimizationRayleighRitz

@doc raw"""
`solve(hamiltonian::Hamiltonian, method::RayleighRitzMethod; perturbation=Hamiltonian(), info=4)`

This function returns the eigenvalues ``E``  and eigenvectors ``\pmb{c}`` for
```math
\pmb{H} \pmb{c} = E \pmb{S} \pmb{c}.
```
The Hamiltonian matrix is defined as ``H_{ij} = \langle \phi_{i} | \hat{H} | \phi_{j} \rangle``. The overlap matrix is defined as ``S_{ij} = \langle \phi_{i} | \phi_{j} \rangle``. The result is returned as a `ResultRayleighRitz`.

| Arguments | Default | Description |
| :-------- | :------ | :---------- |
| `perturbation::Hamiltonian` | `Hamiltonian()` | Operators whose expectation values are calculated without being included in the Hamiltonian matrix. |
| `info::Int`                 | `4`             | `info > 0` keeps the matrices, the eigenvectors and the expectation values for the first `min(nₘₐₓ, info)` states, which are also displayed. `info ≤ 0` keeps only `info` and `E` (and `method`, `geometricbasisset` for `info = 0`), which is used for optimization. |
""" solve(hamiltonian::Hamiltonian, method::RayleighRitzMethod; perturbation=Hamiltonian(), info=4)

@doc raw"""
`solve(hamiltonian::Hamiltonian, basisset::BasisSet; perturbation=Hamiltonian(), info=4)`

This function returns `solve(hamiltonian, RayleighRitzMethod(basisset); perturbation=perturbation, info=info)`, i.e., the eigenvalues ``E``  and eigenvectors ``\pmb{c}`` for
```math
\pmb{H} \pmb{c} = E \pmb{S} \pmb{c}.
```
The Hamiltonian matrix is defined as ``H_{ij} = \langle \phi_{i} | \hat{H} | \phi_{j} \rangle``. The overlap matrix is defined as ``S_{ij} = \langle \phi_{i} | \phi_{j} \rangle``.
""" solve(hamiltonian::Hamiltonian, basisset::BasisSet; perturbation=Hamiltonian(), info=4)

@doc raw"""
`function optimize(hamiltonian::Hamiltonian, basisset::BasisSet; perturbation=Hamiltonian(), info=4, progress=true, optimizer=Optim.NelderMead(), options...)`

This function minimizes the energy by changing the exponents of the basis functions using Optim.jl.
```math
\frac{\partial E}{\partial a_i} = 0
```
The other keyword arguments `options...` are passed to `Optim.optimize`, e.g., `iterations=100`. The result is returned as a `ResultOptimizationRayleighRitz`.
""" optimize(hamiltonian::Hamiltonian, basisset::BasisSet; perturbation=Hamiltonian(), info=4, progress=true, optimizer=Optim.NelderMead(), options...)

@doc raw"""
`solve(hamiltonian::Hamiltonian, basis::Basis; perturbation=Hamiltonian(), info=4)`

This a solver for 1-basis calculations. This function returns `solve(hamiltonian, RayleighRitzMethod(basis); perturbation=perturbation, info=info)`.
""" solve(hamiltonian::Hamiltonian, basis::Basis; perturbation=Hamiltonian(), info=4)

@doc raw"""
`optimize(hamiltonian::Hamiltonian, basis::Basis; perturbation=Hamiltonian(), info=1, progress=true, optimizer=Optim.NelderMead(), options...)`

This a optimizer for 1-basis calculations. This function returns `optimize(hamiltonian, BasisSet(basis); perturbation=perturbation, info=info, progress=progress, optimizer=optimizer, options...)`.
""" optimize(hamiltonian::Hamiltonian, basis::Basis; perturbation=Hamiltonian(), info=1, progress=true, optimizer=Optim.NelderMead(), options...)

@doc raw"""
`solve(hamiltonian::Hamiltonian, basisset::GeometricBasisSet; perturbation=Hamiltonian(), info=4)`

This function returns `solve(hamiltonian, RayleighRitzMethod(basisset); perturbation=perturbation, info=info)`. The geometric progression is kept as `result.geometricbasisset`.
""" solve(hamiltonian::Hamiltonian, basisset::GeometricBasisSet; perturbation=Hamiltonian(), info=4)

@doc raw"""
`optimize(hamiltonian::Hamiltonian, basisset::GeometricBasisSet; perturbation=Hamiltonian(), info=4, progress=true, optimizer=Optim.NelderMead(), options...)`

This function minimizes the energy by optimizing $r_1$ and $r_n$ using Optim.jl.
```math
\frac{\partial E}{\partial r_1} = \frac{\partial E}{\partial r_n} = 0
```
The result is returned as a `ResultOptimizationRayleighRitz`.
""" optimize(hamiltonian::Hamiltonian, basisset::GeometricBasisSet; perturbation=Hamiltonian(), info=4, progress=true, optimizer=Optim.NelderMead(), options...)

@doc raw"""
`optimize(hamiltonian::Hamiltonian, method::RayleighRitzMethod; kwargs...)`

This function returns `optimize(hamiltonian, method.basisset; kwargs...)`, i.e., the exponents of the basis functions in a `BasisSet`, or $r_1$ and $r_n$ of a `GeometricBasisSet`, are optimized using Optim.jl.
""" optimize(hamiltonian::Hamiltonian, method::RayleighRitzMethod; kwargs...)

@doc raw"""
`expectation(result::ResultRayleighRitz, M::AbstractMatrix)`

`expectation(result::ResultRayleighRitz, operator::Operator)`

`expectation(result::ResultRayleighRitz, hamiltonian::Hamiltonian)`

This function returns the vector of the expectation values for all states ``n = 1, \ldots, n_\mathrm{max}``,
```math
\langle \psi_n | \hat{O} | \psi_n \rangle = \pmb{c}_n^\dagger \pmb{O} \pmb{c}_n,
```
where the matrix ``\pmb{O}`` is given directly or calculated with `matrix(operator, result.basisset)`. The result must contain the eigenvectors, i.e., `solve` must be called with `info > 0`; otherwise an `ArgumentError` is thrown. A `ResultOptimizationRayleighRitz` is also accepted.
""" expectation

@doc raw"""
`verification(result::ResultRayleighRitz)`

This function returns a named tuple `(norm=..., residual=...)` with the vectors
```math
\begin{aligned}
\mathrm{norm}_n &= \langle \psi_n | \psi_n \rangle = \pmb{c}_n^\dagger \pmb{S} \pmb{c}_n = 1, \\
\mathrm{residual}_n &= \left| \langle \psi_n | \hat{H} | \psi_n \rangle - E_n \right| = \left| \pmb{c}_n^\dagger \pmb{H} \pmb{c}_n - E_n \right| = 0,
\end{aligned}
```
for all states. Deviations from these values indicate an ill-conditioned basis set. The result must contain the matrices and eigenvectors, i.e., `solve` must be called with `info > 0`; otherwise an `ArgumentError` is thrown. A `ResultOptimizationRayleighRitz` is also accepted.
""" verification

@doc raw"""
`matrix(basisset::BasisSet)`

This function returns the overlap matrix $\pmb{S}$. The element is written as ``S_{ij} = \langle \phi_{i} | \phi_{j} \rangle``.
""" matrix(basisset::BasisSet)

@doc raw"""
`matrix(operator::Operator, basisset::BasisSet)`

!!! note
  This function is used for the expectation values and is not used in computing the Hamiltonian matrix.

This function returns the matrix corresponding to the operator in the given basis set. The element is written as ``O_{ij} = \langle \phi_{i} | \hat{o} | \phi_{j} \rangle``.
""" matrix(operator::Operator, basisset::BasisSet)

@doc raw"""
`matrix(hamiltonian::Hamiltonian, basisset::BasisSet)`

This function returns the Hamiltonian matrix $\pmb{H}$. The element is written as ``H_{ij} = \langle \phi_{i} | \hat{H} | \phi_{j} \rangle``.
""" matrix(hamiltonian::Hamiltonian, basisset::BasisSet)

@doc raw"""
`element(SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)`

```math
\begin{aligned}
  S_{ij}
   = \langle \phi_{i} | \phi_{j} \rangle
  &= \int
     \phi_{i}^*(r)
     \phi_{j}(r)
     \mathrm{d} \pmb{r} \\
  &= \iiint
     \mathrm{e}^{-\alpha_i r^2}
     \mathrm{e}^{-\alpha_j r^2}
     ~r^2 \sin\theta ~
     \mathrm{d} r
     \mathrm{d} \theta
     \mathrm{d} \varphi \\
  &= \int_0^{2\pi} \mathrm{d}\varphi
     \int_0^\pi \sin\theta ~\mathrm{d}\theta
     \int_0^\infty r^{2} \mathrm{e}^{-(\alpha_i + \alpha_j) r^2} ~\mathrm{d}r \\
  &= 2\pi \times 2 \times \frac{1!!}{2^{2}} \sqrt{\frac{\pi}{a^{3}}} \\
  &= \underline{\left( \frac{\pi}{\alpha_i + \alpha_j} \right)^{3/2}}
\end{aligned}
```

Integral Formula:
```math
\int_0^{\infty} r^{2n} \exp \left(-a r^2\right) ~\mathrm{d}r = \frac{(2n-1)!!}{2^{n+1}} \sqrt{\frac{\pi}{a^{2n+1}}}
```
""" element(SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)

@doc raw"""
`element(o::RestEnergy, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)`

```math
\begin{aligned}
  \langle \phi_{i} | mc^2 | \phi_{j} \rangle
  &= mc^2 \langle \phi_{i} | \phi_{j} \rangle \\
  &= mc^2 \iiint
     \phi_{i}^*(r)
     \phi_{j}(r)
     ~r^2 \sin\theta ~\mathrm{d}r \mathrm{d}\theta \mathrm{d}\varphi \\
  &= mc^2
     \int_0^{2\pi} \mathrm{d}\varphi
     \int_0^\pi \sin\theta ~\mathrm{d}\theta
     \int_0^\infty r^{2} \mathrm{e}^{-(\alpha_i + \alpha_j) r^2} ~\mathrm{d}r \\
  &= mc^2 \times 2\pi \times 2 \times \frac{1!!}{2^{2}} \sqrt{\frac{\pi}{a^{3}}} \\
  &= \underline{mc^2 \left( \frac{\pi}{\alpha_i + \alpha_j} \right)^{3/2}}
\end{aligned}
```

Integral Formula:
```math
\int_0^{\infty} r^{2n} \exp \left(-a r^2\right) ~\mathrm{d}r = \frac{(2n-1)!!}{2^{n+1}} \sqrt{\frac{\pi}{a^{2n+1}}}
```
""" element(o::RestEnergy, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)

@doc raw"""
`element(o::Laplacian, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)`

```math
\begin{aligned}
  \langle \phi_{i} | \nabla^2 | \phi_{j} \rangle
  = \underline{
      -6 \frac{\alpha_i \alpha_j \pi^{\frac{3}{2}}}{(\alpha_i + \alpha_j)^{\frac{5}{2}}}
    }
\end{aligned}
```
""" element(o::Laplacian, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)

@doc raw"""
`element(o::Kinetic, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)`

## Derivation (without Green's identity)

```math
\begin{aligned}
  T_{ij} = \langle \phi_{i} | \hat{T} | \phi_{j} \rangle
  &= \iiint
     \mathrm{e}^{-\alpha_i r^2}
     \left[ -\frac{\hbar^2}{2\mu} \nabla^2 \right]
     \mathrm{e}^{-\alpha_j r^2}
     ~r^2 \sin\theta ~\mathrm{d}r \mathrm{d}\theta \mathrm{d}\varphi \\
  &= -\frac{\hbar^2}{2\mu} \iiint
     \mathrm{e}^{-\alpha_i r^2}
     \left[ \nabla^2 \right]
     \mathrm{e}^{-\alpha_j r^2}
     ~r^2 \sin\theta ~\mathrm{d}r \mathrm{d}\theta \mathrm{d}\varphi \\
  &= -\frac{\hbar^2}{2\mu} \iiint
     \mathrm{e}^{-\alpha_i r^2}
     \left[ -6\alpha_j + 4\alpha_j^2 r^2 \right]
     \mathrm{e}^{-\alpha_j r^2}
     ~r^2 \sin\theta ~\mathrm{d}r \mathrm{d}\theta \mathrm{d}\varphi \\
  &= -\frac{\hbar^2}{2\mu} \iint
     \sin\theta ~\mathrm{d}\theta \mathrm{d}\varphi
     \int
     \left[ -6\alpha_j + 4\alpha_j^2 r^2 \right]
     r^2 \mathrm{e}^{-(\alpha_i + \alpha_j) r^2}
     ~\mathrm{d}r \\
  &= -\frac{\hbar^2}{2\mu} \cdot 4\pi
     \left[
        -6\alpha_j   \mathrm{GGI}(2, \alpha_i + \alpha_j)
        +4\alpha_j^2 \mathrm{GGI}(4, \alpha_i + \alpha_j)
     \right]
     \\
  &= -\frac{\hbar^2}{2\mu} \cdot 4\pi
     \left[
        -6\alpha_j   \frac{\Gamma\left( \frac{3}{2} \right)}{2 (\alpha_i + \alpha_j)^{\frac{3}{2}}}
        +4\alpha_j^2 \frac{\Gamma\left( \frac{5}{2} \right)}{2 (\alpha_i + \alpha_j)^{\frac{5}{2}}}
     \right] \\
  &= -\frac{\hbar^2}{2\mu} \cdot 4\pi
     \left[
        -6\alpha_j   \frac{ \sqrt{\pi}/2}{2 (\alpha_i + \alpha_j)^{\frac{3}{2}}}
        +4\alpha_j^2 \frac{3\sqrt{\pi}/4}{2 (\alpha_i + \alpha_j)^{\frac{5}{2}}}
     \right] \\
  &= -\frac{\hbar^2}{2\mu} \cdot 4\pi
     \left[
        \frac{\alpha_j}{\alpha_i + \alpha_j} - 1
     \right]
     \cdot 6 \alpha_j \cdot \frac{\sqrt{\pi}/2}{2 (\alpha_i + \alpha_j)^{\frac{3}{2}}}
     \\
  &= -\frac{\hbar^2}{2\mu} \cdot 4\pi
     \left[
        - \frac{\alpha_i}{\alpha_i + \alpha_j}
     \right]
     \cdot 6 \alpha_j \cdot \frac{\sqrt{\pi}/2}{2 (\alpha_i + \alpha_j)^{\frac{3}{2}}}
     \\
  &= \underline{
        \frac{\hbar^2}{2\mu}
        \cdot 6
        \cdot \frac{\alpha_i \alpha_j \pi^{\frac{3}{2}}}{(\alpha_i + \alpha_j)^{\frac{5}{2}}}
     }
\end{aligned}
```

## Derivation (with Green's identity)

```math
\begin{aligned}
  T_{ij} = \langle \phi_{i} | \hat{T} | \phi_{j} \rangle
  &= \iiint
     \mathrm{e}^{-\alpha_i r^2}
     \left[ -\frac{\hbar^2}{2\mu} \nabla^2 \right]
     \mathrm{e}^{-\alpha_j r^2}
     ~r^2 \sin\theta ~\mathrm{d}r \mathrm{d}\theta \mathrm{d}\varphi \\
  &= -\frac{\hbar^2}{2\mu} \iiint
     \mathrm{e}^{-\alpha_i r^2}
     \nabla^2
     \mathrm{e}^{-\alpha_j r^2}
     ~r^2 \sin\theta ~\mathrm{d}r \mathrm{d}\theta \mathrm{d}\varphi \\
  &= \frac{\hbar^2}{2\mu} \iiint
     \left[ \nabla \mathrm{e}^{-\alpha_i r^2} \right]
     \left[ \nabla \mathrm{e}^{-\alpha_j r^2} \right]
     ~r^2 \sin\theta ~\mathrm{d}r \mathrm{d}\theta \mathrm{d}\varphi \\
  &= \frac{\hbar^2}{2\mu} \iiint
     \left[ -2 \alpha_i r \mathrm{e}^{-\alpha_i r^2} \right]
     \left[ -2 \alpha_j r \mathrm{e}^{-\alpha_j r^2} \right]
     ~r^2 \sin\theta ~\mathrm{d}r \mathrm{d}\theta \mathrm{d}\varphi \\
  &= \frac{\hbar^2}{2\mu} \cdot 4 \alpha_i \alpha_j \iiint
     \left[ r \mathrm{e}^{-\alpha_i r^2} \right]
     \left[ r \mathrm{e}^{-\alpha_j r^2} \right]
     ~r^2 \sin\theta ~\mathrm{d}r \mathrm{d}\theta \mathrm{d}\varphi \\
  &= \frac{\hbar^2}{2\mu}
     \cdot 4 \alpha_i \alpha_j
     \iint \sin\theta ~\mathrm{d}\theta \mathrm{d}\varphi
     \int r^4
     \mathrm{e}^{- (\alpha_i + \alpha_j) r^2}
     ~\mathrm{d}r \\
  &= \frac{\hbar^2}{2\mu}
     \cdot 4 \alpha_i \alpha_j
     \cdot 4 \pi
     \cdot \mathrm{GGI}(4, \alpha_i + \alpha_j) \\
  &= \frac{\hbar^2}{2\mu}
     \cdot 4 \alpha_i \alpha_j
     \cdot 4 \pi
     \cdot \frac{\Gamma\left( \frac{5}{2} \right)}{2 (\alpha_i + \alpha_j)^{\frac{5}{2}}} \\
  &= \frac{\hbar^2}{2\mu}
     \cdot 4 \alpha_i \alpha_j
     \cdot 4 \pi
     \cdot \frac{3\sqrt{\pi}/4}{2 (\alpha_i + \alpha_j)^{\frac{5}{2}}} \\
  &= \underline{
        \frac{\hbar^2}{2\mu}
        \cdot 6
        \cdot \frac{\alpha_i \alpha_j \pi^{\frac{3}{2}}}{(\alpha_i + \alpha_j)^{\frac{5}{2}}}
     }
\end{aligned}
```

## Formula

[Green's first identity](https://en.wikipedia.org/wiki/Green%27s_identities#Green's_first_identity):
```math
\begin{aligned}
  \iiint_V
  f \pmb{\nabla}^2 g
  ~ \mathrm{d}V
+ \iiint_V
  \pmb{\nabla} f \cdot
  \pmb{\nabla} g
  ~ \mathrm{d}V
= \iint_{\partial V}
  f \pmb{\nabla} g \cdot \pmb{n}
  ~ \mathrm{d}S
\end{aligned}
```

[generalized Gaussian integral](https://en.wikipedia.org/wiki/Gaussian_integral#Integrals_of_similar_form):
```math
\begin{aligned}
  \mathrm{GGI}(n,b)
  = \int_0^{\infty} x^{n} \exp \left(-b x^2\right) \mathrm{d}x
  = \frac{\Gamma\left( \frac{n+1}{2} \right)}{2 b^{\frac{n+1}{2}}}
\end{aligned}
```
""" element(o::Kinetic, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)

@doc raw"""
`element(o::Constant, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)`

```math
\begin{aligned}
  \langle \phi_{i} | c | \phi_{j} \rangle
  &= c \langle \phi_{i} | \phi_{j} \rangle \\
  &= c \iiint
     \phi_{i}^*(r)
     \phi_{j}(r)
     ~r^2 \sin\theta ~\mathrm{d}r \mathrm{d}\theta \mathrm{d}\varphi \\
  &= c
     \int_0^{2\pi} \mathrm{d}\varphi
     \int_0^\pi \sin\theta ~\mathrm{d}\theta
     \int_0^\infty r^{2} \mathrm{e}^{-(\alpha_i + \alpha_j) r^2} ~\mathrm{d}r \\
  &= c \times 2\pi \times 2 \times \frac{1!!}{2^{2}} \sqrt{\frac{\pi}{a^{3}}} \\
  &= \underline{c \left( \frac{\pi}{\alpha_i + \alpha_j} \right)^{3/2}}
\end{aligned}
```

Integral Formula:
```math
\int_0^{\infty} r^{2n} \exp \left(-a r^2\right) ~\mathrm{d}r = \frac{(2n-1)!!}{2^{n+1}} \sqrt{\frac{\pi}{a^{2n+1}}}
```
""" element(o::Constant, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)

@doc raw"""
`element(o::Linear, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)`

```math
\begin{aligned}
  \langle \phi_{i} | r | \phi_{j} \rangle
  &= \iiint
     \phi_{i}^*(r)
     \times r \times
     \phi_{j}(r)
     ~r^2 \sin\theta ~\mathrm{d}r \mathrm{d}\theta \mathrm{d}\varphi \\
  &= \int_0^{2\pi} \mathrm{d}\varphi
     \int_0^\pi \sin\theta ~\mathrm{d}\theta
     \int_0^\infty r^3 \mathrm{e}^{-(\alpha_i + \alpha_j) r^2} ~\mathrm{d}r \\
  &= 2\pi \times 2 \times \frac{1!}{2 (\alpha_i + \alpha_j)^{2}} \\
  &= \underline{\frac{2\pi}{(\alpha_i + \alpha_j)^2}}
\end{aligned}
```

Integral Formula:
```math
\int_0^{\infty} r^{2n+1} \exp \left(-a r^2\right) ~\mathrm{d}r = \frac{n!}{2 a^{n+1}}
```
""" element(o::Linear, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)

@doc raw"""
`element(o::Coulomb, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)`

```math
\begin{aligned}
  \langle \phi_{i} | \frac{1}{r} | \phi_{j} \rangle
  &= \iiint
    \phi_{i}^*(r)
    \times \frac{1}{r} \times
    \phi_{j}(r)
    ~r^2 \sin\theta ~\mathrm{d}r \mathrm{d}\theta \mathrm{d}\varphi \\
  &= \int_0^{2\pi} \mathrm{d}\varphi
    \int_0^\pi \sin\theta ~\mathrm{d}\theta
    \int_0^\infty r \mathrm{e}^{-(\alpha_i + \alpha_j) r^2} ~\mathrm{d}r \\
  &= 2\pi \times 2 \times \frac{0!}{2 (\alpha_i + \alpha_j)} \\
  &= \underline{\frac{2\pi}{\alpha_i + \alpha_j}}
\end{aligned}
```

Integral Formula:
```math
\int_0^{\infty} r^{2n+1} \exp \left(-a r^2\right) ~\mathrm{d}r = \frac{n!}{2 a^{n+1}}
```
""" element(o::Coulomb, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)

@doc raw"""
`element(o::PowerLaw, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)`

```math
\begin{aligned}
  \langle \phi_{i} | r^n | \phi_{j} \rangle
  &= \iiint
     \phi_{i}^*(r)
     \times r^n \times
     \phi_{j}(r)
     ~r^2 \sin\theta ~\mathrm{d}r \mathrm{d}\theta \mathrm{d}\varphi \\
  &= \int_0^{2\pi} \mathrm{d}\varphi
     \int_0^\pi \sin\theta ~\mathrm{d}\theta
     \int_0^\infty r^{n+2} \mathrm{e}^{-(\alpha_i + \alpha_j) r^2} ~\mathrm{d}r \\
  &= 2\pi \times 2 \times \frac{\Gamma\left( \frac{n+3}{2} \right)}{2 (\alpha_i + \alpha_j)^{\frac{n+3}{2}}} \\
  &= \underline{2\pi\frac{\Gamma\left( \frac{n+3}{2} \right)}{(\alpha_i + \alpha_j)^{\frac{n+3}{2}}}}
\end{aligned}
```

Integral Formula:
```math
\int_0^{\infty} r^{n} \exp \left(-a r^2\right) ~\mathrm{d}r = \frac{\Gamma\left( \frac{n+1}{2} \right)}{2 a^{\frac{n+1}{2}}}
```
""" element(o::PowerLaw, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)

@doc raw"""
`element(o::Gaussian, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)`

```math
\begin{aligned}
  \langle \phi_{i} | \exp(-br^2) | \phi_{j} \rangle
  &= \iiint
     \phi_{i}^*(r)
     \times \exp(-br^2) \times
     \phi_{j}(r)
     ~r^2 \sin\theta ~\mathrm{d}r \mathrm{d}\theta \mathrm{d}\varphi \\
  &= \int_0^{2\pi} \mathrm{d}\varphi
     \int_0^\pi \sin\theta ~\mathrm{d}\theta
     \int_0^\infty r^2 \mathrm{e}^{-(b+\alpha_i + \alpha_j) r^2} ~\mathrm{d}r \\
  &= 2\pi \times 2 \times \frac{1!!}{2^{2}} \sqrt{\frac{\pi}{(b + \alpha_i + \alpha_j)^{2\cdot1+1}}} \\
  &= \underline{\left( \frac{\pi}{b + \alpha_i + \alpha_j} \right)^{3/2}}
\end{aligned}
```

Integral Formula:
```math
\int_0^{\infty} r^{2n} \exp \left(-a r^2\right) ~\mathrm{d}r = \frac{(2n-1)!!}{2^{n+1}} \sqrt{\frac{\pi}{a^{2n+1}}}
```
""" element(o::Gaussian, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)

@doc raw"""
`element(o::Custom, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)`

The matrix element for a custom central potential is evaluated numerically with
adaptive Gauss--Kronrod quadrature:
```math
\langle \phi_i | V | \phi_j \rangle
= 4\pi \int_0^\infty
r^2 \phi_i(r) V(r) \phi_j(r)\,\mathrm{d}r.
```
""" element(o::Custom, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)

@doc raw"""
`element(o::Hamiltonian, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)`

```math
\begin{aligned}
  H_{ij}
  &= \langle \phi_{i} | \hat{H} | \phi_{j} \rangle \\
  &= \langle \phi_{i} | \sum_k \hat{o}_k | \phi_{j} \rangle \\
  &= \sum_k \langle \phi_{i} | \hat{o}_k | \phi_{j} \rangle \\
\end{aligned}
```
""" element(o::Hamiltonian, B1::Basis, B2::Basis)
