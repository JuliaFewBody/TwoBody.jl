export solve, optimize

import LinearAlgebra
import Optim
import Printf
import SpecialFunctions
import Subscripts

# result

struct ResultRayleighRitz
  data::Any
  ResultRayleighRitz(; args...) = new(NamedTuple(Dict(args)))
end

function ψ(result::ResultRayleighRitz, r; n::Int=1)
  return sum(result.C[i,n] * TwoBody.φ(result.basisset[i], r) for i in 1:result.nₘₐₓ)
end

Base.getproperty(result::ResultRayleighRitz, symbol::Symbol) = Base.getproperty(getfield(result,:data), symbol)
Base.haskey(result::ResultRayleighRitz, symbol::Symbol) = Base.haskey(getfield(result,:data), symbol)
Base.show(io::IO, result::ResultRayleighRitz) = print(io, Base.string(result))

function Base.string(result::ResultRayleighRitz)
  if 0 ≤ result.info
    # method
    text = "# method\n\n"
    text *= "Rayleigh-Ritz method with $(typeof(result.basisset[1]))\n"
    text *= "J. Thijssen, Computational Physics 2nd Edition (2013)\n"
    text *= "https://doi.org/10.1017/CBO9781139171397\n"
    # optimization
    if haskey(result, :optimizer)
      text *= "\n# optimizer\n\n"
      text *= string(result.optimizer) * "\n"
      text *= "P. K. Mogensen, A. N. Riseth, J. Open Source Softw., 3(24), 615 (2018)\n"
      text *= "https://doi.org/10.21105/joss.00615\n"
      # optimization log
      if result.progress
        text *= "\n# optimization log\n\n"
        for (E, x) in result.history
          if isinf(E)
            text *= Printf.@sprintf("%+.9e              %s\n", E, "[" * join([Printf.@sprintf("%+.3e", x[i]) for i in keys(x)], ", ") *"]")
          else
            text *= Printf.@sprintf("%+.9e  %s\n", E, "[" * join([Printf.@sprintf("%+.3e", x[i]) for i in keys(x)], ", ") *"]")
          end
        end
      end
      # initial basis set
      if haskey(result, :geometricbasisset)
        # initial geometric progression
        text *= "\n# initial geometric progression\n\n"
        text *= "type \t$(result.initialgeometricbasisset.basistype)\n"
        text *= string("range\tr", Subscripts.sub("$(result.initialgeometricbasisset.nₘᵢₙ)"), " - r", Subscripts.sub("$(result.initialgeometricbasisset.nₘₐₓ)"), "\n")
        text *= string("r", Subscripts.sub("$(result.initialgeometricbasisset.nₘᵢₙ)"), " \t", result.initialgeometricbasisset.r₁, "\n")
        text *= string("r", Subscripts.sub("$(result.initialgeometricbasisset.n)"  ), " \t", result.initialgeometricbasisset.rₙ, "\n")
      else
        # initial basis function
        text *= "\n# initial basis function\n\n"
        for n in 1:result.nₘₐₓ
          text *= Printf.@sprintf("φ%s(r) = TwoBody.φ(%s, r)\n", Subscripts.sub("$n"), result.initialbasisset[n])
        end
      end
    end
    # geometric progression
    if haskey(result, :geometricbasisset) && 0 < result.info
      if haskey(result, :optimizer)
        text *= "\n# optimized geometric progression\n\n"
      else
        text *= "\n# geometric progression\n\n"
      end
      text *= "type \t$(result.geometricbasisset.basistype)\n"
      text *= string("range\tr", Subscripts.sub("$(result.geometricbasisset.nₘᵢₙ)"), " - r", Subscripts.sub("$(result.geometricbasisset.nₘₐₓ)"), "\n")
      text *= string("r", Subscripts.sub("$(result.geometricbasisset.nₘᵢₙ)"), " \t", result.geometricbasisset.r₁, "\n")
      text *= string("r", Subscripts.sub("$(result.geometricbasisset.n)"  ), " \t", result.geometricbasisset.rₙ, "\n")
    end
    # basis set
    if haskey(result, :optimizer) && !haskey(result, :geometricbasisset)
      text *= "\n# optimized basis function\n\n"
    else
      text *= "\n# basis function\n\n"
    end
    for n in 1:result.nₘₐₓ
      text *= Printf.@sprintf("φ%s(r) = TwoBody.φ(%s, r)\n", Subscripts.sub("$n"), result.basisset[n])
    end
    # eigenfunction
    text *= "\n# eigenfunction\n\n"
    for n in 1:result.nₘₐₓ
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
    for n in 1:result.nₘₐₓ
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
  else
    # info < 0
    text = "ResultRayleighRitz:\n"
    text *= "  E: $(typeof(result.E))\n"
    for e in result.E
      text *= "    $(e)\n"
    end
    return text
  end
  return "error"
end

# solver

function solve(hamiltonian::Hamiltonian, basisset::BasisSet; perturbation=Hamiltonian(), info=4)

  # initialization
  nₘₐₓ = length(basisset.basis)

  # matrix element
  S = matrix(basisset)
  H = matrix(hamiltonian, basisset)

  # calculation
  E, C = LinearAlgebra.eigen(H, S)

  # expectation value
  expectation = Dict()
  if 0 < info
    expectation[:S] = [C[:,n]' * S * C[:,n] for n in 1:min(nₘₐₓ, info)]
    expectation[:H] = [C[:,n]' * H * C[:,n] for n in 1:min(nₘₐₓ, info)]
    expectation[:0] = [expectation[:H][n] - E[n] for n in 1:min(nₘₐₓ, info)]
    if !isempty(perturbation.terms)
      M = matrix(perturbation, basisset)
      expectation[:perturbation] = [C[:,n]' * M * C[:,n] for n in 1:min(nₘₐₓ, info)]  
    end
    for term in [hamiltonian.terms..., perturbation.terms...]
      M = matrix(term, basisset)
      expectation[term] = [C[:,n]' * M * C[:,n] for n in 1:min(nₘₐₓ, info)]
    end
  end

  # return
  if 0 < info
    # full information
    return ResultRayleighRitz(;
      info = info,
      hamiltonian = hamiltonian, 
      perturbation = perturbation,
      basisset = basisset,
      nₘₐₓ = nₘₐₓ,
      H = H,
      S = S,
      E = E,
      C = C,
      expectation = expectation,
    )
  else
    # for optimization
    return ResultRayleighRitz(;
      info = info,
      E = E,
    )
  end

end

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
        push!(history, (energy=E, parameters=x))
      end
      E
    catch
      if 0 ≤ info
        push!(history, (energy=Inf, parameters=x))
      end
      Inf
    end,
    [basisset.basis[i].a for i in keys(basisset.basis)],
    method = optimizer,
    options...
  )

  # result
  res = solve(hamiltonian, BasisSet([_replace_exponent(basisset.basis[i], res.minimizer[i]) for i in keys(basisset.basis)]...), perturbation=perturbation, info=info)
  if 0 ≤ info
    return ResultRayleighRitz(;
      optimizer = optimizer,
      initialbasisset = basisset,
      options = options,
      progress = progress,
      history = history,
      getfield(res, :data)...,
    )
  else
    return ResultRayleighRitz(;
      getfield(res, :data)...,
    )
  end

end

function solve(hamiltonian::Hamiltonian, basis::Basis; perturbation=Hamiltonian(), info=4)
  return solve(hamiltonian, BasisSet(basis); perturbation=perturbation, info=info)
end

function optimize(hamiltonian::Hamiltonian, basis::Basis; perturbation=Hamiltonian(), info=1, progress=true, optimizer=Optim.NelderMead(), options...)
  return optimize(hamiltonian, BasisSet(basis); perturbation=perturbation, info=info, progress=progress, optimizer=optimizer, options...)
end

function solve(hamiltonian::Hamiltonian, basisset::GeometricBasisSet; perturbation=Hamiltonian(), info=4)
  res = solve(hamiltonian, BasisSet(basisset.basis...); perturbation=perturbation, info=info)
  if 0 ≤ info
    return ResultRayleighRitz(;
      geometricbasisset = basisset,
      getfield(res, :data)...,
    )
  else
    return ResultRayleighRitz(;
      getfield(res, :data)...,
    )
  end
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
        push!(history, (energy=E, parameters=x))
      end
      E
    catch
      if 0 ≤ info
        push!(history, (energy=Inf, parameters=x))
      end
      Inf
    end,
    [basisset.r₁, basisset.rₙ],
    method = optimizer,
    options...
  )

  # result
  res = solve(hamiltonian, GeometricBasisSet(basisset.basistype, res.minimizer..., basisset.n, nₘₐₓ=basisset.nₘₐₓ, nₘᵢₙ=basisset.nₘᵢₙ); perturbation=perturbation, info=info)
  if 0 ≤ info
    return ResultRayleighRitz(;
      optimizer = optimizer,
      initialbasisset = BasisSet(basisset.basis...),
      initialgeometricbasisset = basisset,
      options = options,
      progress = progress,
      history = history,
      getfield(res, :data)...,
    )
  else
    return ResultRayleighRitz(;
      getfield(res, :data)...,
    )
  end

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

function element(o::RestEnergy, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)
  return o.m * o.c^2 * (π/(SGB1.a+SGB2.a))^(3/2)
end

function element(o::Laplacian, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)
  return - 6*π^(3/2)*SGB1.a*SGB2.a/(SGB1.a+SGB2.a)^(5/2)
end

function element(o::Kinetic, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)
  return o.hbar^2/(2*o.m) * 6*π^(3/2)*SGB1.a*SGB2.a/(SGB1.a+SGB2.a)^(5/2)
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

function element(o::PowerLaw, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)
  return o.coefficient * 2 * π * SpecialFunctions.gamma((o.exponent+3)/2) / (SGB1.a+SGB2.a)^((o.exponent+3)/2)
end

function element(o::Gaussian, SGB1::SimpleGaussianBasis, SGB2::SimpleGaussianBasis)
  return o.coefficient * (π/(o.exponent+SGB1.a+SGB2.a))^(3/2)
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
`solve(hamiltonian::Hamiltonian, basisset::BasisSet)`

This function returns the eigenvalues ``E``  and eigenvectors ``\pmb{c}`` for
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
""" optimize(hamiltonian::Hamiltonian, basisset::BasisSet; perturbation=Hamiltonian(), info=4, progress=true, optimizer=Optim.NelderMead(), options...)

@doc raw"""
`solve(hamiltonian::Hamiltonian, basis::Basis; perturbation=Hamiltonian(), info=4)`

This a solver for 1-basis calculations. This function returns `solve(hamiltonian, BasisSet(basis); perturbation=perturbation, info=info)`.
""" solve(hamiltonian::Hamiltonian, basis::Basis; perturbation=Hamiltonian(), info=4)

@doc raw"""
`optimize(hamiltonian::Hamiltonian, basis::Basis; perturbation=Hamiltonian(), info=4, optimizer=Optim.NelderMead())`

This a optimizer for 1-basis calculations. This function returns `optimize(hamiltonian, BasisSet(basis); perturbation=perturbation, info=info, progress=progress, optimizer=optimizer, options...)`.
""" optimize(hamiltonian::Hamiltonian, basis::Basis; perturbation=Hamiltonian(), info=1, progress=true, optimizer=Optim.NelderMead(), options...)

@doc raw"""
`solve(hamiltonian::Hamiltonian, basisset::GeometricBasisSet; perturbation=Hamiltonian(), info=4)`

This function is a wrapper for `solve(hamiltonian::Hamiltonian, basisset::BasisSet, ...)`.
""" solve(hamiltonian::Hamiltonian, basisset::GeometricBasisSet; perturbation=Hamiltonian(), info=4)

@doc raw"""
`optimize(hamiltonian::Hamiltonian, basisset::GeometricBasisSet; perturbation=Hamiltonian(), info=4, optimizer=Optim.NelderMead())`

This function minimizes the energy by optimizing $r_1$ and $r_n$ using Optim.jl.
```math
\frac{\partial E}{\partial r_1} = \frac{\partial E}{\partial r_n} = 0
```
""" optimize(hamiltonian::Hamiltonian, basisset::GeometricBasisSet; perturbation=Hamiltonian(), info=4, progress=true, optimizer=Optim.NelderMead(), options...)

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
