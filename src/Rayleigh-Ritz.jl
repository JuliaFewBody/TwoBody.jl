export RayleighRitzMethod, ResultRayleighRitz, ResultOptimizationRayleighRitz,
       solve, optimize, expectation, verification

import LinearAlgebra
import Optim
import QuadGK
import SpecialFunctions
import Subscripts

# method and results

struct RayleighRitzMethod{B<:Union{BasisSet,GeometricBasisSet}}
  basisset::B
end

RayleighRitzMethod(basis::Basis) = RayleighRitzMethod(BasisSet(basis))

Base.string(method::RayleighRitzMethod) = "RayleighRitzMethod($(method.basisset))"
Base.show(io::IO, method::RayleighRitzMethod) = print(io, Base.string(method))

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

function Base.getproperty(result::ResultRayleighRitz, symbol::Symbol)
  symbol === :data && return getfield(result, :data)
  return Base.getproperty(getfield(result, :data), symbol)
end

Base.haskey(result::ResultRayleighRitz, symbol::Symbol) =
  Base.haskey(getfield(result, :data), symbol)

function Base.getproperty(result::ResultOptimizationRayleighRitz, symbol::Symbol)
  symbol === :data && return getfield(result, :data)
  data = getfield(result, :data)
  return Base.haskey(data, symbol) ? Base.getproperty(data, symbol) : Base.getproperty(data.result, symbol)
end

Base.haskey(result::ResultOptimizationRayleighRitz, symbol::Symbol) =
  Base.haskey(getfield(result, :data), symbol) || Base.haskey(getfield(result, :data).result, symbol)

Base.show(io::IO, result::ResultRayleighRitz) = print(io, Base.string(result))
Base.show(io::IO, result::ResultOptimizationRayleighRitz) = print(io, Base.string(result))

ψ(result::ResultOptimizationRayleighRitz, r; n::Int=1) = ψ(result.result, r; n=n)

function expectation(result::ResultRayleighRitz, M::AbstractMatrix)
  size(M) == size(result.H) || throw(DimensionMismatch("the matrix dimensions must match the result"))
  return [LinearAlgebra.dot(result.C[:,n], M * result.C[:,n]) for n in eachindex(result.E)]
end

expectation(result::ResultRayleighRitz, operator::Operator) =
  expectation(result, matrix(operator, result.basisset))

expectation(result::ResultRayleighRitz, hamiltonian::Hamiltonian) =
  expectation(result, matrix(hamiltonian, result.basisset))

expectation(result::ResultOptimizationRayleighRitz, operator) =
  expectation(result.result, operator)

function verification(result::ResultRayleighRitz)
  norm = expectation(result, result.S)
  residual = abs.(expectation(result, result.H) .- result.E)
  return (; norm, residual)
end

verification(result::ResultOptimizationRayleighRitz) = verification(result.result)

function Base.string(result::ResultRayleighRitz)
  check = verification(result)
  text = "# method\n\n$(result.method)\n\n# eigenvalue\n\n"
  for n in eachindex(result.E)
    text *= "E$(Subscripts.sub(string(n))) = $(result.E[n])\n"
  end
  text *= "\n# verification\n\nn\tnorm\tresidual\n"
  for n in eachindex(result.E)
    text *= "$n\t$(check.norm[n])\t$(check.residual[n])\n"
  end
  return text
end

function Base.string(result::ResultOptimizationRayleighRitz)
  text = "# optimizer\n\n$(result.optimizer)\n"
  if result.progress
    text *= "\n# optimization log\n\nenergy\tparameters\n"
    for entry in result.history
      text *= "$(entry.energy)\t$(entry.parameters)\n"
    end
  end
  return text * "\n" * Base.string(result.result)
end

# solver

function solve(hamiltonian::Hamiltonian, method::RayleighRitzMethod; perturbation=Hamiltonian())
  source = method.basisset
  basisset = source isa GeometricBasisSet ? BasisSet(source.basis...) : source
  nₘₐₓ = length(basisset.basis)
  S = matrix(basisset)
  H = matrix(hamiltonian, basisset)
  E, C = LinearAlgebra.eigen(H, S)
  common = (method=method, hamiltonian=hamiltonian, perturbation=perturbation,
            basisset=basisset, nₘₐₓ=nₘₐₓ, H=H, S=S, E=E, C=C)
  return source isa GeometricBasisSet ?
    ResultRayleighRitz(; geometricbasisset=source, common...) :
    ResultRayleighRitz(; common...)
end

function optimize(hamiltonian::Hamiltonian, method::RayleighRitzMethod{<:BasisSet};
                  perturbation=Hamiltonian(), progress=true,
                  optimizer=Optim.NelderMead(), options...)
  basisset = method.basisset
  history = []
  optimization = Optim.optimize(
    x -> try
      varied = BasisSet([_replace_exponent(basisset.basis[i], x[i]) for i in keys(basisset.basis)]...)
      E = solve(hamiltonian, RayleighRitzMethod(varied); perturbation=perturbation).E[1]
      progress && push!(history, (energy=E, parameters=copy(x)))
      E
    catch
      progress && push!(history, (energy=Inf, parameters=copy(x)))
      Inf
    end,
    [float(basisset.basis[i].a) for i in keys(basisset.basis)],
    optimizer,
    Optim.Options(; options...)
  )

  optimized = BasisSet([_replace_exponent(basisset.basis[i], optimization.minimizer[i]) for i in keys(basisset.basis)]...)
  result = solve(hamiltonian, RayleighRitzMethod(optimized); perturbation=perturbation)
  return ResultOptimizationRayleighRitz(;
    result, initialmethod=method, optimizer, options, progress, history, optimization,
  )
end

function optimize(hamiltonian::Hamiltonian, method::RayleighRitzMethod{<:GeometricBasisSet};
                  perturbation=Hamiltonian(), progress=true,
                  optimizer=Optim.NelderMead(), options...)
  basisset = method.basisset
  history = []
  optimization = Optim.optimize(
    x -> try
      varied = GeometricBasisSet(basisset.basistype, x..., basisset.n,
                                 nₘₐₓ=basisset.nₘₐₓ, nₘᵢₙ=basisset.nₘᵢₙ)
      E = solve(hamiltonian, RayleighRitzMethod(varied); perturbation=perturbation).E[1]
      progress && push!(history, (energy=E, parameters=copy(x)))
      E
    catch
      progress && push!(history, (energy=Inf, parameters=copy(x)))
      Inf
    end,
    [float(basisset.r₁), float(basisset.rₙ)],
    optimizer,
    Optim.Options(; options...)
  )

  optimized = GeometricBasisSet(basisset.basistype, optimization.minimizer..., basisset.n,
                                nₘₐₓ=basisset.nₘₐₓ, nₘᵢₙ=basisset.nₘᵢₙ)
  result = solve(hamiltonian, RayleighRitzMethod(optimized); perturbation=perturbation)
  return ResultOptimizationRayleighRitz(;
    result, initialmethod=method, optimizer, options, progress, history, optimization,
  )
end

solve(hamiltonian::Hamiltonian, basisset::Union{BasisSet,GeometricBasisSet}; perturbation=Hamiltonian()) =
  solve(hamiltonian, RayleighRitzMethod(basisset); perturbation=perturbation)

solve(hamiltonian::Hamiltonian, basis::Basis; perturbation=Hamiltonian()) =
  solve(hamiltonian, RayleighRitzMethod(basis); perturbation=perturbation)

optimize(hamiltonian::Hamiltonian, basisset::Union{BasisSet,GeometricBasisSet}; kwargs...) =
  optimize(hamiltonian, RayleighRitzMethod(basisset); kwargs...)

optimize(hamiltonian::Hamiltonian, basis::Basis; kwargs...) =
  optimize(hamiltonian, RayleighRitzMethod(basis); kwargs...)

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

Configure a Rayleigh--Ritz calculation with a `BasisSet`, `GeometricBasisSet`,
or single `Basis`.
""" RayleighRitzMethod

@doc raw"""
`ResultRayleighRitz`

Result of a Rayleigh--Ritz calculation.
""" ResultRayleighRitz

@doc raw"""
`ResultOptimizationRayleighRitz`

Result of nonlinear basis optimization. Solver fields are forwarded from `result`.
""" ResultOptimizationRayleighRitz

@doc raw"""
`solve(hamiltonian, method::RayleighRitzMethod; perturbation=Hamiltonian())`

Solve ``\boldsymbol{H}\boldsymbol{c}=E\boldsymbol{S}\boldsymbol{c}`` and return a
`ResultRayleighRitz`. A basis or basis set may be passed in place of `method`.
""" solve(hamiltonian::Hamiltonian, method::RayleighRitzMethod; perturbation=Hamiltonian())

@doc raw"""
`optimize(hamiltonian, method::RayleighRitzMethod; kwargs...)`

Minimize the ground-state energy with Optim.jl. Individual exponents are varied
for a `BasisSet`; ``r_1`` and ``r_n`` are varied for a `GeometricBasisSet`.
""" optimize

@doc raw"""
`expectation(result, operator)`

Return the expectation value of a matrix, operator, or Hamiltonian for each state.
""" expectation

@doc raw"""
`verification(result)`

Return the norms and absolute eigenvalue residuals for a result.
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
