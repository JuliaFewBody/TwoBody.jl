export FiniteDifferenceMethod #, solve

import FiniteDifferenceMatrices
import SparseArrays
import ArnoldiMethod
import LinearAlgebra
import Subscripts
import Printf

# struct

struct FiniteDifferenceMethod
  Δr::Real
  rₘₐₓ::Real
  R::AbstractRange
  l::Int
  direction::Symbol
  solver::Symbol
  function FiniteDifferenceMethod(;Δr=0.1, rₘₐₓ=50.0, R=Δr:Δr:rₘₐₓ, l=0, direction=:c, solver=:LinearAlgebra)
    new(Δr, rₘₐₓ, R, l, direction, solver)
  end
end

_jacobian(method::FiniteDifferenceMethod) = SparseArrays.spdiagm(method.R .^ 2)

function _rayleigh_quotient(ψ::AbstractVector, H::AbstractMatrix, J::AbstractMatrix)
  denominator = LinearAlgebra.dot(ψ, J * ψ)
  return LinearAlgebra.dot(ψ, J * (H * ψ)) / denominator
end

function _normalization(ψ::AbstractVector, method::FiniteDifferenceMethod, J)
  norm² = 4 * π * method.Δr * real(LinearAlgebra.dot(ψ, J * ψ))
  isfinite(norm²) && 0 < norm² ||
    throw(ArgumentError("the wavefunction norm must be positive and finite"))
  return inv(sqrt(norm²))
end

_normalize_wavefunction(ψ::AbstractVector, method::FiniteDifferenceMethod, J) =
  _normalization(ψ, method, J) * ψ

Base.string(method::FiniteDifferenceMethod) = "FiniteDifferenceMethod(" * join(["$(symbol)=$(getproperty(method,symbol))" for symbol in fieldnames(typeof(method))], ", ") * ")"
Base.show(io::IO, method::FiniteDifferenceMethod) = print(io, Base.string(method))

# matrix

function matrix(o::Hamiltonian, method::FiniteDifferenceMethod)
  return sum(matrix(term, method) for term in o.terms)
end

function matrix(o::RestEnergy, method::FiniteDifferenceMethod)
  return SparseArrays.spdiagm([o.m * o.c^2 for r in method.R])
end

function matrix(o::Kinetic, method::FiniteDifferenceMethod)
  D  = FiniteDifferenceMatrices.fdmatrix(Int64(length(method.R)), n=1, m=2, d=method.direction, h=method.Δr, t=typeof(method.Δr))
  D² = FiniteDifferenceMatrices.fdmatrix(Int64(length(method.R)), n=2, m=2, d=method.direction, h=method.Δr, t=typeof(method.Δr))
  return -o.hbar^2/2/o.m * (D² + SparseArrays.spdiagm(2 ./ method.R) * D - method.l*(method.l+1) * SparseArrays.spdiagm(1 ./ method.R .^ 2))
end

function matrix(o::PotentialTerm, method::FiniteDifferenceMethod)
  return SparseArrays.spdiagm([V(o, r) for r in method.R])
end

# solver

function solve(hamiltonian::Hamiltonian, method::FiniteDifferenceMethod; perturbation=Hamiltonian(), info=4, nₘₐₓ=4)
  
  # Initialization
  nₘₐₓ = min(nₘₐₓ, length(method.R))

  # Hamiltonian
  H = matrix(hamiltonian, method)

  # Jacobian
  J = _jacobian(method)

  # Eigenvalues
  if method.solver == :LinearAlgebra
    E, C = LinearAlgebra.eigen(Matrix{Float64}(H))
  elseif method.solver == :ArnoldiMethod
    decomp, history = ArnoldiMethod.partialschur(H, nev=nₘₐₓ, tol=1e-9, which=:SR)
    E, C = ArnoldiMethod.partialeigen(decomp)
  else
    error("Unknown solver: $(method.solver)")
  end

  # Print
  if 0 < info
    println("\n# method\n")
    println(method)
    println("\n# eigenvalue\n")
    for n in 1:min(nₘₐₓ, info)
      Printf.@printf("E%s = %s\n", Subscripts.sub("$n"), "$(E[n])")
    end
    println("\n# others")
    println("\nn \tnorm, <ψₙ|ψₙ> = cₙ' * cₙ")
    for n in 1:min(nₘₐₓ, info)
      println("$n\t", C[:,n]' * C[:,n])
    end
    if !isempty(perturbation.terms)
      println("\nn \tperturbation")
      M = matrix(perturbation, method)
      for n in 1:min(nₘₐₓ, info)
        println("$n\t", C[:,n]' * M * C[:,n])
      end
      println("\nn \teigenvalue + perturbation")
      for n in 1:min(nₘₐₓ, info)
        println("$n\t", E[n] + C[:,n]' * M * C[:,n])
      end
    end
    println("\nn \terror check, |<ψₙ|H|ψₙ> - E| = |cₙ' * H * cₙ - E| = 0")
    for n in 1:min(nₘₐₓ, info)
      println("$n\t", abs(C[:,n]' * H * C[:,n] - E[n]))
    end
    for term in [hamiltonian.terms..., perturbation.terms...]
      println("\nn \texpectation value of $(term)")
      M = matrix(term, method)
      for n in 1:min(nₘₐₓ, info)
        println("$n\t", C[:,n]' * M * C[:,n])
      end
    end
    println()
  end

  # Normalization
  N = LinearAlgebra.diagm([1 / sqrt(4 * π * method.Δr * C[:,n]' * J * C[:,n]) for n in 1:nₘₐₓ])
  ψ = C[:, 1:nₘₐₓ] * N

  # Return
  if 0 ≤ info
    return (
      hamiltonian = hamiltonian, 
      perturbation = perturbation,
      method = method,
      nₘₐₓ = nₘₐₓ,
      H = H,
      J = J,
      E = E[1:nₘₐₓ],
      C = C[:, 1:nₘₐₓ],
      ψ = ψ,
    )
  else
    return (
      E = E[1:nₘₐₓ],
    )
  end
end

function solve(hamiltonian::Hamiltonian, wavefunction::Function, method::FiniteDifferenceMethod, info=4, nₘₐₓ=4)

  # Initialization
  nₘₐₓ = min(nₘₐₓ, length(method.R))

  # Hamiltonian
  H = matrix(hamiltonian, method)

  # Jacobian
  J = _jacobian(method)

  # Wave Function
  ψ = wavefunction.(method.R)

  # Energy
  E = _rayleigh_quotient(ψ, H, J)

  # Return
  if 0 ≤ info
    return (
      hamiltonian = hamiltonian,
      method = method,
      nₘₐₓ = nₘₐₓ,
      H = H,
      J = J,
      E = E,
      ψ = _normalize_wavefunction(ψ, method, J),
    )
  else
    return (
      E = E,
    )
  end

end

# docstring

@doc raw"""
`FiniteDifferenceMethod(Δr=0.1, rₘₐₓ=50.0, R=Δr:Δr:rₘₐₓ, l=0, direction=:c, solver=:LinearAlgebra)`

| Arguments | Default | Description |
| :-------- | :------ | :---------- |
| `Δr::Real`          | `0.1`            | Radial grid spacing. A uniform grid spacing is used, ``r_{i+1} = r_{i} + \Delta r``. |
| `rₘₐₓ::Real`        | `50.0`           | The maximum value of the radial grid. This value is not directly used in the calculation, but it is used to determine the `R`. |
| `R::AbstractRange`  | `Δr:Δr:rₘₐₓ`     | Radial grid. The origin must be excluded from the grid to avoid divergence of the Coulomb potential and the centrifugal potential at the origin. |
| `l::Int`            | `0`              | Angular momentum quantum number. This is a positive integer, ``0 \leq l``. |
| `direction::Symbol` | `:c`             | The direction of the finite difference, `:c` for central, :f for forward, `:b` for backward. |
| `solver::Symbol`    | `:LinearAlgebra` | The solver for eigenvalue problem, `:LinearAlgebra` or `:ArnoldiMethod`. |
""" FiniteDifferenceMethod

@doc raw"""
`matrix(o::Hamiltonian, method::FiniteDifferenceMethod)`

The matrix for the Hamiltonian is a sum of matrices for each term,

```math
\pmb{H} = \sum_i \pmb{O}_i.
```
""" matrix(o::Hamiltonian, method::FiniteDifferenceMethod)

@doc raw"""
`matrix(o::RestEnergy, method::FiniteDifferenceMethod)`

The matrix for the rest energy ``mc^2`` is a diagonal matrix,

```math
mc^2
\left(\begin{array}{ccccccc}
  1 & 0 & 0 & \ldots \\
  0 & 1 & 0 & \ldots \\
  0 & 0 & 1 & \ldots \\
  \vdots & \vdots & \vdots & \ddots \\
\end{array}\right).
```
""" matrix(o::RestEnergy, method::FiniteDifferenceMethod)

@doc raw"""
`matrix(o::Kinetic, method::FiniteDifferenceMethod)`

We use the shorthand notation $\psi'(r) = \frac{\mathrm{d}\psi}{\mathrm{d}r}(r)$ and $\psi''(r) = \frac{\mathrm{d}^{2}\psi}{\mathrm{d}r^{2}}(r)$. For the uniform grid spacing ($r_{i+1} = r_{i} + \Delta r$), the finite difference for the first derivative,

```math
\frac{\mathrm{d}\psi}{\mathrm{d}r}(r) = \frac{\psi(r+\Delta r) - \psi(r-\Delta r)}{2\Delta r} + O(\Delta r^{2})
```

is written as

```math
\left(\begin{array}{ccccc}
  \psi'(r_1) \\
  \psi'(r_2) \\
  \psi'(r_3) \\
  \psi'(r_4) \\
  \vdots
\end{array}\right)
\simeq
\frac{1}{2\Delta r}
\left(\begin{array}{ccccc}
   0 &  1 &  0 &  0 &\ldots \\
  -1 &  0 &  1 &  0 &\ldots \\
   0 & -1 &  0 &  1 &\ldots \\
   0 &  0 & -1 &  0 &\ldots \\
  \vdots & \vdots & \vdots & \vdots & \ddots \\
\end{array}\right)
\left(\begin{array}{ccccccc}
  \psi(r_1) \\
  \psi(r_2) \\
  \psi(r_3) \\
  \psi(r_4) \\
  \vdots
\end{array}\right),
```

and the finite difference for the second derivative,

```math
\frac{\mathrm{d}^{2}\psi}{\mathrm{d}r^{2}}(r) = \frac{\psi(r+\Delta r) - 2\psi(r) + \psi(r-\Delta r)}{\Delta r^{2}} + O(\Delta r^{2}).
```

is written as

```math
\left(\begin{array}{ccccc}
  \psi''(r_1) \\
  \psi''(r_2) \\
  \psi''(r_3) \\
  \psi''(r_4) \\
  \vdots
\end{array}\right)
\simeq
\frac{1}{\Delta r^2}
\left(\begin{array}{ccccccc}
  -2 &  1 &  0 &  0 & \ldots \\
   1 & -2 &  1 &  0 & \ldots \\
   0 &  1 & -2 &  1 & \ldots \\
   0 &  0 &  1 & -2 & \ldots \\
  \vdots & \vdots & \vdots & \vdots & \ddots 
\end{array}\right)
\left(\begin{array}{ccccccc}
  \psi(r_1) \\
  \psi(r_2) \\
  \psi(r_3) \\
  \psi(r_4) \\
  \vdots
\end{array}\right).
```

Similarly, the matrix for the kinetic energy,

```math
\hat{T}
=
-\frac{\hbar^2}{2\mu}
\left[
      \frac{\partial^2}{\partial r^2}
    + \frac{2}{r} \frac{\partial}{\partial r}
    - \frac{l(l+1)}{r^2}
\right]
```

is written as

```math
\pmb{T}
= - \frac{\hbar^2}{2\mu}
  \left[
    \frac{1}{{\Delta r}^2}
    \left(\begin{array}{ccccccc}
      -2 & 1 & 0 & \ldots \\
      1 & -2 & 1 & \ldots \\
      0 & 1 & -2 & \ldots \\
      \vdots & \vdots & \vdots & \ddots \\
    \end{array}\right)
    +
    \left(\begin{array}{ccccccc}
       2/r_1 & 0 & 0 & \ldots \\
       0 & 2/r_2 & 0 & \ldots \\
       0 & 0 & 2/r_3 & \ldots \\
      \vdots & \vdots & \vdots & \ddots \\
    \end{array}\right)
    \frac{1}{2\Delta r}
    \left(\begin{array}{ccccccc}
       0 &  1 &  0 & \ldots \\
      -1 &  0 &  1 & \ldots \\
       0 & -1 &  0 & \ldots \\
      \vdots & \vdots & \vdots & \ddots \\
    \end{array}\right)
    - l(l+1)
    \left(\begin{array}{ccccccc}
       1/{r_1}^2 & 0 & 0 & \ldots \\
       0 & 1/{r_2}^2 & 0 & \ldots \\
       0 & 0 & 1/{r_3}^2 & \ldots \\
      \vdots & \vdots & \vdots & \ddots \\
    \end{array}\right)
  \right].
```
""" matrix(o::Kinetic, method::FiniteDifferenceMethod)

@doc raw"""
`matrix(o::PotentialTerm, method::FiniteDifferenceMethod)`

The matrix for the potential energy ``V(r)`` is a diagonal matrix,

```math
\pmb{V} = 
\left(\begin{array}{ccccccc}
  V(r_1) & 0 & 0 & \ldots \\
  0 & V(r_2) & 0 & \ldots \\
  0 & 0 & V(r_3) & \ldots \\
  \vdots & \vdots & \vdots & \ddots \\
\end{array}\right).
```
""" matrix(o::PotentialTerm, method::FiniteDifferenceMethod)

@doc raw"""
`solve(hamiltonian::Hamiltonian, method::FiniteDifferenceMethod; perturbation=Hamiltonian(), info=4, nₘₐₓ=4)`

This method solve the eigenvalue problem for the Hamiltonian discretized as a sparse matrix with finite difference approximation,
```math
\pmb{H} \pmb{\psi} = E \pmb{\psi}.
```
The eigenvalue ``E`` is an approximation of the exact energy and the eigenvector ``\pmb{\psi}`` is a vector of the approximated values of the exact wavefunction ``\psi(r)`` on points of the grid,
```math
\pmb{\psi}
=
\left(\begin{array}{c}
  \psi(r_1) \\
  \psi(r_2) \\
  \psi(r_3) \\
  \vdots \\
\end{array}\right).
```
""" solve(hamiltonian::Hamiltonian, method::FiniteDifferenceMethod; perturbation=Hamiltonian(), info=4, nₘₐₓ=4)

@doc raw"""
`solve(hamiltonian::Hamiltonian, wavefunction::Function, method::FiniteDifferenceMethod, info=4, nₘₐₓ=4)`
""" solve(hamiltonian::Hamiltonian, wavefunction::Function, method::FiniteDifferenceMethod, info=4, nₘₐₓ=4)
