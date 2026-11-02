```@meta
CurrentModule = TwoBody
```

# Gradient Variational Method

The gradient variational method (GVM) optimizes all the nonlinear parameters of
the basis set at the same time. The linear coefficients are always optimal,
because they are the solution of the generalized eigenvalue problem

```math
\pmb{H} \pmb{c} = E \pmb{S} \pmb{c},
\qquad
\pmb{c}^\top \pmb{S} \pmb{c} = 1,
```

so the energy is a function of the exponents alone. Its derivative is the
Hellmann-Feynman derivative of the lowest eigenvalue,

```math
\frac{\partial E}{\partial a_k}
= \pmb{c}^\top
  \left(
    \frac{\partial \pmb{H}}{\partial a_k} - E \frac{\partial \pmb{S}}{\partial a_k}
  \right)
  \pmb{c}
= 2 c_k \sum_j c_j
  \left(
    \frac{\partial H_{kj}}{\partial a_k} - E \frac{\partial S_{kj}}{\partial a_k}
  \right),
```

where the second equality uses that only the ``k``-th row and column of the
matrices depend on ``a_k``. The derivatives of the matrix elements are
evaluated with forward-mode automatic differentiation
([ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)), so no analytic
formula has to be added for each operator. Only one eigenvalue problem and
``\mathcal{O}(n^2)`` matrix-element derivatives are needed per step, while a
derivative-free optimizer such as the one used by
[`optimize`](@ref Rayleigh-Ritz-Method) needs many energy evaluations to
estimate the same information.

The optimization variables are ``\log a_k``, which keeps every exponent
positive, and the optimizer is any gradient-based algorithm of
[Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl).

## Usage

A warm start optimizes the exponents of a given basis set. The four Gaussian
functions of [Thijssen (2013)](https://doi.org/10.1017/CBO9781139171397) are
already optimized, and GVM reproduces them:

```@example gvm-hydrogen
using TwoBody

H = Hamiltonian(
  Kinetic(hbar=1, m=1),
  Coulomb(coefficient=-1),
)
BS = BasisSet(
  SimpleGaussianBasis(13.00773),
  SimpleGaussianBasis(1.962079),
  SimpleGaussianBasis(0.444529),
  SimpleGaussianBasis(0.1219492),
)
result = solve(H, BS, GVM(), info=1, progress=false)
(energy=result.E[1], iterations=result.iterations, converged=result.converged,
 exponents=[basis.a for basis in result.basisset.basis])
```

A cold start needs `max_basis` and a prototype basis function. The initial
exponents are a geometric progression from `amin` to `amax`:

```@example gvm-hydrogen
cold = solve(H, SimpleGaussianBasis(), GVM(max_basis=10, amin=1e-2, amax=1e+2), info=1, progress=false)
(energy=cold.E[1], evaluations=cold.n_evaluations)
```

A single power-Slater function is the exact ``1s`` orbital of the hydrogen
atom, and the optimization recovers it from a poor initial guess:

```@example gvm-hydrogen
slater = solve(H, BasisSet(PowerSlaterBasis(0, 2.0)), GVM(), info=1, progress=false)
(energy=slater.E[1], exponent=slater.basisset[1].a)
```

The optimizer and its tolerances are options of the method:

```@example gvm-hydrogen
import Optim
solve(H, BS, GVM(optimizer=Optim.ConjugateGradient(), gradtol=1e-10), info=1, progress=false).E[1]
```

`history` is the optimization log, one entry per energy evaluation, and it is
printed with the result unless `progress=false`. `Inf` marks a rejected,
nearly linearly dependent basis set encountered by the line search.

GVM optimizes the exponents it is given, but it does not choose how many
functions to use or where to start. Combining it with the
[Stochastic Variational Method](@ref Stochastic-Variational-Method), which
explores the parameter space globally, is the recommended workflow:

```@example gvm-hydrogen
stochastic = solve(H, SimpleGaussianBasis(), SVM(max_basis=10, candidates=20, amin=1e-2, amax=1e+2), info=1)
(stochastic=stochastic.E[1], refined=solve(H, stochastic.basisset, GVM(), info=1, progress=false).E[1])
```

## API reference

```@docs; canonical=false
TwoBody.GradientVariationalMethod
TwoBody.solve(
  hamiltonian::Hamiltonian,
  basisset::BasisSet,
  method::GradientVariationalMethod,
)
TwoBody._gvm_gradient!
```
