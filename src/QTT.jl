export QuanticsTensorTrainMethod, order, ranks, qttvalue

"Return the number of binary sites in a QTT."
function order end

"Return the left boundary, internal, and right boundary ranks of a QTT."
function ranks end

"Contract a QTT at one vector or matrix index without materializing its entries."
function qttvalue end

struct QuanticsTensorTrainMethod
  quantics::Int
  r₀::Float64
  rₘₐₓ::Float64
  R::LinRange{Float64,Int}
  Δr::Float64
  l::Int
  tolerance::Float64
  maxbonddim::Int
  maxoperatorbonddim::Int
  maxiter::Int
  sweeps::Int
  deflation_shift::Float64

  function QuanticsTensorTrainMethod(;
    quantics=10,
    r₀=0.0,
    rₘₐₓ=50.0,
    l=0,
    tolerance=1e-10,
    maxbonddim=64,
    maxoperatorbonddim=2 * maxbonddim,
    maxiter=100,
    sweeps=4,
    deflation_shift=10.0,
  )
    quantics >= 2 || throw(ArgumentError("quantics must be at least 2"))
    0 <= r₀ < rₘₐₓ || throw(ArgumentError("the radial interval must satisfy 0 <= r₀ < rₘₐₓ"))
    l >= 0 || throw(ArgumentError("l must be nonnegative"))
    tolerance > 0 || throw(ArgumentError("tolerance must be positive"))
    maxbonddim >= 1 || throw(ArgumentError("maxbonddim must be positive"))
    maxoperatorbonddim >= 1 || throw(ArgumentError("maxoperatorbonddim must be positive"))
    maxiter >= 1 || throw(ArgumentError("maxiter must be positive"))
    sweeps >= 1 || throw(ArgumentError("sweeps must be positive"))
    deflation_shift > 0 || throw(ArgumentError("deflation_shift must be positive"))

    npoints = 1 << quantics
    Δr = Float64(rₘₐₓ - r₀) / (npoints + 1)
    R = range(Float64(r₀) + Δr, Float64(rₘₐₓ) - Δr; length=npoints)
    new(
      quantics,
      Float64(r₀),
      Float64(rₘₐₓ),
      R,
      Δr,
      l,
      Float64(tolerance),
      maxbonddim,
      maxoperatorbonddim,
      maxiter,
      sweeps,
      Float64(deflation_shift),
    )
  end
end

Base.string(method::QuanticsTensorTrainMethod) =
  "QuanticsTensorTrainMethod(" *
  join(["$(symbol)=$(getproperty(method, symbol))" for symbol in
        (:quantics, :r₀, :rₘₐₓ, :l, :tolerance, :maxbonddim,
         :maxoperatorbonddim, :maxiter, :sweeps, :deflation_shift)], ", ") * ")"
Base.show(io::IO, method::QuanticsTensorTrainMethod) = print(io, string(method))

function _qtt_unavailable()
  throw(ArgumentError(
    "QTT support requires TensorTrainNumerics.jl. Run " *
    "`Pkg.add(\"TensorTrainNumerics\")`, then load it with " *
    "`using TensorTrainNumerics`.",
  ))
end

matrix(::Any, ::QuanticsTensorTrainMethod) = _qtt_unavailable()

function _qtt_extension()
  extension = Base.get_extension(@__MODULE__, :TwoBodyQTTExt)
  isnothing(extension) && _qtt_unavailable()
  return extension
end

function solve(
  hamiltonian::Hamiltonian,
  method::QuanticsTensorTrainMethod;
  initial=r -> exp(-r),
  info=4,
  nₘₐₓ=4,
)
  return _qtt_extension().solve_qtt(
    hamiltonian,
    method;
    initial=initial,
    info=info,
    nₘₐₓ=nₘₐₓ,
  )
end

function solve(
  hamiltonian::Hamiltonian,
  wavefunction::Function,
  method::QuanticsTensorTrainMethod;
  info=4,
  nₘₐₓ=4,
)
  return _qtt_extension().solve_qtt(
    hamiltonian,
    wavefunction,
    method;
    info=info,
    nₘₐₓ=nₘₐₓ,
  )
end

@doc raw"""
    QuanticsTensorTrainMethod(; quantics=10, r₀=0.0, rₘₐₓ=50.0, l=0,
                               tolerance=1e-10, maxbonddim=64,
                               maxoperatorbonddim=128, maxiter=100, sweeps=4,
                               deflation_shift=10.0)

Configure a radial QTT grid with ``2^{\mathtt{quantics}}`` uniformly spaced interior
points. The excluded endpoints `r₀` and `rₘₐₓ` impose zero-value (Dirichlet)
boundaries. `maxbonddim` and `maxoperatorbonddim` bound state and MPO ranks;
`deflation_shift` is the penalty for each previously computed state.
""" QuanticsTensorTrainMethod

@doc raw"""
    solve(hamiltonian, method::QuanticsTensorTrainMethod;
          initial=r -> exp(-r), info=4, nₘₐₓ=4)
    solve(hamiltonian, initial::Function, method::QuanticsTensorTrainMethod;
          info=4, nₘₐₓ=4)

Compute the lowest `nₘₐₓ` QTT eigenstates. `initial` seeds the first DMRG solve;
`info` controls progress output. The result contains energies `E`, unit-norm tensor
trains `C`, radial functions `ψ` and `u`, and the diagnostics `overlaps`, `residuals`,
`histories`, and `rank_histories`.
""" solve(
  hamiltonian::Hamiltonian,
  method::QuanticsTensorTrainMethod;
  initial=r -> exp(-r),
  info=4,
  nₘₐₓ=4,
)
