@testset "DMC.jl" begin
  H = Hamiltonian(
    Kinetic(hbar=1, m=1),
    PowerLaw(coefficient=1 / 2, exponent=2),
  )
  method = DiffusionMonteCarlo(
    n_steps=2_400,
    equilibration=400,
    n_walkers=1_000,
    Δt=0.01,
    initial_scale=1.0,
  )
  result = solve(H, method; rng=MersenneTwister(123))
  repeated = solve(H, method; rng=MersenneTwister(123))

  # The energy is a statistical estimate. With these settings the spread over
  # random streams is about 0.014 (reference implementation, 40 streams), and
  # the random streams differ between Julia versions, so the same seed does
  # not reproduce the same walk across versions. The tolerances below are
  # about four standard deviations wide.
  @test result isa ResultDiffusionMonteCarlo
  @test result.E ≈ 1.5 atol=0.06
  @test result.E == repeated.E
  @test result.walkers == repeated.walkers
  @test length(result.reference_energies) == method.n_steps
  @test length(result.growth_energies) == method.n_steps
  @test result.E ≈ sum(result.growth_energies[method.equilibration + 1:end]) / (method.n_steps - method.equilibration)
  @test size(result.walkers) == (3, method.n_walkers)
  @test isfinite(result.standard_error) && result.standard_error > 0
  @test occursin("# energy", string(result))

  hydrogen = Hamiltonian(Kinetic(hbar=1, m=1), Coulomb(coefficient=-1))
  hydrogen_method = DiffusionMonteCarlo(
    n_steps=1_500,
    equilibration=500,
    n_walkers=1_000,
    Δt=0.01,
  )
  # Without importance sampling the Coulomb singularity adds a bias of a few
  # percent; the spread over random streams is about 0.022.
  @test solve(hydrogen, hydrogen_method).E ≈ -0.5 atol=0.1

  # Laplacian with its coefficient and RestEnergy as a constant shift
  shifted = solve(
    Hamiltonian(Laplacian(coefficient=-0.5), PowerLaw(coefficient=1 / 2, exponent=2), RestEnergy(c=1, m=2)),
    method;
    rng=MersenneTwister(7),
  )
  @test shifted.E ≈ 3.5 atol=0.06

  small = DiffusionMonteCarlo(n_steps=20, equilibration=5, n_walkers=50)
  @test isfinite(solve(H, small; initial=0.5 .* ones(3, small.n_walkers)).E)
  @test DiffusionMonteCarlo(log_weight_cap=Inf).log_weight_cap == Inf
  @test_throws ArgumentError DiffusionMonteCarlo(log_weight_cap=0)
  @test_throws ArgumentError solve(Hamiltonian(Kinetic(hbar=1, m=1), Delta(coefficient=1.0)), small)
  @test_throws ArgumentError solve(Hamiltonian(RelativisticKinetic(c=1, m=1), Coulomb(coefficient=-1)), small)

  @test_throws ArgumentError DiffusionMonteCarlo(n_steps=0)
  @test_throws ArgumentError DiffusionMonteCarlo(equilibration=2_000)
  @test_throws ArgumentError DiffusionMonteCarlo(n_walkers=0)
  @test_throws ArgumentError DiffusionMonteCarlo(Δt=0)
  @test_throws ArgumentError solve(Hamiltonian(Coulomb(coefficient=-1)), method)
  @test_throws DimensionMismatch solve(H, method; initial=zeros(2, method.n_walkers))
end
