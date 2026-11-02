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
  # random streams is about 0.01, and the floating-point summation order of
  # the branching weights differs between Julia versions, so the same seed
  # does not reproduce the same walk across versions. The tolerances below
  # are several standard deviations wide.
  @test result isa ResultDiffusionMonteCarlo
  @test result.E ≈ 1.5 atol=0.06
  @test result.E == repeated.E
  @test result.walkers == repeated.walkers
  @test length(result.reference_energies) == method.n_steps
  @test size(result.walkers) == (3, method.n_walkers)
  @test isfinite(result.standard_error)
  @test occursin("# energy", string(result))

  hydrogen = Hamiltonian(Kinetic(hbar=1, m=1), Coulomb(coefficient=-1))
  hydrogen_method = DiffusionMonteCarlo(
    n_steps=1_500,
    equilibration=500,
    n_walkers=1_000,
    Δt=0.01,
  )
  # Without importance sampling the Coulomb singularity adds a bias of a few
  # percent; the spread over random streams is about 0.02.
  @test solve(hydrogen, hydrogen_method).E ≈ -0.5 atol=0.1

  @test_throws ArgumentError DiffusionMonteCarlo(n_steps=0)
  @test_throws ArgumentError DiffusionMonteCarlo(equilibration=2_000)
  @test_throws ArgumentError DiffusionMonteCarlo(n_walkers=0)
  @test_throws ArgumentError DiffusionMonteCarlo(Δt=0)
  @test_throws ArgumentError solve(Hamiltonian(Coulomb(coefficient=-1)), method)
  @test_throws DimensionMismatch solve(H, method; initial=zeros(2, method.n_walkers))
end
