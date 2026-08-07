@testset "VMC.jl" begin
  using Random

  @testset "method validation" begin
    @test_throws ArgumentError VariationalMonteCarlo(n_steps=0)
    @test_throws ArgumentError VariationalMonteCarlo(burn_in=-1)
    @test_throws ArgumentError VariationalMonteCarlo(thinning=0)
    @test_throws ArgumentError VariationalMonteCarlo(n_walkers=0)
    @test_throws ArgumentError VariationalMonteCarlo(δ=0)
    @test_throws ArgumentError VariationalMonteCarlo(δ=Inf)
    @test_throws ArgumentError VariationalMonteCarlo(r₀=Float64[])
  end

  @testset "local energy" begin
    H = Hamiltonian(
      NonRelativisticKinetic(ℏ=1, m=1),
      PowerLawPotential(coefficient=1 / 2, exponent=2),
    )
    ψ(r) = exp(-sum(abs2, r) / 2)

    @test local_energy(H, ψ, [0.2, -0.3, 0.4]) ≈ 1.5 atol=1e-12
    @test_throws ArgumentError local_energy(
      Hamiltonian(RelativisticKinetic()),
      ψ,
      [0.2, -0.3, 0.4],
    )
  end

  @testset "harmonic oscillator" begin
    H = Hamiltonian(
      NonRelativisticKinetic(ℏ=1, m=1),
      PowerLawPotential(coefficient=1 / 2, exponent=2),
    )
    ψ(r) = exp(-sum(abs2, r) / 2)
    method = VariationalMonteCarlo(
      n_steps=200,
      burn_in=20,
      thinning=2,
      δ=1.0,
      r₀=[1.0, 0.0, 0.0],
    )
    result = solve(H, ψ, method; rng=MersenneTwister(123), info=0)

    @test result.E ≈ 1.5 atol=1e-12
    @test result.n_samples == method.n_steps
    @test result.n_discarded == 0
    @test size(result.samples) == (3, method.n_steps)
    @test length(result.local_energies) == method.n_steps
    @test 0 < result.acceptance_rate < 1
    mean_radius_squared =
      sum(sum(abs2, result.samples[:, i]) for i in axes(result.samples, 2)) / method.n_steps
    @test mean_radius_squared ≈ 1.5 atol=0.5

    singular_H = Hamiltonian(
      NonRelativisticKinetic(ℏ=1, m=1),
      FunctionPotential(f=r -> r < 1 ? Inf : r^2 / 2),
    )
    singular_result = solve(
      singular_H,
      ψ,
      method;
      rng=MersenneTwister(123),
      info=0,
    )
    @test 0 < singular_result.n_discarded < method.n_steps
  end

  @testset "multiple walkers" begin
    H = Hamiltonian(
      NonRelativisticKinetic(ℏ=1, m=1),
      PowerLawPotential(coefficient=1 / 2, exponent=2),
    )
    ψ(r) = exp(-sum(abs2, r) / 2)
    method = VariationalMonteCarlo(
      n_steps=50,
      burn_in=10,
      n_walkers=3,
      δ=1.0,
      r₀=[1.0, 0.0, 0.0],
    )
    result = solve(H, ψ, method; rng=MersenneTwister(123), info=0)
    total_samples = method.n_walkers * method.n_steps
    transitions_per_walker = method.burn_in + method.n_steps * method.thinning

    @test result.E ≈ 1.5 atol=1e-12
    @test result.n_samples == total_samples
    @test result.n_discarded == 0
    @test size(result.samples) == (3, total_samples)
    @test length(result.local_energies) == total_samples
    @test result.n_attempted == method.n_walkers * transitions_per_walker
    @test result.n_accepted == round(Int, result.acceptance_rate * result.n_attempted)
    @test result.n_burn_in_discarded == method.n_walkers * method.burn_in
    @test 0 < result.acceptance_rate < 1

    default_result = solve(H, ψ, method)
    repeated_default_result = solve(H, ψ, method)
    @test default_result.samples == repeated_default_result.samples
    @test default_result.E == repeated_default_result.E
  end

  @testset "Coulomb singularity" begin
    H = Hamiltonian(
      NonRelativisticKinetic(ℏ=1, m=1),
      CoulombPotential(coefficient=-1),
    )
    α = 0.2829
    ψ(r) = exp(-α * sum(abs2, r))
    method = VariationalMonteCarlo(n_steps=10_000, burn_in=2_000, δ=2.0)
    result = solve(H, ψ, method; rng=MersenneTwister(123), info=0)

    @test isfinite(result.E)
    @test result.E ≈ -0.42441317922678223 atol=0.03
  end
end
