@testset "VNN.jl" begin
  import Lux
  import Optimisers

  @testset "method validation" begin
    @test VNN === VariationalNeuralNetwork
    @test_throws ArgumentError VNN(Δr=0)
    @test_throws ArgumentError VNN(rₘₐₓ=0)
    @test_throws ArgumentError VNN(l=-1)
    @test_throws ArgumentError VNN(direction=:invalid)
    @test_throws ArgumentError VNN(architecture=Int[])
    @test_throws ArgumentError VNN(architecture=[2, 0])
    @test_throws ArgumentError VNN(maxiters=-1)
    @test_throws ArgumentError VNN(abstol=-1)
    @test_throws ArgumentError VNN(abstol=Inf)
    @test_throws ArgumentError VNN(patience=0)
    @test_throws ArgumentError VNN(every=0)
    @test_throws ArgumentError VNN(fdm=FiniteDifferenceMethod(), Δr=0.2)
  end

  @testset "standard model" begin
    hamiltonian = Hamiltonian(
      Kinetic(hbar=1, m=1),
      PowerLaw(coefficient=1 / 2, exponent=2),
    )
    method = VNN(
      Δr=0.2,
      rₘₐₓ=2.0,
      architecture=[2],
      maxiters=2,
      abstol=0,
      every=1,
    )
    result = solve(hamiltonian, method)
    repeated = solve(hamiltonian, method)

    @test isfinite(result.E)
    @test result.history == repeated.history
    @test result.E == last(result.history)
    @test result.n_iterations == 2
    @test length(result.history) == result.n_iterations + 1
    @test size(result.ψ) == size(method.fdm.R)
    @test 4π * method.fdm.Δr * (result.ψ' * result.J * result.ψ) ≈ 1 atol=1e-6
    @test isfinite(result.wavefunction(0.5))
  end

  @testset "custom hydrogen trial wavefunction" begin
    hamiltonian = Hamiltonian(
      Kinetic(hbar=1, m=1),
      Coulomb(coefficient=-1),
    )
    fdm = FiniteDifferenceMethod(Δr=0.1, rₘₐₓ=10.0)
    initial_weight = (rng, output_dimension, input_dimension) ->
      fill(0.6f0, output_dimension, input_dimension)
    model = Lux.Dense(
      1 => 1,
      value -> exp(-abs(value));
      use_bias=false,
      init_weight=initial_weight,
    )

    evaluation_method = VNN(fdm=fdm, maxiters=0)
    evaluation = solve(hamiltonian, model, evaluation_method)
    reference = solve(hamiltonian, r -> exp(-0.6f0 * r), fdm, 0, 1)

    @test evaluation.E ≈ reference.E rtol=1e-6
    @test evaluation.n_iterations == 0
    @test !evaluation.converged
    @test length(evaluation.history) == 1
    @test 4π * fdm.Δr * sum(fdm.R .^ 2 .* abs2.(evaluation.ψ)) ≈ 1 atol=1e-12
    @test evaluation.wavefunction(1.0) ≈ evaluation.ψ[10] rtol=1e-6

    training_method = VNN(
      fdm=fdm,
      optimizer=Optimisers.Adam(0.03),
      maxiters=100,
      abstol=1e-7,
      patience=5,
      every=25,
    )
    result = solve(hamiltonian, model, training_method)

    @test result.E < first(result.history)
    @test result.E < -0.49
    @test result.parameters.weight[1] ≈ 1 atol=0.02
    @test length(result.history) == result.n_iterations + 1
    @test result.n_iterations <= training_method.maxiters

    resumed = solve(
      hamiltonian,
      model,
      evaluation_method;
      parameters=result.parameters,
      states=result.states,
    )
    @test resumed.E ≈ result.E rtol=1e-10

    continued = solve(
      hamiltonian,
      model,
      VNN(fdm=fdm, optimizer=training_method.optimizer, maxiters=1, abstol=0);
      parameters=result.parameters,
      states=result.states,
      optimizer_state=result.optimizer_state,
    )
    @test continued.n_iterations == 1
  end

  @testset "invalid model output" begin
    zero_initializer = (rng, output_dimension, input_dimension) ->
      zeros(Float32, output_dimension, input_dimension)
    zero_model = Lux.Dense(
      1 => 1;
      use_bias=false,
      init_weight=zero_initializer,
    )
    wrong_size_model = Lux.Dense(1 => 2)
    method = VNN(Δr=0.5, rₘₐₓ=2.0, maxiters=0)
    hamiltonian = Hamiltonian(Kinetic())

    @test_throws ArgumentError solve(hamiltonian, zero_model, method)
    @test_throws DimensionMismatch solve(hamiltonian, wrong_size_model, method)
    @test_throws ArgumentError solve(
      hamiltonian,
      zero_model,
      method;
      optimizer_state=NamedTuple(),
    )
  end
end
