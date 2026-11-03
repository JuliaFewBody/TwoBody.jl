using LinearAlgebra

@testset "QTT.jl" begin
  @test_throws ArgumentError QuanticsTensorTrainMethod(quantics=1)
  @test_throws ArgumentError QuanticsTensorTrainMethod(r₀=-0.1)
  @test_throws ArgumentError QuanticsTensorTrainMethod(deflation_shift=0)

  method = QuanticsTensorTrainMethod(
    quantics=5,
    r₀=0.0,
    rₘₐₓ=8.0,
    tolerance=1e-8,
    maxbonddim=16,
    maxoperatorbonddim=32,
    maxiter=100,
    sweeps=4,
    deflation_shift=10.0,
  )
  @test length(method.R) == 2^method.quantics
  @test method.R[1] ≈ method.r₀ + method.Δr
  @test method.R[end] ≈ method.rₘₐₓ - method.Δr

  kinetic = TwoBody.matrix(Kinetic(hbar=1, m=1), method)
  for row in eachindex(method.R), column in eachindex(method.R)
    expected = row == column ? 1 / method.Δr^2 :
      abs(row - column) == 1 ? -1 / (2 * method.Δr^2) : 0
    @test qttvalue(kinetic, row, column) ≈ expected atol=1e-10
  end

  H = Hamiltonian(
    Kinetic(hbar=1, m=1),
    PowerLaw(coefficient=1 / 2, exponent=2),
  )
  Random.seed!(1234)
  result = solve(
    H,
    method;
    initial=r -> exp(-r^2 / 2),
    nₘₐₓ=2,
    info=0,
  )
  @test result.E ≈ [1.5, 3.5] atol=6e-2
  @test abs(result.overlaps[1, 2]) < 1e-6
  @test result.residuals[1] < 1e-5
  @test result.residuals[2] < 1e-5
  @test length(result.ψ) == length(result.u) == length(result.C) == 2
  @test all(order(state) == method.quantics for state in result.C)
  @test all(first(ranks(state)) == last(ranks(state)) == 1 for state in result.C)
  @test qttvalue(result.ψ[1], 1) isa Real

  discrete_norm = 4π * method.Δr *
    sum(abs2(qttvalue(result.u[1], index)) for index in eachindex(method.R))
  @test discrete_norm ≈ 1 atol=1e-8

  @testset "Discrete eigenpairs and radial normalization" begin
    N = length(method.R)
    reference_matrix = SymTridiagonal(
      1 / method.Δr^2 .+ method.R.^2 ./ 2,
      fill(-1 / (2 * method.Δr^2), N - 1),
    )
    reference = eigen(reference_matrix)
    dense_qtt = [qttvalue(result.H, i, j) for i in 1:N, j in 1:N]
    @test dense_qtt ≈ Matrix(reference_matrix) atol=1e-7 rtol=0
    @test result.E ≈ reference.values[1:2] atol=1e-5 rtol=0

    for level in 1:2
      c = [qttvalue(result.C[level], i) for i in 1:N]
      u = [qttvalue(result.u[level], i) for i in 1:N]
      ψ = [qttvalue(result.ψ[level], i) for i in 1:N]
      # Eigenvectors are only defined up to a global sign or phase.
      @test norm(c) ≈ 1 atol=1e-8
      @test abs(dot(reference.vectors[:, level], c)) ≈ 1 atol=1e-6
      @test u ≈ c / sqrt(4π * method.Δr) atol=1e-8
      @test method.R .* ψ ≈ u atol=1e-7
      @test 4π * method.Δr * sum(abs2, u) ≈ 1 atol=1e-8
      @test 4π * method.Δr * sum(abs2.(ψ) .* method.R.^2) ≈ 1 atol=1e-6
    end
  end

  @testset "QTT accessors" begin
    N = length(method.R)
    for tt in (result.C[1], result.H)
      returned_ranks = ranks(tt)
      returned_ranks[1] = 0
      @test first(ranks(tt)) == 1
    end
    @test_throws BoundsError qttvalue(result.C[1], 0)
    @test_throws BoundsError qttvalue(result.C[1], N + 1)
    @test_throws BoundsError qttvalue(result.H, 0, 1)
    @test_throws BoundsError qttvalue(result.H, 1, N + 1)
  end

  potential = TwoBody.matrix(PowerLaw(coefficient=1 / 2, exponent=2), method)
  @test qttvalue(potential, 5, 5) ≈ method.R[5]^2 / 2 atol=1e-7
  @test qttvalue(potential, 5, 6) ≈ 0 atol=1e-12
  @testset "Centrifugal and rest-energy diagonals" begin
    angular = QuanticsTensorTrainMethod(
      quantics=3,
      r₀=1.0,
      rₘₐₓ=4.0,
      l=1,
      tolerance=1e-10,
      maxbonddim=16,
    )
    Random.seed!(5678)
    angular_kinetic = TwoBody.matrix(Kinetic(hbar=1, m=1), angular)
    rest_energy = TwoBody.matrix(RestEnergy(m=2, c=3), angular)
    for i in eachindex(angular.R), j in eachindex(angular.R)
      expected = i == j ? 1 / angular.Δr^2 + 1 / angular.R[i]^2 :
        abs(i - j) == 1 ? -1 / (2 * angular.Δr^2) : 0
      @test qttvalue(angular_kinetic, i, j) ≈ expected atol=1e-7 rtol=0
      @test qttvalue(rest_energy, i, j) ≈ (i == j ? 18.0 : 0.0) atol=1e-7 rtol=0
    end
  end
  @test_throws ArgumentError TwoBody.matrix(Delta(), method)
end
