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
  for row in 1:8, column in 1:8
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

  potential = TwoBody.matrix(PowerLaw(coefficient=1 / 2, exponent=2), method)
  @test qttvalue(potential, 5, 5) ≈ method.R[5]^2 / 2 atol=1e-7
  @test qttvalue(potential, 5, 6) ≈ 0 atol=1e-12
  @test_throws ArgumentError TwoBody.matrix(Delta(), method)
end
