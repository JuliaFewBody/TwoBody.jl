@testset "QTT.jl" begin
  @test_throws ArgumentError QuanticsTensorTrainMethod(quantics=1)
  @test_throws ArgumentError QuanticsTensorTrainMethod(r₀=-0.1)

  method = QuanticsTensorTrainMethod(
    quantics=7,
    r₀=0.0,
    rₘₐₓ=8.0,
    tolerance=1e-9,
    maxbonddim=32,
  )
  @test length(method.R) == 2^method.quantics
  @test method.R[1] ≈ method.r₀ + method.Δr
  @test method.R[end] ≈ method.rₘₐₓ - method.Δr

  tridiagonal = TwoBody._tridiagonal_qtt(3, -1, 2, -1)
  for row in 1:8, column in 1:8
    expected = row == column ? 2 : abs(row - column) == 1 ? -1 : 0
    @test qttvalue(tridiagonal, row, column) ≈ expected atol=1e-12
  end

  H = Hamiltonian(
    Kinetic(hbar=1, m=1),
    PowerLaw(coefficient=1 / 2, exponent=2),
  )
  result = solve(H, r -> exp(-r^2 / 2), method; info=0)
  @test result.E ≈ 1.5 atol=2e-3
  @test order(result.ψ) == method.quantics
  @test first(ranks(result.ψ)) == last(ranks(result.ψ)) == 1
  @test qttvalue(result.ψ, 1) isa Real

  potential = TwoBody.matrix(PowerLaw(coefficient=1 / 2, exponent=2), method)
  @test qttvalue(potential, 5, 5) ≈ method.R[5]^2 / 2 atol=1e-7
  @test qttvalue(potential, 5, 6) ≈ 0 atol=1e-12
  @test_throws ArgumentError TwoBody.matrix(Delta(), method)
end
