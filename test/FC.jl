@testset "Free Complement" begin
  basis = PowerSlaterBasis(0, 1.5)
  @test TwoBody.φ(basis, 2.0) ≈ exp(-3.0)

  hamiltonian = Hamiltonian(
    Kinetic(hbar = 1, m = 1),
    Coulomb(coefficient = -1),
  )

  first_complement = FC(hamiltonian, basis)
  @test [b.n for b in first_complement.basis] == [1, 0]
  @test all(b.a == 1.5 for b in first_complement.basis)

  second_complement = FC(hamiltonian, first_complement)
  @test [b.n for b in second_complement.basis] == [2, 1, 0]
  @test all(isfinite(TwoBody.φ(b, 0.0)) for b in second_complement.basis)
  @test_throws MethodError FC(basis)

  coulomb_complement = FC(
    Hamiltonian(Coulomb(coefficient = -1)),
    PowerSlaterBasis(1, 1.5),
  )
  @test [b.n for b in coulomb_complement.basis] == [2, 1]

  powerlaw_complement = FC(
    Hamiltonian(PowerLaw(coefficient = 1, exponent = 2)),
    basis,
  )
  @test [b.n for b in powerlaw_complement.basis] == [1, 3]
  @test_throws ArgumentError FC(
    Hamiltonian(PowerLaw(coefficient = 1, exponent = 0.5)),
    basis,
  )
  @test_throws ArgumentError FC(Hamiltonian(Gaussian()), basis)
  @test_throws ArgumentError FC(
    hamiltonian,
    BasisSet(basis, SimpleGaussianBasis(1.0)),
  )

  bra = PowerSlaterBasis(1, 1.2)
  ket = PowerSlaterBasis(2, 0.8)
  overlap = 4π * quadgk(r -> r^2 * TwoBody.φ(bra, r) * TwoBody.φ(ket, r), 0, Inf)[1]
  @test TwoBody.element(bra, ket) ≈ overlap

  kinetic = Kinetic(hbar = 1, m = 1)
  dφ(r) = ForwardDiff.derivative(x -> TwoBody.φ(ket, x), r)
  d²φ(r) = ForwardDiff.derivative(dφ, r)
  kinetic_integral = 4π * quadgk(
    r -> r^2 * TwoBody.φ(bra, r) * (-1/2 * (d²φ(r) + 2/r*dφ(r))),
    0,
    Inf,
  )[1]
  @test TwoBody.element(kinetic, bra, ket) ≈ kinetic_integral
  @test TwoBody.element(kinetic, bra, ket) ≈ TwoBody.element(kinetic, ket, bra)

  coulomb = Coulomb(coefficient = -1)
  coulomb_integral = 4π * quadgk(
    r -> r^2 * TwoBody.φ(bra, r) * TwoBody.V(coulomb, r) * TwoBody.φ(ket, r),
    0,
    Inf,
  )[1]
  @test TwoBody.element(coulomb, bra, ket) ≈ coulomb_integral

  basisset = BasisSet(basis)
  expected = [
    -0.375000000000000,
    -0.491025403784439,
    -0.499316142679167,
    -0.499954132454839,
    -0.499997229001983,
    -0.499999844138451,
  ]
  energies = Float64[]
  for _ in eachindex(expected)
    push!(energies, solve(hamiltonian, basisset, info=-1).E[1])
    basisset = FC(hamiltonian, basisset)
  end

  @test energies ≈ expected atol=5e-13
  @test issorted(energies; rev=true)
end
