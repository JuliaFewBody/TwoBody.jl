@testset "Rayleigh-Ritz.jl" begin

  # Testing Results

  H = Hamiltonian(
    Kinetic(hbar = 1, m = 1),
    Coulomb(coefficient = -1),
  )
  BS = BasisSet(
    SimpleGaussianBasis(13.00773),
    SimpleGaussianBasis(1.962079),
    SimpleGaussianBasis(0.444529),
    SimpleGaussianBasis(0.1219492),
  )
  method = RayleighRitzMethod(BS)
  res = solve(H, method)

  @test res.method === method
  @test verification(res).norm ≈ ones(res.nₘₐₓ)
  @test verification(res).residual ≈ zeros(res.nₘₐₓ) atol=1e-12
  @test expectation(res, H) ≈ res.E atol=1e-12
  @test occursin("# verification", string(res))
  
  println("4π×∫|ψ(r)|²r²dr = 1")
  println("  i\tnumerical  \tanalytical")
  for i in 1:res.nₘₐₓ
    ψ(r) = TwoBody.ψ(res, r, n=i)
    numerical  = 4π*quadgk(r -> r^2 * abs(ψ(r))^2, 0, Inf, rtol=1e-12, maxevals=10^7)[1]
    analytical = 1
    error = iszero(analytical) ? abs(numerical-analytical) : abs((numerical-analytical)/analytical)
    acceptance = error < 1e-5
    @printf("%3d\t%.9f\t%.9f\t%s\n", i, numerical, analytical, acceptance ? "✔" :  "✗")
    @test acceptance
  end

  println("<ψₙ|ψₙ> = cₙ' * S * cₙ = 1")
  println("  i\tnumerical  \tanalytical")
  for i in 1:res.nₘₐₓ
    numerical  = res.C[:,i]' * res.S * res.C[:,i]
    analytical = 1
    error = iszero(analytical) ? abs(numerical-analytical) : abs((numerical-analytical)/analytical)
    acceptance = error < 1e-5
    @printf("%3d\t%.9f\t%.9f\t%s\n", i, numerical, analytical, acceptance ? "✔" :  "✗")
    @test acceptance
  end
  
  println("|<ψₙ|H|ψₙ> - E| = |cₙ' * H * cₙ - E| = 0")
  println("  i\tnumerical  \tanalytical")
  for i in 1:res.nₘₐₓ
    numerical  = abs((res.C[:,i]' * res.H * res.C[:,i]) - res.E[i])
    analytical = 0
    error = iszero(analytical) ? abs(numerical-analytical) : abs((numerical-analytical)/analytical)
    acceptance = error < 1e-5
    @printf("%3d\t%.9f\t%.9f\t%s\n", i, numerical, analytical, acceptance ? "✔" :  "✗")
    @test acceptance
  end
  
  println("Thijssen(2007)")
  println("  i\tnumerical  \tanalytical")
  numerical  = res.E[1]
  analytical = -0.499278
  error = iszero(analytical) ? abs(numerical-analytical) : abs((numerical-analytical)/analytical)
  acceptance = error < 1e-5
  @printf("%3d\t%.9f\t%.9f\t%s\n", 1, numerical, analytical, acceptance ? "✔" :  "✗")
  @test acceptance

  @testset "Pérez-Torres STO-3G contraction" begin
    exponents = (0.109818, 0.405771, 2.227660)
    contraction = (0.444635, 0.535328, 0.154329)
    primitives = ntuple(i -> SimpleGaussianBasis(exponents[i]), 3)
    normalization = ntuple(i -> (2exponents[i] / π)^(3/4), 3)
    coefficients = ntuple(i -> contraction[i] * normalization[i], 3)
    sto3g = ContractedBasis(coefficients, primitives)

    primitive_set = BasisSet(primitives...)
    coefficient_vector = collect(coefficients)
    primitive_overlap = TwoBody.matrix(primitive_set)

    @test TwoBody.element(sto3g, sto3g) ≈ coefficient_vector' * primitive_overlap * coefficient_vector
    @test TwoBody.element(sto3g, primitives[1]) ≈ sum(
      conj(coefficients[i]) * TwoBody.element(primitives[i], primitives[1])
      for i in eachindex(coefficients)
    )
    @test TwoBody.element(primitives[1], sto3g) ≈ sum(
      coefficients[i] * TwoBody.element(primitives[1], primitives[i])
      for i in eachindex(coefficients)
    )

    for operator in H.terms
      primitive_matrix = TwoBody.matrix(operator, primitive_set)
      @test TwoBody.element(operator, sto3g, sto3g) ≈ coefficient_vector' * primitive_matrix * coefficient_vector
      @test TwoBody.element(operator, sto3g, primitives[1]) ≈ sum(
        conj(coefficients[i]) * TwoBody.element(operator, primitives[i], primitives[1])
        for i in eachindex(coefficients)
      )
      @test TwoBody.element(operator, primitives[1], sto3g) ≈ sum(
        coefficients[i] * TwoBody.element(operator, primitives[1], primitives[i])
        for i in eachindex(coefficients)
      )
    end

    primitive_hamiltonian = TwoBody.matrix(H, primitive_set)
    @test TwoBody.element(H, sto3g, sto3g) ≈ coefficient_vector' * primitive_hamiltonian * coefficient_vector

    radius = Linear(coefficient=1)
    radius_squared = PowerLaw(coefficient=1, exponent=2)
    inverse_radius = Coulomb(coefficient=1)
    properties = Hamiltonian(radius, radius_squared, inverse_radius)

    contracted_result = solve(H, BasisSet(sto3g), perturbation=properties)
    uncontracted_result = solve(H, primitive_set, perturbation=properties)

    @test contracted_result.E[1] ≈ -0.494907096570 atol=1e-11
    @test uncontracted_result.E[1] ≈ -0.495010402012 atol=1e-11
    @test contracted_result.E[1] ≈ -0.494908 atol=2e-6
    @test uncontracted_result.E[1] ≈ -0.495010 atol=1e-6
    @test uncontracted_result.E[1] < contracted_result.E[1]

    @test expectation(contracted_result, radius)[1] ≈ 1.500392 atol=2e-6
    @test expectation(contracted_result, radius_squared)[1] ≈ 2.996124 atol=5e-6
    @test expectation(contracted_result, inverse_radius)[1] ≈ 0.989206 atol=2e-6
    @test 2 * expectation(contracted_result, H.terms[1])[1] ≈ 0.988596 atol=2e-6

    @test expectation(uncontracted_result, radius)[1] ≈ 1.514699 atol=1e-6
    @test expectation(uncontracted_result, radius_squared)[1] ≈ 3.046195 atol=1e-6
    @test expectation(uncontracted_result, inverse_radius)[1] ≈ 0.977070 atol=1e-6
    @test 2 * expectation(uncontracted_result, H.terms[1])[1] ≈ 0.964119 atol=1e-6

    @test contracted_result.S[1,1] ≈ 1.0 atol=2e-6
    @test 4π * quadgk(r -> r^2 * abs2(TwoBody.ψ(contracted_result, r)), 0, Inf)[1] ≈ 1.0 atol=1e-9
    @test 4π * quadgk(r -> r^2 * abs2(TwoBody.ψ(uncontracted_result, r)), 0, Inf)[1] ≈ 1.0 atol=1e-9

    # Independent radial quadrature does not use element() or matrix().
    numerical_components = function (orbital)
      orbital_derivative = r -> ForwardDiff.derivative(orbital, r)
      overlap = 4π * quadgk(r -> r^2 * abs2(orbital(r)), 0, Inf, rtol=1e-12)[1]
      kinetic = 2π * quadgk(r -> r^2 * abs2(orbital_derivative(r)), 0, Inf, rtol=1e-12)[1]
      coulomb = -4π * quadgk(r -> r * abs2(orbital(r)), 0, Inf, rtol=1e-12)[1]
      return (overlap=overlap, kinetic=kinetic, coulomb=coulomb, energy=(kinetic+coulomb)/overlap)
    end

    contracted_numerical = numerical_components(r -> TwoBody.φ(sto3g, r))
    uncontracted_numerical = numerical_components(r -> TwoBody.ψ(uncontracted_result, r))

    @test contracted_numerical.overlap ≈ TwoBody.element(sto3g, sto3g) atol=1e-10
    @test contracted_numerical.kinetic ≈ TwoBody.element(H.terms[1], sto3g, sto3g) atol=1e-10
    @test contracted_numerical.coulomb ≈ TwoBody.element(H.terms[2], sto3g, sto3g) atol=1e-10
    @test contracted_numerical.energy ≈ contracted_result.E[1] atol=1e-10
    @test uncontracted_numerical.overlap ≈ 1.0 atol=1e-10
    @test uncontracted_numerical.energy ≈ uncontracted_result.E[1] atol=1e-10

    complex_coefficients = (1 + 0.5im, -0.25im)
    complex_basis = ContractedBasis(complex_coefficients, (primitives[1], primitives[2]))
    @test TwoBody.element(complex_basis, primitives[1]) ≈ sum(
      conj(complex_coefficients[i]) * TwoBody.element(primitives[i], primitives[1])
      for i in eachindex(complex_coefficients)
    )
    @test TwoBody.element(primitives[1], complex_basis) ≈ sum(
      complex_coefficients[i] * TwoBody.element(primitives[1], primitives[i])
      for i in eachindex(complex_coefficients)
    )
  end

  # comparison with Antique.jl
  HA = Antique.HydrogenAtom(Z=1, mₑ=1.0, a₀=1.0, Eₕ=1.0, ℏ=1.0)

  println("ψ(r)")
  println("  r\tnumerical  \tanalytical")
  for r in 0.2:0.1:2.0
    numerical  = abs(TwoBody.ψ(res, r, n=1))
    analytical = abs(Antique.wavefunction(HA, r, 0, 0))
    error = iszero(analytical) ? abs(numerical-analytical) : abs((numerical-analytical)/analytical)
    acceptance = error < 1e-2
    @printf("%.1f\t%.9f\t%.9f\t%s\n", r, numerical, analytical, acceptance ? "✔" :  "✗")
  end

  # Testing Matrix Elements

  @testset "element()" begin

    # kinetic terms
    o = Kinetic(hbar=1, m=1)
    println(o)
    println("  i  j\tnumerical  \tanalytical")
    l = 0
    for j in keys(BS.basis)
    for i in keys(BS.basis)
      φᵢ(r) = TwoBody.φ(BS.basis[i], r)
      φⱼ(r) = TwoBody.φ(BS.basis[j], r)
      dφⱼ(r) = ForwardDiff.derivative(r -> φⱼ(r), r)
      d²φⱼ(r) = ForwardDiff.derivative(r -> dφⱼ(r), r)
      numerical  = 4*π*QuadGK.quadgk(r -> r^2 * φᵢ(r) * (-o.hbar^2/2/o.m * (d²φⱼ(r) + 2/r*dφⱼ(r) - l*(l+1)/r^2*φⱼ(r))), 0, Inf, maxevals=10^3)[1]
      analytical = TwoBody.element(o, BS.basis[i], BS.basis[j])
      error = isinf(analytical) ? 0.0 : abs((numerical-analytical)/analytical)
      acceptance = error < 1e-5
      @printf("%3d%3d\t%.9f\t%.9f\t%s\n", i, j, numerical, analytical, acceptance ? "✔" :  "✗")
      @test acceptance
    end
    end

    # potential terms
    for o in [
      Constant(),
      Linear(),
      Coulomb(),
      PowerLaw(),
      PowerLaw(exponent=2),
      Gaussian(),
      # Exponential(),
      # Yukawa(),
      # Logarithmic(),
      # Delta(),
    ]
      println(o)
      println("  i  j\tnumerical  \tanalytical")
      for j in keys(BS.basis)
      for i in keys(BS.basis)
        numerical  = 4*π*quadgk(r -> r^2 * TwoBody.V(o, r) * TwoBody.φ(BS.basis[i],r) * TwoBody.φ(BS.basis[j],r), 0, Inf, maxevals=10^3)[1]
        analytical = TwoBody.element(o, BS.basis[i], BS.basis[j])
        error = isinf(analytical) ? 0.0 : abs((numerical-analytical)/analytical)
        acceptance = error < 1e-5
        @printf("%3d%3d\t%.9f\t%.9f\t%s\n", i, j, numerical, analytical, acceptance ? "✔" :  "✗")
        @test acceptance
      end
      end
    end

    custom = Custom(f=r -> exp(-r^2))
    gaussian = Gaussian(coefficient=1, exponent=1)
    for b₁ in BS.basis, b₂ in BS.basis
      @test TwoBody.element(custom, b₁, b₂) ≈ TwoBody.element(gaussian, b₁, b₂) rtol=1e-9
    end

  end

  @testset "Custom potential" begin
    custom_H = Hamiltonian(
      Kinetic(hbar=1, m=1),
      Custom(f=r -> -exp(-r^2)),
    )
    gaussian_H = Hamiltonian(
      Kinetic(hbar=1, m=1),
      Gaussian(coefficient=-1, exponent=1),
    )

    custom_result = solve(custom_H, BS)
    gaussian_result = solve(gaussian_H, BS)
    @test custom_result.H ≈ gaussian_result.H rtol=1e-9
    @test custom_result.E ≈ gaussian_result.E rtol=1e-9
  end

  @testset "Optimization result" begin
    result = optimize(H, SimpleGaussianBasis(1); progress=false, iterations=5)
    @test result isa ResultOptimizationRayleighRitz
    @test result.result isa ResultRayleighRitz
    @test result.E == result.result.E
    @test isempty(result.history)
    @test occursin("# optimizer", string(result))
  end
end
