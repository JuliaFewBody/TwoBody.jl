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
  res = solve(H, BS)

  @test TwoBody.matrix(H, BS) ≈ sum(TwoBody.matrix(term, BS) for term in H.terms)
  
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

  @testset "Weinstein bound" begin
    bound = weinstein(res)
    @test bound.E == res.E[1]
    @test bound.lower < -0.5 < bound.upper
    @test bound.variance >= 0
    @test sprint(show, bound) == "WeinsteinBound(E=$(bound.E), lower=$(bound.lower), upper=$(bound.upper))"
    @test_throws BoundsError weinstein(res; n=0)
    @test_throws ArgumentError weinstein(solve(H, BS; info=0))

    # normalized GEM primitives use the radial measure without 4π
    gem = solve(H, GeometricBasisSet(GaussianBasis, 0.1, 80.0, 20))
    gem_bound = weinstein(gem)
    @test gem_bound.lower < -0.5 < gem_bound.upper
    @test 0 <= gem_bound.variance < 0.1
    @test gem_bound.variance < bound.variance

    # the exact 1s orbital has zero variance up to quadrature error
    exact = weinstein(solve(H, PowerSlaterBasis(0, 1.0)))
    @test exact.E ≈ -0.5 atol=1e-12
    @test exact.variance ≈ 0 atol=1e-7

    # the centrifugal term enters through basis.l, also inside a contraction
    p_wave = solve(H, BasisSet((GaussianBasis(a; l=1, m=0) for a in TwoBody.geometric(0.1, 80.0, 20))...))
    p_bound = weinstein(p_wave)
    @test p_bound.lower < -0.125 < p_bound.upper
    @test 0 <= p_bound.variance < 0.1
    contracted_p = ContractedBasis((1.0, 0.5), (GaussianBasis(1.0; l=1), GaussianBasis(2.0; l=1)))
    @test TwoBody._weinstein_angular_momentum(BasisSet(contracted_p, GaussianBasis(0.5; l=1))) == 1
    @test_throws ArgumentError TwoBody._weinstein_angular_momentum(BasisSet(GaussianBasis(1.0; l=1), GaussianBasis(2.0; l=0)))

    # unsupported operators are rejected explicitly
    contact = solve(Hamiltonian(Kinetic(hbar=1, m=1), Delta(coefficient=1.0)), GeometricBasisSet(GaussianBasis, 0.1, 10.0, 5))
    @test_throws ArgumentError weinstein(contact)

    # the Laplacian element honours its coefficient
    @test TwoBody.element(Laplacian(coefficient=-0.5), BS[1], BS[2]) ≈ TwoBody.element(Kinetic(hbar=1, m=1), BS[1], BS[2])
  end

  @testset "hadron spectroscopy" begin
    Mqq = 0.725
    Mc = 1.750
    μ = inv(inv(Mqq) + inv(Mc))
    lambda_c = Hamiltonian(
      RestEnergy(m=Mqq),
      RestEnergy(m=Mc),
      Kinetic(hbar=1, m=μ),
      Coulomb(coefficient=-0.06 / μ),
      Linear(coefficient=0.165),
      Constant(constant=-0.83116597),
    )
    lambda_c_basis = GeometricBasisSet(GaussianBasis, 0.01, 9.0, 40)
    @test solve(lambda_c, lambda_c_basis; info=0).E[1] ≈ 2.286 atol=1e-6

    m₁ = 1.836
    m₂ = 1.836
    κ = 0.5069
    κ′ = 1.8609
    spin = -3
    μ = inv(inv(m₁) + inv(m₂))
    r₀ = 1.6553 * (2m₁ * m₂ / (m₁ + m₂))^(-0.2204)
    meng = Hamiltonian(
      RestEnergy(m=m₁),
      RestEnergy(m=m₂),
      Kinetic(hbar=1, m=μ),
      Coulomb(coefficient=-κ),
      Linear(coefficient=0.1653),
      Constant(constant=-0.8321),
      Gaussian(
        coefficient=2π * κ′ * spin / (3m₁ * m₂ * (sqrt(π) * r₀)^3),
        exponent=inv(r₀^2),
      ),
    )
    meng_basis = GeometricBasisSet(GaussianBasis, 0.1, 80.0, 20)
    @test solve(meng, meng_basis; info=0).E[1] ≈ 3.005 atol=1e-3

    m₁ = 1.515
    m₂ = 1.515
    αs = 0.285
    spin = -3/4
    μ = inv(inv(m₁) + inv(m₂))
    λ = 1.437 * sqrt(μ)
    arifi = Hamiltonian(
      RestEnergy(m=m₁),
      RestEnergy(m=m₂),
      RelativisticKinetic(m=m₁),
      RelativisticKinetic(m=m₂),
      Constant(constant=-0.189),
      Linear(coefficient=0.092),
      Coulomb(coefficient=-4αs/3),
      Gaussian(
        coefficient=32π * αs * (λ / sqrt(π))^3 * spin / (9m₁ * m₂),
        exponent=λ^2,
      ),
    )
    arifi_basis = GeometricBasisSet(GaussianBasis, 0.358, 2.720, 10)
    @test solve(arifi, arifi_basis; info=0).E[1] ≈ 3.019 atol=2e-3
  end

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

    @test contracted_result.expectation[radius][1] ≈ 1.500392 atol=2e-6
    @test contracted_result.expectation[radius_squared][1] ≈ 2.996124 atol=5e-6
    @test contracted_result.expectation[inverse_radius][1] ≈ 0.989206 atol=2e-6
    @test 2 * contracted_result.expectation[H.terms[1]][1] ≈ 0.988596 atol=2e-6

    @test uncontracted_result.expectation[radius][1] ≈ 1.514699 atol=1e-6
    @test uncontracted_result.expectation[radius_squared][1] ≈ 3.046195 atol=1e-6
    @test uncontracted_result.expectation[inverse_radius][1] ≈ 0.977070 atol=1e-6
    @test 2 * uncontracted_result.expectation[H.terms[1]][1] ≈ 0.964119 atol=1e-6

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

    custom_result = solve(custom_H, BS, info=1)
    gaussian_result = solve(gaussian_H, BS, info=1)
    @test custom_result.H ≈ gaussian_result.H rtol=1e-9
    @test custom_result.E ≈ gaussian_result.E rtol=1e-9

    custom_basis = BasisSet((CustomBasis(r -> exp(-b.a * r^2)) for b in BS.basis)...)
    custom_basis_result = solve(gaussian_H, custom_basis, info=1)
    @test custom_basis_result.S ≈ gaussian_result.S rtol=1e-9
    @test custom_basis_result.H ≈ gaussian_result.H rtol=1e-9
    @test custom_basis_result.E ≈ gaussian_result.E rtol=1e-9

    # mixed with the unnormalized Gaussians that share the 4π r² measure
    mixed = BasisSet(CustomBasis(r -> exp(-BS.basis[1].a * r^2)), BS.basis[2:end]...)
    mixed_result = solve(gaussian_H, mixed, info=1)
    @test mixed_result.S ≈ gaussian_result.S rtol=1e-9
    @test mixed_result.E ≈ gaussian_result.E rtol=1e-9

    # a contact term evaluates the functions at the origin
    delta = Delta(coefficient=0.7)
    c1 = CustomBasis(r -> exp(-r^2))
    c2 = CustomBasis(r -> (1 + r) * exp(-2r))
    @test TwoBody.element(delta, c1, c2) ≈ 0.7
    @test TwoBody.element(delta, c1, SimpleGaussianBasis(2.0)) ≈ 0.7
    @test TwoBody.element(delta, PowerSlaterBasis(1, 1.0), c1) ≈ 0

    # the normalized GEM primitives use a different measure and are rejected
    @test_throws MethodError TwoBody.element(c1, GaussianBasis(1.0))
    @test_throws MethodError TwoBody.element(gaussian_H.terms[1], GaussianBasis(1.0), c1)

    # the order of custom and primitive functions does not matter, and a
    # CustomBasis is a primitive that can be contracted
    reversed = BasisSet(BS.basis[1:end-1]..., CustomBasis(r -> exp(-BS.basis[end].a * r^2)))
    @test solve(gaussian_H, reversed, info=1).E ≈ gaussian_result.E rtol=1e-9
    contracted_custom = ContractedBasis((1.0, 0.5), (SimpleGaussianBasis(1.0), CustomBasis(r -> exp(-2r^2))))
    contracted_gaussian = ContractedBasis((1.0, 0.5), (SimpleGaussianBasis(1.0), SimpleGaussianBasis(2.0)))
    @test TwoBody.element(contracted_custom, contracted_custom) ≈ TwoBody.element(contracted_gaussian, contracted_gaussian) rtol=1e-9
    @test TwoBody.element(gaussian_H, contracted_custom, contracted_custom) ≈ TwoBody.element(gaussian_H, contracted_gaussian, contracted_gaussian) rtol=1e-9
    @test TwoBody.element(RestEnergy(c=2.0, m=1.5), c1, SimpleGaussianBasis(2.0)) ≈ 6 * TwoBody.element(c1, SimpleGaussianBasis(2.0))
    @test TwoBody.element(delta, SimpleGaussianBasis(1.0), SimpleGaussianBasis(2.0)) ≈ 0.7

    # a Slater-type custom function: the exact hydrogen 1s orbital, also mixed
    # with the analytic PowerSlaterBasis
    hydrogen = Hamiltonian(Kinetic(hbar=1, m=1), Coulomb(coefficient=-1))
    @test solve(hydrogen, CustomBasis(r -> exp(-r)), info=1).E[1] ≈ -0.5 atol=1e-8
    slater_pair = BasisSet(PowerSlaterBasis(0, 1.0), CustomBasis(r -> exp(-1.5r)))
    slater_reference = BasisSet(PowerSlaterBasis(0, 1.0), PowerSlaterBasis(0, 1.5))
    @test solve(hydrogen, slater_pair, info=1).E ≈ solve(hydrogen, slater_reference, info=1).E rtol=1e-8

    @test_throws ArgumentError TwoBody.element(Tabulated(0.0:0.1:1.0, ones(11)), c1, c2)
  end

  @testset "RayleighRitzMethod and results" begin
    @test :RayleighRitzMethod in names(TwoBody)
    @test :ResultRayleighRitz in names(TwoBody)
    @test :ResultOptimizationRayleighRitz in names(TwoBody)
    @test :expectation in names(TwoBody)
    @test :verification in names(TwoBody)

    # method
    method = RayleighRitzMethod(BS)
    @test method.basisset === BS
    @test sprint(show, method) == string(method)
    @test startswith(string(method), "RayleighRitzMethod(BasisSet(")
    single = RayleighRitzMethod(SimpleGaussianBasis(1.0))
    @test single.basisset isa BasisSet
    @test length(single.basisset) == 1
    geometric = GeometricBasisSet(SimpleGaussianBasis, 0.1, 10.0, 5)
    @test startswith(string(RayleighRitzMethod(geometric)), "RayleighRitzMethod(GeometricBasisSet(")

    # solve
    result = solve(H, method)
    @test result isa ResultRayleighRitz
    @test result.method === method
    @test result.E ≈ res.E
    @test res.method isa RayleighRitzMethod
    @test res.method.basisset === BS
    @test solve(H, SimpleGaussianBasis(1.0)).method.basisset isa BasisSet
    geometric_result = solve(H, geometric)
    @test geometric_result.geometricbasisset === geometric
    @test geometric_result.method.basisset === geometric
    @test geometric_result.basisset isa BasisSet
    @test geometric_result.E ≈ solve(H, RayleighRitzMethod(geometric)).E

    # info
    minimal = solve(H, method; info=-1)
    @test minimal.E ≈ res.E
    @test !haskey(minimal, :C)
    @test !haskey(minimal, :method)
    @test_throws ArgumentError TwoBody.expectation(minimal, res.S)
    @test_throws ArgumentError TwoBody.expectation(minimal, H)
    @test_throws ArgumentError TwoBody.verification(minimal)
    @test startswith(string(minimal), "ResultRayleighRitz:")
    @test startswith(string(solve(H, BS; info=0)), "ResultRayleighRitz:")
    @test startswith(string(solve(H, geometric; info=0)), "ResultRayleighRitz:")

    # expectation
    @test TwoBody.expectation(res, res.S) ≈ res.expectation[:S]
    @test TwoBody.expectation(res, H) ≈ res.expectation[:H]
    @test TwoBody.expectation(res, H) ≈ res.E
    for term in H.terms
      @test TwoBody.expectation(res, term) ≈ res.expectation[term]
    end
    @test_throws DimensionMismatch TwoBody.expectation(res, zeros(2, 2))
    perturbed = solve(H, BS; perturbation=Hamiltonian(Linear(coefficient=1)), info=2)
    @test length(perturbed.expectation[:perturbation]) == 2
    @test TwoBody.expectation(perturbed, Linear(coefficient=1))[1:2] ≈ perturbed.expectation[:perturbation]

    # verification
    check = TwoBody.verification(res)
    @test keys(check) == (:norm, :residual)
    @test length(check.norm) == res.nₘₐₓ
    @test length(check.residual) == res.nₘₐₓ
    @test all(n -> isapprox(n, 1; atol=1e-8), check.norm)
    @test all(r -> r < 1e-8, check.residual)
    @test check.norm ≈ res.expectation[:S]
    @test check.residual ≈ abs.(res.expectation[:0]) atol=1e-12

    # string
    text = string(res)
    @test sprint(show, res) == text
    for header in ["# method", "# basis function", "# eigenfunction", "# eigenvalue", "# verification", "# expectation value"]
      @test occursin(header, text)
    end
    @test !occursin("# optimizer", text)
    @test occursin("# geometric progression", string(geometric_result))

    # results extended by other solvers (e.g. BVM) override `method` and add keys
    extended = ResultRayleighRitz(; getfield(res, :data)..., method=:other, history=[1, 2, 3])
    @test extended.method === :other
    @test extended.E == res.E
    @test string(extended) == text
    legacy = ResultRayleighRitz(; getfield(res, :data)..., optimizer=:legacy, initialbasisset=BS, progress=false, history=[])
    @test occursin("# optimizer", string(legacy))
    @test occursin("# optimized basis function", string(legacy))

    # optimize
    optimized = optimize(H, SimpleGaussianBasis(1.0); progress=false, iterations=5)
    @test optimized isa ResultOptimizationRayleighRitz
    @test optimized.result isa ResultRayleighRitz
    @test optimized.E == optimized.result.E
    @test optimized.C == optimized.result.C
    @test optimized.basisset === optimized.result.basisset
    @test optimized.nₘₐₓ == 1
    @test optimized.info == 1
    @test optimized.expectation[:S] == optimized.result.expectation[:S]
    @test optimized.method isa RayleighRitzMethod
    @test optimized.initialbasisset[1].a == 1.0
    @test optimized.progress == false
    @test !isempty(optimized.history)
    @test haskey(optimized, :optimizer)
    @test haskey(optimized, :E)
    @test !haskey(optimized, :geometricbasisset)
    @test optimized.E[1] ≤ solve(H, SimpleGaussianBasis(1.0)).E[1] + 1e-10
    @test TwoBody.ψ(optimized, 0.5) == TwoBody.ψ(optimized.result, 0.5)
    @test TwoBody.expectation(optimized, H) ≈ optimized.E
    @test TwoBody.verification(optimized).norm[1] ≈ 1 atol=1e-8
    optimized_text = string(optimized)
    @test sprint(show, optimized) == optimized_text
    for header in ["# method", "# optimizer", "# initial basis function", "# optimized basis function", "# eigenvalue", "# verification"]
      @test occursin(header, optimized_text)
    end
    @test !occursin("# optimization log", optimized_text)

    optimized_method = optimize(H, RayleighRitzMethod(SimpleGaussianBasis(1.0)); progress=false, iterations=5)
    @test optimized_method isa ResultOptimizationRayleighRitz
    @test optimized_method.E ≈ optimized.E

    @test_throws ArgumentError optimize(H, SimpleGaussianBasis(1.0); method=TwoBody.Optim.NelderMead())

    optimized_minimal = optimize(H, SimpleGaussianBasis(1.0); info=-1, iterations=5)
    @test optimized_minimal isa ResultOptimizationRayleighRitz
    @test optimized_minimal.E[1] ≈ optimized.E[1]
    @test !haskey(optimized_minimal, :C)
    @test isempty(optimized_minimal.history)
    @test startswith(string(optimized_minimal), "ResultOptimizationRayleighRitz:")

    optimized_geometric = optimize(H, geometric; iterations=3)
    @test optimized_geometric isa ResultOptimizationRayleighRitz
    @test optimized_geometric.initialgeometricbasisset === geometric
    @test optimized_geometric.initialbasisset isa BasisSet
    @test haskey(optimized_geometric, :geometricbasisset)
    @test optimized_geometric.E[1] ≤ geometric_result.E[1] + 1e-10
    optimized_geometric_text = string(optimized_geometric)
    for header in ["# optimizer", "# optimization log", "# initial geometric progression", "# optimized geometric progression", "# eigenvalue"]
      @test occursin(header, optimized_geometric_text)
    end
  end
end
