@testset "Rayleigh-Ritz.jl" begin

  # Testing Results

  H = Hamiltonian(
    NonRelativisticKinetic(ℏ = 1 , m = 1),
    CoulombPotential(coefficient = -1),
  )
  BS = BasisSet(
    SimpleGaussianBasis(13.00773),
    SimpleGaussianBasis(1.962079),
    SimpleGaussianBasis(0.444529),
    SimpleGaussianBasis(0.1219492),
  )
  res = solve(H, BS)
  
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

  # comparison with Antique.jl
  HA = Antique.HydrogenAtom(Z=1, m_e=1.0, a_0=1.0, E_h=1.0, hbar=1.0)

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
    o = NonRelativisticKinetic(ℏ=1, m=1)
    println(o)
    println("  i  j\tnumerical  \tanalytical")
    l = 0
    for j in keys(BS.basis)
    for i in keys(BS.basis)
      φᵢ(r) = TwoBody.φ(BS.basis[i], r)
      φⱼ(r) = TwoBody.φ(BS.basis[j], r)
      dφⱼ(r) = ForwardDiff.derivative(r -> φⱼ(r), r)
      d²φⱼ(r) = ForwardDiff.derivative(r -> dφⱼ(r), r)
      numerical  = 4*π*QuadGK.quadgk(r -> r^2 * φᵢ(r) * (-o.ℏ^2/2/o.m * (d²φⱼ(r) + 2/r*dφⱼ(r) - l*(l+1)/r^2*φⱼ(r))), 0, Inf, maxevals=10^3)[1]
      analytical = TwoBody.element(o, BS.basis[i], BS.basis[j])
      error = isinf(analytical) ? 0.0 : abs((numerical-analytical)/analytical)
      acceptance = error < 1e-5
      @printf("%3d%3d\t%.9f\t%.9f\t%s\n", i, j, numerical, analytical, acceptance ? "✔" :  "✗")
      @test acceptance
    end
    end

    # potential terms
    for o in [
      ConstantPotential(),
      LinearPotential(),
      CoulombPotential(),
      PowerLawPotential(),
      PowerLawPotential(exponent=2),
      GaussianPotential(),
      # ExponentialPotential(),
      # YukawaPotential(),
      # LogarithmicPotential(),
      # DeltaPotential(),
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

  end
end
