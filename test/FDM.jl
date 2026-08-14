@testset "FDM.jl" begin

  # Testing Results

  H = Hamiltonian(
    Kinetic(hbar = 1, m = 1),
    Coulomb(coefficient = -1),
  )
  FDM = FiniteDifferenceMethod(
    Δr = 0.1,
    rₘₐₓ = 50.0,
    l = 0,
    direction = :c,
    solver = :LinearAlgebra,
  )
  res = solve(H, FDM, info=4)

  @test res isa ResultFiniteDifferenceMethod
  @test occursin("# eigenvalue", string(res))
  @test sprint(show, res) == string(res)

  compact = solve(H, FDM, info=-1)
  @test compact isa ResultFiniteDifferenceMethod
  @test compact.E == res.E
  @test string(compact) == "ResultFiniteDifferenceMethod(E=$(compact.E))"

  trial = solve(H, r -> exp(-r), FDM, 0)
  @test trial isa ResultFiniteDifferenceMethod
  @test string(trial) == "ResultFiniteDifferenceMethod(E=$(trial.E))"
  
  println("<ψₙ|ψₙ> = cₙ' * cₙ = 1")
  println("  i\tnumerical  \tanalytical")
  for i in 1:res.nₘₐₓ
    numerical  = res.C[:,i]' * res.C[:,i]
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
  
  # comparison with Antique.jl
  HA = Antique.HydrogenAtom(Z=1, mₑ=1.0, a₀=1.0, Eₕ=1.0, ℏ=1.0)

  println("Energy")
  println("  i\tnumerical  \tanalytical")
  for i in 1:res.nₘₐₓ
    numerical  = res.E[i]
    analytical = Antique.energy(HA; n=i)
    error = iszero(analytical) ? abs(numerical-analytical) : abs((numerical-analytical)/analytical)
    acceptance = error < 1e-2
    @printf("%3d\t%.9f\t%.9f\t%s\n", i, numerical, analytical, acceptance ? "✔" :  "✗")
    @test acceptance
  end

  println("Wave Function")
  println("  i\t  r\tnumerical  \tanalytical")
  for n in 1:res.nₘₐₓ
    @show n
    for i in keys(res.method.R[begin:min(10,length(res.method.R))])
        r = res.method.R[i]
        numerical  = abs(res.ψ[i,n])
        analytical = abs(Antique.wavefunction(HA, r, 0, 0; n=n))
        error = iszero(analytical) ? abs(numerical-analytical) : abs((numerical-analytical)/analytical)
        acceptance = error < 5e-2
        @printf("%3d\t%.1f\t%.9f\t%.9f\t%s\n", i, r, numerical, analytical, acceptance ? "✔" :  "✗")
        @test acceptance
    end
  end

end
