@testset "Database" begin
    hydrogen = TwoBody.db(:hydrogen)

    @test hydrogen isa TwoBody.DatabaseEntry{Float64}
    @test hydrogen.hamiltonian isa Hamiltonian
    @test hydrogen.energy == -0.5
    @test TwoBody.db("hydrogen").energy == hydrogen.energy
    @test TwoBody.dbkeys() == [:harmonic_oscillator, :hydrogen, :positronium]
    @test_throws ArgumentError TwoBody.db(:unknown)

    positronium = TwoBody.db(:positronium)
    @test positronium.energy == -0.25
    @test positronium.hamiltonian[1].m == 0.5

    # A caller may modify its Hamiltonian without corrupting the registry.
    pop!(hydrogen.hamiltonian.terms)
    @test length(hydrogen.hamiltonian) == 1
    @test length(TwoBody.db(:hydrogen).hamiltonian) == 2

    key = :temporary_test_problem
    hamiltonian = Hamiltonian(
        Kinetic(ℏ = 1.0, m = 1.0),
        Coulomb(coefficient = -2.0),
    )

    try
        registered = TwoBody.put!(key, hamiltonian, -2.0)
        @test registered isa TwoBody.DatabaseEntry{Float64}
        @test TwoBody.db(key).energy == -2.0
        @test_throws ArgumentError TwoBody.put!(key, hamiltonian, -2.0)

        # Registration also isolates the stored Hamiltonian from the caller.
        pop!(hamiltonian.terms)
        @test length(TwoBody.db(key).hamiltonian) == 2
    finally
        delete!(TwoBody._DATABASE, key)
    end
end
