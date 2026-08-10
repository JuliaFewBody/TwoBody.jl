@testset "Hamiltonian.jl" begin
  @test Kinetic() isa Operator

  potentials = (
    Constant(),
    Linear(),
    Coulomb(),
    PowerLaw(),
    Gaussian(),
    Exponential(),
    Yukawa(),
    Delta(),
    Custom(identity),
    Tabulated(range(1.0, step=1.0, length=3), Number[1, 2, 3]),
  )
  @test all(p -> p isa TwoBody.PotentialTerm, potentials)
  @test TwoBody.V(Custom(r -> r^2), 2) == 4
end
