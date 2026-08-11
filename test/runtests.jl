using TwoBody
using Test
using QuadGK
using Printf
using Antique
using SpecialFunctions
using ForwardDiff

@testset verbose = true "TwoBody.jl" begin
  include("Hamiltonian.jl")
  include("DB.jl")
  include("Basis.jl")
  include("FC.jl")
  include("Rayleigh-Ritz.jl")
  include("GEM.jl")
  include("FDM.jl")
  include("VMC.jl")
end
