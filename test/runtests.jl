using TwoBody
using Test
using QuadGK
using Printf
using Antique
using SpecialFunctions
using ForwardDiff

@testset verbose = true "TwoBody.jl" begin
  include("Basis.jl")
  include("Rayleigh-Ritz.jl")
  include("FDM.jl")
  include("VMC.jl")
end
