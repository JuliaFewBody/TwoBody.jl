using TwoBody
using Documenter

DocMeta.setdocmeta!(TwoBody, :DocTestSetup, :(using TwoBody); recursive=true)

local_build = get(ENV, "DOCUMENTER_LOCAL", "false") == "true"
gem_only = get(ENV, "DOCUMENTER_GEM_ONLY", "false") == "true"
remote_options = local_build ?
    (repo = "", remotes = nothing) :
    (repo = "https://github.com/JuliaFewBody/TwoBody.jl/blob/{commit}{path}#{line}",)

pages = gem_only ?
    ["Gaussian Expansion Method" => "GEM.md"] :
    [
        "Home" => "index.md",
        "Hamiltonian" => "Hamiltonian.md",
        "Database" => "DB.md",
        "Rayleigh-Ritz Method" => "Rayleigh-Ritz.md",
        "Gaussian Expansion Method" => "GEM.md",
        "Finite Difference Method" => "FDM.md",
        "Variational Monte Carlo" => "VMC.md",
        "API reference" => "API.md",
    ]

makedocs(;
    modules=[TwoBody],
    authors="Shuhei Ohno",
    sitename="TwoBody.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://juliafewbody.github.io/TwoBody.jl/",
        edit_link=local_build ? nothing : :commit,
        assets=String[
            "./assets/logo.ico",
        ],
    ),
    pages=pages,
    pagesonly=gem_only,
    checkdocs=gem_only ? :none : :all,
    remote_options...,
)

if !local_build
    deploydocs(;
        repo="github.com/JuliaFewBody/TwoBody.jl",
        push_preview=true,
    )
end
