using TwoBody
using Documenter

DocMeta.setdocmeta!(TwoBody, :DocTestSetup, :(using TwoBody); recursive=true)

makedocs(;
    modules=[TwoBody],
    authors="Shuhei Ohno",
    repo="https://github.com/ohno/TwoBody.jl/blob/{commit}{path}#{line}",
    sitename="TwoBody.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ohno.github.io/TwoBody.jl",
        assets=String[
            "./assets/logo.ico",
        ],
    ),
    pages=[
        "Home" => "index.md",
        "Hamiltonian" => "Hamiltonian.md",
        "Database" => "DB.md",
        "Rayleigh-Ritz Method" => "Rayleigh-Ritz.md",
        "Finite Difference Method" => "FDM.md",
        "Variational Monte Carlo" => "VMC.md",
        "API reference" => "API.md",
    ],
)

deploydocs(;
    repo="github.com/ohno/TwoBody.jl",
)
