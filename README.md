# TwoBody.jl [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliafewbody.github.io/TwoBody.jl/stable) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliafewbody.github.io/TwoBody.jl/dev) [![Build Status](https://github.com/JuliaFewBody/TwoBody.jl/workflows/CI/badge.svg)](https://github.com/JuliaFewBody/TwoBody.jl/actions)

TwoBody.jl: a Julia package for quantum mechanical two-body problems

## Documentation 

https://juliafewbody.github.io/TwoBody.jl/dev/

## Dependency

```mermaid
---
config:
  layout: elk
  theme: mc
---
flowchart TD
  A["Hamiltonian.jl"]
  C["Rayleigh-Ritz.jl"]
  F["FDM.jl"]
  N["VNN.jl"]
  G["VMC.jl"]
  H["DB.jl"]
  Z["TwoBody.jl"]
  A --> H
  A --> C & F & N & G
  H --> C & F & N & G
  F --> N
  C & F & N & G --> Z
```

## Developer's Guide

There are several tools for developers.

```sh
git clone https://github.com/JuliaFewBody/TwoBody.jl.git
cd TwoBody.jl
julia
julia> include("dev/revice.jl")
julia> include("dev/test.jl")
julia> include("dev/docs.jl")
```
