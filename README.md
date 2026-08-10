# TwoBody.jl [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ohno.github.io/TwoBody.jl/stable) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ohno.github.io/TwoBody.jl/dev) [![Build Status](https://github.com/ohno/TwoBody.jl/workflows/CI/badge.svg)](https://github.com/ohno/TwoBody.jl/actions)

TwoBody.jl: a Julia package for quantum mechanical two-body problems

## Documentation 

https://ohno.github.io/TwoBody.jl/dev/

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
  G["VMC.jl"]
  H["DB.jl"]
  Z["TwoBody.jl"]
  A --> H
  A --> C & F & G
  H --> C & F & G
  C & F & G --> Z
```

## Developer's Guide

There are several tools for developers.

```sh
git clone https://github.com/ohno/TwoBody.jl.git
cd TwoBody.jl
julia
julia> include("dev/revice.jl")
julia> include("dev/test.jl")
julia> include("dev/docs.jl")
```
