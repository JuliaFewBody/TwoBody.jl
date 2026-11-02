# [Developer Guide](@id developer-guide)

If you are planning significant changes, please open an [issue](https://github.com/JuliaFewBody/TwoBody.jl/issues) first. The [ColPrac](https://github.com/SciML/ColPrac) guidelines are recommended. For Julia package development basics, see:

- [How to develop a Julia package](https://julialang.org/contribute/developing_package/)
- [Pkg: Creating packages](https://pkgdocs.julialang.org/v1/creating-packages/)

## Local Setup

This procedure is required only once. Install Git and Julia on your local machine before starting.

1. Fork [the repository](https://github.com/JuliaFewBody/TwoBody.jl) on GitHub.
2. Clone the forked repository. Replace `xxxxxx` with your GitHub username.

   ```sh
   git clone https://github.com/xxxxxx/TwoBody.jl.git
   cd TwoBody.jl
   ```

3. Install [Revise.jl](https://github.com/timholy/Revise.jl).

   ```sh
   julia --startup-file=no -e 'import Pkg; Pkg.add("Revise")'
   ```

## Development Flow

This is the typical workflow for making changes.

1. Create a branch for your changes. Replace `xxx` with the issue number, for example `issue/20`.

   ```sh
   git switch -c issue/xxx
   ```

2. Start an interactive session with [Revise.jl](https://github.com/timholy/Revise.jl).

   ```sh
   julia --startup-file=no -i -e 'using Revise; import Pkg; Pkg.activate("."); using TwoBody'
   ```

3. Change the source code. When adding functions or updating docstrings, refer to [Documenter: Adding docstrings](https://documenter.juliadocs.org/stable/man/guide/#Adding-Some-Docstrings).
4. If you need a new dependency, replace `SomePackage` with its package name and run:

   ```sh
   julia --project=. --startup-file=no -e 'import Pkg; Pkg.add("SomePackage"); Pkg.resolve(); Pkg.instantiate()'
   ```

5. Run the tests. They may take a few minutes.

   ```sh
   julia --project=. --startup-file=no -e 'using Pkg; Pkg.test()'
   ```

6. Build the documentation locally. HTML files are generated in `docs/build/`; open `docs/build/index.html` in a web browser to review them.

   ```sh
   julia --project=docs --startup-file=no -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
   julia --project=docs --startup-file=no -e 'include("docs/make.jl")'
   ```

7. After the tests and documentation build succeed, commit and push the changed files.

   ```sh
   git add "path/to/changed/file"
   git commit -m "commit message"
   git push origin issue/xxx
   ```

8. Submit a pull request on GitHub.

## GitHub Actions

CI tests Julia 1.10 and 1.12, checks installation without optional solver dependencies,
and builds the documentation. The Julia 1.10 test job collects coverage for `src` and
`ext`, converts it to `lcov.info`, and uploads it to Codecov. Repository builds other
than Dependabot runs authenticate using GitHub OIDC; no `CODECOV_TOKEN` secret is
needed. Upload errors fail these jobs so that a broken integration is visible.
Fork and Dependabot runs attempt tokenless uploads without OIDC; failures in the
upload step do not fail their test jobs. Test and coverage-generation failures still
fail CI. Codecov's action handles public fork PRs using prefixed branch names, so
organization-wide token authentication can remain enabled; see
[Codecov's token documentation](https://docs.codecov.com/docs/codecov-tokens#commits-on-unprotected-branches).

To check formatting, run **Runic formatting** from the repository's Actions tab and
select the branch to check. The workflow reports formatting differences as a failed
check and saves a `runic-formatting-patch` artifact only when the patch is nonempty.
Download and extract the artifact,
then apply `runic.patch` with `git apply runic.patch` in a checkout of the same commit.
Review and commit the changes as usual. Formatting runs only on manual dispatch.

CompatHelper runs daily and can also be dispatched manually. In **Settings > Actions >
General**, enable **Allow GitHub Actions to create and approve pull requests** so that
it can open dependency updates. If GitHub disables its schedule after repository
inactivity, re-enable the workflow in the Actions tab. The workflow uses
`DOCUMENTER_KEY` as `COMPATHELPER_PRIV` so its pull requests can trigger CI.

TagBot uses `GITHUB_TOKEN` to create releases and `DOCUMENTER_KEY` to push tags that
trigger documentation builds. The key must be configured as a deploy key with write
access. Releases for commits that modify workflow files may require manual creation
or a personal access token with workflow scope; see the
[TagBot troubleshooting guide](https://github.com/JuliaRegistries/TagBot#commits-that-modify-workflow-files).

## Adding New Operators and Solvers

TwoBody.jl uses Julia's multiple dispatch to keep the physical problem separate from its numerical solution.

### Operators

1. Define the operator in `src/Hamiltonian.jl` as a subtype of `KineticTerm` or `PotentialTerm`.
2. Implement the solver-specific operations required to support it, such as `element`, `matrix`, or local-energy evaluation.
3. Add tests to the corresponding files under `test/`.
4. Add or update the mathematical definition and API documentation under `docs/src/`.

An unsupported operator–solver combination should fail explicitly rather than silently choosing an approximation.

### Solvers

1. Create `src/MethodName.jl` and define the method type and its `solve(hamiltonian::Hamiltonian, method::MethodName; ...)` implementation.
2. Include the source file from `src/TwoBody.jl` after its dependencies.
3. Create `test/MethodName.jl` and include it from `test/runtests.jl`.
4. Create `docs/src/MethodName.md` and add it to the `pages` list in `docs/make.jl`.
5. Run the complete test suite and documentation build as described in [Development Flow](@ref).

## Versioning and Registering (for Maintainers)

This project follows [Semantic Versioning](https://semver.org/). When bumping the version, update `version` in [`Project.toml`](https://github.com/JuliaFewBody/TwoBody.jl/blob/main/Project.toml).

To register a release in the [General](https://github.com/JuliaRegistries/General) registry, use [Registrator](https://github.com/JuliaRegistries/Registrator.jl#via-the-github-app) through its GitHub App workflow.

## Architecture

`src/TwoBody.jl` defines the `TwoBody` module and includes the source files in dependency order. `Hamiltonian.jl` defines the shared problem representation. `Basis.jl` supports the Rayleigh–Ritz implementation, and `FDM.jl` supplies the discretization used by the variational neural-network method. The solver files extend `solve` for their respective method types.

```@raw html
<pre class="mermaid">
---
config:
  layout: elk
  theme: mc
---
flowchart TD
  H["Hamiltonian.jl"]
  D["DB.jl"]
  B["Basis.jl"]
  R["Rayleigh-Ritz.jl"]
  F["FDM.jl"]
  Q["QTT.jl"]
  N["VNN.jl"]
  V["VMC.jl"]
  T["TwoBody.jl"]

  H --> D
  H --> R &amp; F &amp; Q &amp; N &amp; V
  B --> R
  F --> N
  H &amp; D &amp; B &amp; R &amp; F &amp; Q &amp; N &amp; V --> T
</pre>
<script type="module">
  // Use the pinned ESM conversion on this page only. The dist bundle's FastDOM
  // dependency registers anonymous AMD modules and breaks Documenter's math loader.
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11.17.2/+esm';
  mermaid.initialize({ startOnLoad: false, theme: "neutral" });
  await mermaid.run({ querySelector: '.mermaid' });
</script>
```
