# dealii-internodes

A standalone [deal.II](https://www.dealii.org/) implementation of the
**INTERNODES** method for coupling PDE solutions across non-conforming
subdomain interfaces (discretization and/or geometric non-conformity), using
Lagrange interpolation for geometrically conforming interfaces and rescaled
localized radial-basis-function (RL-RBF) interpolation otherwise. The coupled
problem is solved via a preconditioned, matrix-free Schur-complement strategy.

This repository is a `deal.II`-only port of the implementation described in
*[paper title/venue — to be added]*, originally built on the
[`lifex`](https://lifex.gitlab.io/) framework. It has no `lifex` dependency.

## Dependencies

- **CMake** ≥ 3.13
- **MPI** (e.g. OpenMPI or MPICH)
- **[deal.II](https://www.dealii.org/)**, latest release, configured with:
  - MPI support
  - **Trilinos** (for `TrilinosWrappers` distributed linear algebra and the
    `ML`/`PreconditionAMG` algebraic multigrid preconditioner)
  - **p4est** (for `parallel::distributed::Triangulation`)

The easiest way to get a correctly configured `deal.II` is
[`candi`](https://github.com/dealii/candi), which builds `deal.II` and all of
the above dependencies from source:

```bash
git clone https://github.com/dealii/candi.git
cd candi
./candi.sh --packages="p4est trilinos dealii"
```

Alternatively, on a cluster with a module system, load (or build) `deal.II`,
`p4est`, and `Trilinos` modules matching the versions `deal.II` was configured
against, and set `DEAL_II_DIR` accordingly (see below).

## Building

```bash
mkdir build && cd build
cmake -DDEAL_II_DIR=/path/to/dealii/install ..
make -j<N>
```

If `deal.II` is installed in a standard location (e.g. via `candi`'s default
prefix, or a system package), `cmake ..` alone should find it automatically.

## Running the small-mesh example

`examples/coupled_diffusion` reproduces the paper's basic accuracy check: two
adjacent subdomains with a non-conforming interface, coarse enough to run
interactively on a laptop. It compares the coupled two-subdomain solution
against a monolithic single-domain reference, the same check used in the
paper.

```bash
cd build/examples/coupled_diffusion
mpirun -np 2 ./coupled_diffusion parameters.prm
```

Expected output: convergence of the broken $H^1$-error at the optimal rate
under mesh refinement, and a reported error against the monolithic reference
consistent with discretization error alone (no additional error introduced by
the interface coupling).

## Reproducing the paper's full results

`reproduce/` contains the parameter files for the exact configurations used
in the paper's accuracy and strong-scaling studies (Table with test
configurations, both geometries). **These are not meant to be run by a
reviewer or casual user** — the scalability configurations require
distributed-memory HPC resources (the paper's runs used up to 500+ MPI
processes on a cluster with ~10 million degrees of freedom). See
`reproduce/README.md` for details on each configuration and the resources it
requires.

## Repository layout

```
include/internodes/   Header files (SubProblem, MultiDomainProblem,
                       InternodesSchurComplement, SchurPreconditioner,
                       interface DoF handlers, R-tree neighbor search)
source/                Corresponding implementation files
examples/               Small, reviewer-runnable example(s)
reproduce/              Full paper-reproduction configurations (HPC-scale)
```

## Status

This repository is under active development — it is a port in progress from
an internal `lifex`-based implementation. See `CHANGELOG.md` (once added) for
progress.

## License

MIT — see `LICENSE`.

## Citation

If you use this code, please cite:

```
[BibTeX entry — to be added once the paper is published]
```
