# dealii-internodes

A standalone [deal.II](https://www.dealii.org/) implementation of the
**INTERNODES** method for coupling PDE solutions across non-conforming
subdomain interfaces (discretization and/or geometric non-conformity), using
Lagrange interpolation for conforming interfaces and rescaled localized
radial-basis-function (RL-RBF) interpolation otherwise. The coupled problem is
solved through a preconditioned, matrix-free Schur-complement strategy
(GMRES on the interface problem, Dirichlet–Neumann-type preconditioner).

This repository is a `deal.II`-only port of the implementation described in
*[paper title/venue — to be added]*, originally built on the
[`lifex`](https://lifex.gitlab.io/) framework. It has no `lifex` dependency.

## Dependencies

- **CMake** ≥ 3.13 and a C++17 compiler
- **MPI** (OpenMPI or MPICH)
- **[deal.II](https://www.dealii.org/)** ≥ 9.5 (developed and tested with
  9.7.0), configured with MPI, **Trilinos** (`TrilinosWrappers` linear algebra
  and the ML/MueLu algebraic multigrid) and **p4est** (hexahedral meshes).
  METIS is *not* required.

The easiest way to get a correctly configured `deal.II` is
[`candi`](https://github.com/dealii/candi), which builds `deal.II` together
with Trilinos and p4est from source:

```bash
git clone https://github.com/dealii/candi.git
cd candi
./candi.sh --packages="p4est trilinos dealii"
```

On a cluster with a module system, load (or build) `deal.II` with the same
components and set `DEAL_II_DIR` as below.

## Building

```bash
mkdir build && cd build
cmake -DDEAL_II_DIR=/path/to/dealii/install ..
make -j<N>
```

If `deal.II` is installed in a standard location, `cmake ..` alone should find
it. The configure step stops with a clear message if `deal.II` lacks Trilinos
or p4est.

## Running the example

`examples/coupled_diffusion` solves the paper's manufactured-solution test,
$-\Delta u + u = f$ with $u = e^{x+y+z}$ on $\Omega = (-2,2)\times(-1,1)^2$,
split at $x = 0$ into a *master* and a *slave* subdomain, and prints the
number of interface-GMRES iterations and the broken $H^1$ error against the
exact solution. Everything is set through a parameter file.

```bash
cd build/examples/coupled_diffusion
mpirun -np 2 ./coupled_diffusion coupled_diffusion.prm      # about 1 s
```

The default file is Geometry-A (two adjacent boxes, hexahedral meshes,
Lagrange interpolation). Copy it and change a few lines to explore the other
cases:

| Case | Parameters to set |
|---|---|
| Non-matching meshes across the interface | `Subdivisions master`, `Subdivisions slave` different (e.g. `8,4,4` vs `4,2,2`) and `RBF radius` > 0 (or `= 0` for Lagrange interpolation) |
| Different polynomial degrees | `Master degree`, `Slave degree` |
| Geometry-B (curved shell interface, tetrahedra) | `Type = half_hyper_shells`, `RBF radius` > 0, `Shell refinement master/slave` |
| Dirichlet / Neumann / interface assignment | `Dirichlet ids …`, `Neumann ids …`, `Interface id …` per subdomain, as lists of boundary ids |

Boundary ids are those of the mesh you build: for the boxes the
`colorize = true` ids of `GridGenerator::subdivided_hyper_rectangle`
(0/1 = −x/+x, 2/3 = −y/+y, 4/5 = −z/+z), for the shells those of
`GridGenerator::half_hyper_shell` (0 = inner surface, 1 = outer surface,
2 = flat cut). Any id, including 0, may be used on hexahedral and on
tetrahedral meshes.

### Verified results

All numbers below were produced by this code (deal.II 9.7.0, Trilinos 16.2.0,
p4est 2.8.7, OpenMPI). Results are independent of the number of MPI ranks
(checked on 1, 2 and 4 ranks; identical to the printed digits).

Geometry-A, conforming interface (`RBF radius = 0`), broken $H^1$ error:

| Mesh (per subdomain) | $\mathbb{P}_1$ | ratio | $\mathbb{P}_2$ | ratio |
|---|---|---|---|---|
| `4,2,2`    | 9.48113 |      | 0.989073  |      |
| `8,4,4`    | 4.27465 | 2.22 | 0.251874  | 3.93 |
| `16,8,8`   | 2.07294 | 2.06 | 0.0632763 | 3.98 |
| `32,16,16` | 1.02826 | 2.02 |           |      |

i.e. the optimal rates 1 and 2 under refinement.

Geometry-A, geometrically matching but discretization-non-conforming
interface (master mesh twice as fine as the slave's), $\mathbb{P}_1$:

| Master / slave subdivisions | Lagrange: error (GMRES its) | RL-RBF: error (GMRES its) |
|---|---|---|
| `8,4,4` / `4,2,2`      | 9.42269 (7) | 9.42358 (6) |
| `16,8,8` / `8,4,4`     | 4.24635 (7) | 4.24654 (7) |
| `32,16,16` / `16,8,8`  | 2.05898 (7) | 2.05902 (7) |

The RL-RBF support radii used were 2.0, 1.0 and 0.5 for the three rows, i.e.
proportional to the interface mesh size. The interface iteration count does
not grow under refinement, and both interpolation choices reach the same
discretization-limited accuracy.

Geometry-B (two half shells, tetrahedral, RL-RBF), $\mathbb{P}_1$:

| Shell refinement master / slave | error | GMRES its |
|---|---|---|
| 1 / 1 | 0.812484 | 9  |
| 2 / 2 | 0.421800 | 9  |
| 2 / 1 (non-matching) | 0.845792 | 14 |

## Reproducing the paper's full results

`reproduce/` contains parameter files for the configurations of the paper's
accuracy and strong-scaling studies (Table 4). **These are not meant to be
run by a reviewer or casual user**: the scalability configurations need
distributed-memory HPC resources (the paper's runs used up to 500+ MPI
processes and ~10 million degrees of freedom). See `reproduce/README.md` for
what each configuration requires and how it maps to the paper.

Two things differ from the paper's setup, so figures will not match to the
last digit:

- the paper's Geometry-A meshes were generated with `lifex`, and its
  Geometry-B tetrahedral meshes with Gmsh. Here both are generated by deal.II
  (`GridGenerator`), the shells by converting a hexahedral half shell to
  tetrahedra;
- the mesh sizes in `reproduce/*.prm` are placeholders to be adjusted to the
  degrees-of-freedom counts of the paper's Table 4.

## Notes on the port

- **Meshes.** `MeshHandler` distributes a serial `Triangulation` built by the
  user: hexahedral meshes with `parallel::distributed::Triangulation`
  (p4est), tetrahedral meshes with `parallel::fullydistributed::Triangulation`
  (z-order partitioning, no METIS). Matching `FE_Q` / `FE_SimplexP`,
  quadratures and mappings are chosen from the cell type. For tetrahedral
  meshes the boundary ids are shifted by one internally, because a
  fully-distributed triangulation labels partition boundaries with the default
  id 0; this is transparent to the user.
- **Interface mass matrix inverse.** $M_{\Gamma_2}^{-1}$ in the residual
  transfer is applied with Jacobi-preconditioned CG. The diagonally scaled
  mass matrix is well conditioned independently of the mesh, and this proved
  more robust than algebraic multigrid with a direct coarse solver on small
  parallel interface matrices.
- **Inner tolerances.** The subdomain solves use CG tolerances of
  $10^{-13}$ (absolute) / $10^{-11}$ (relative), as in the original code.
  These are much tighter than the outer GMRES tolerance ($10^{-8}$) and can
  probably be relaxed for speed.
- **Naming.** Class names follow the original code (`SubProblemBase`,
  `SubProblemDiffusionReaction`, `InterfaceDoFHandler`,
  `InterfaceDoFHandlerRBF`, `MultiDomainProblem`,
  `InternodesSchurComplement`, `SchurPreconditioner`); the PDE model is a
  subclass of `SubProblemBase` overriding `localBilinearForm()`.

## Repository layout

```
include/internodes/   Headers (SubProblemBase and PDE subclass, MultiDomainProblem,
                      InternodesSchurComplement, SchurPreconditioner, interface
                      DoF handlers (Lagrange, RBF), MeshHandler, R-tree search)
source/               Corresponding implementation files
examples/             Small, reviewer-runnable example (coupled_diffusion)
reproduce/            Paper-reproduction configurations (HPC-scale)
```

## Status

The hexahedral and tetrahedral code paths, serial and parallel, are validated
as described above. The HPC-scale configurations in `reproduce/` have not yet
been run at full size in this port. Only the diffusion–reaction model is
provided; other PDEs need a new `SubProblemBase` subclass.

## License

MIT — see `LICENSE`.

## Citation

If you use this code, please cite:

```
[BibTeX entry — to be added once the paper is published]
```
