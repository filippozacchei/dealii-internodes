# Reproducing the paper's results

These configurations correspond to the five scalability tests in Table 4 of
the paper, plus the accuracy tests of Sect. 4.1. **They are not meant to be
run by a reviewer.** The original runs used up to 500+ MPI processes on
CINECA's Galileo100 cluster (dual-socket Intel Xeon Platinum 8276/8276L,
48 cores/node) with problems up to ~17 million degrees of freedom; Test 1
alone needs a machine with a lot of memory and a lot of patience to run
serially. This directory exists for transparency and for anyone with
comparable HPC access, not as a "click to verify" step.

## What's exactly reproducible vs. approximate

For each test, the **polynomial degrees** ($p_1$, $p_2$), the **geometry
type** (Geometry-A / Geometry-B), and the **RL-RBF support radius**
($r = r_f h_2$, using the paper's own scaling factor $r_f$ and slave mesh
size $h_2$) are taken directly from Table 4 and are exact.

The **exact meshes** are not, and the two geometries differ in kind, not
just in degree of approximation:

- **Geometry-A** (Tests 1-3): structured **hexahedral**, generated
  internally by `lifex`'s own mesh-handling utilities. `build_adjacent_boxes()`
  reproduces this with plain `GridGenerator` calls -- the cell type matches
  exactly, but the exact per-direction subdivision counts that produce the
  reported $h_1$/$h_2$ and DoF counts aren't reconstructable from Table 4's
  summary statistics alone, so `Subdivisions master`/`Subdivisions slave`
  below are placeholders to tune upward until you hit the target DoF count.
- **Geometry-B** (Tests 4-5): **tetrahedral**, generated externally with
  Gmsh -- a genuinely different cell type, needed for a well-shaped mesh on
  curved geometry. We don't have the original Gmsh files, so
  `build_half_hyper_shells()` instead builds a hexahedral half-shell with
  `GridGenerator::half_hyper_shell()` and converts it to simplex cells via
  `GridGenerator::convert_hypercube_to_simplex_mesh()`. This matches the
  paper's *cell type* (tetrahedral, exercising the same `FE_SimplexP`/
  `QGaussSimplex`/`MappingFE` and fully-distributed-triangulation code path
  that a Gmsh-imported tetrahedral mesh would use via `MeshHandler::create()`)
  but not the exact mesh Gmsh would produce for the same geometry. Reading
  Gmsh files is not wired into the example driver.

## Configurations

| File | Test | Geometry | $p_1$ | $p_2$ | Interpolation | $r_f$ |
|---|---|---|---|---|---|---|
| `test1.prm` | 1 | A | 1 | 1 | Lagrange | -- |
| `test2.prm` | 2 | A | 1 | 1 | RL-RBF | 1 |
| `test3.prm` | 3 | A | 2 | 4 | RL-RBF | 1 |
| `test4.prm` | 4 | B | 1 | 1 | RL-RBF | 1 |
| `test5.prm` | 5 | B | 2 | 2 | RL-RBF | 10 |

## Running

Same binary as the small-mesh example, just pointed at one of these files
and (for anything beyond a quick sanity check) launched with many more MPI
processes and a much finer mesh than the defaults:

```bash
mpirun -np <N> ./coupled_diffusion reproduce/test1.prm
```

## MPI process counts for the strong-scaling sweep

The paper reports strong scaling "up to more than 500 MPI processes" with a
reference count $p_0=48$, but the exact sequence of process counts used
between those two points isn't stated as a list in the text -- only shown
in the scaling figures. If you want to reproduce the exact scaling curve
(not just one configuration), you'll need to either read the process counts
off Figs. 9-10 of the paper directly, or ask the authors for the raw
timing data.
