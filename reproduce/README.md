# Reproducing the paper's figures

Scripts that regenerate the numerical-results figures of the paper from this
code: run the configurations behind each figure, cache the results as JSON,
and draw the figures under the file names used in the paper (same axes,
series, markers and colours). **They are not meant to be run by a reviewer**:
the finest levels and the scalability tests need a workstation or an HPC
cluster (the paper's runs used up to 768 MPI processes on CINECA's Galileo100,
48 cores per node). A scaled-down version of everything runs on a laptop, as
a check that the scripts work.

## Setup

```bash
pip install -r reproduce/requirements.txt      # numpy, matplotlib (no LaTeX needed)
```

The example executable must be built (see the top-level README); the scripts
look for `build/examples/coupled_diffusion/coupled_diffusion`, or use
`--exe` / `INTERNODES_EXE`. MPI runs are started with `mpirun -np N`; set
`INTERNODES_LAUNCHER="srun -n {np}"` (for instance) to change that.

On a cluster with no existing deal.II install, see `CLUSTER_SETUP.md` for a
candi build recipe before running anything below.

## Figures

| Paper figure | File(s) | Script |
|---|---|---|
| Convergence, INTERNODES vs single domain ($p=1,2$) | `P1_convergence`, `P2_convergence` | `accuracy.py` |
| Lagrange vs RBF, $p_1=p_2=1$, $r_f=1$ | `P1P1_slave_convergence_Lagrange_vs_RBF` | `accuracy.py` |
| Lagrange vs RBF, $p_1=2$, $p_2=3$, $r_f=1,2,5,10$ | `P2P3_convergence_Lagrange_vs_RBF` | `accuracy.py` |
| CPU time of the RBF interpolation vs radius | `RBF_radius_vs_time_hslave`, `..._gray` | `accuracy.py` |
| GMRES iterations with and without preconditioner | `GMRES_iterations_precond_vs_unprecond` | `accuracy.py` |
| Hexahedral–tetrahedral couplings | `P1P2_slave_convergence`, `P1P1_P2P2_slave_convergence` | `accuracy.py` |
| Strong scalability, Tests 1–5 (speed-up and time) | `<test>_scalability`, `<test>_total` | `scalability.py` |

The mesh pictures of the paper (Geometry-A/B, the partitions) are ParaView
screenshots and are not produced here.

## Accuracy figures

```bash
cd reproduce
python accuracy.py all --np 4 --max-level 4      # laptop: levels up to 4, about an hour
python accuracy.py run --max-level 6 --np 48     # the paper's levels (workstation / cluster)
python accuracy.py plot --formats pdf,png,eps    # figures/accuracy/
```

`run` caches each configuration in `results/accuracy/<case>.json` (with its
`.prm` and `.log`), so an interrupted run restarts where it stopped; runs that
fail or exceed `--timeout` are recorded and skipped on the next call
(`--retry-failed` to retry). `--figures conforming,p1p1,p2p3,gmres,hybrid`
selects a subset.

*Geometry-A* is two cubes of side 2, $(-2,0)\times(-1,1)^2$ and
$(0,2)\times(-1,1)^2$, each a single coarse cell refined $k$ times, so
$h=2/2^k$. Conforming tests use $k=0,\dots,6$; the non-conforming ones use
master level $k=2,\dots,6$ and slave level $k-1$ ($h_2=2h_1$). The GMRES
tolerance is a relative reduction of $10^{-8}$, as in the paper. The RBF
radius is $r=r_f h_{avg}$ with $h_{avg}$ the average cell diameter of each
subdomain's own mesh.

Assumptions made where the paper does not spell out the setup (please check
them against your own runs):

- The non-conforming figures plot the *master* mesh size $h_1=2/2^k$
  ($=1/2,\dots,1/32$), which is the paper's tick range, so the x axis is
  labelled $h_1$; `--paper-labels` labels it $h_2$ as in the paper. With these
  meshes the errors agree with the paper's plots (for example $1.06\cdot10^{-1}$,
  $1.6\cdot10^{-2}$, $3.0\cdot10^{-3}$ for Lagrange P2/P3, and $2.1\cdot10^{-2}$,
  $7.1\cdot10^{-3}$ for $r_f=1$ at the second and third points).
- The GMRES study uses $p=2$ on both sides with Lagrange interpolation and the
  same master levels ($9^3,\dots,129^3$ master DoFs).
- The CPU-time bars are built from the P2/P3 RBF runs: wall time (maximum over
  ranks) of the RBF interpolations and normal-derivative evaluations, summed
  over the whole solve.
- The hybrid figures use tetrahedra obtained by splitting the hexahedra of the
  same refined box (`Cell type master/slave = tet`), not Gmsh meshes, so their
  mesh sizes are $2/2^k$ instead of the paper's; radii are $r_f=1$ for
  $p=1$ and $r_f=5$ for $p=2$.

## Scalability figures

The configuration of each test is in `test1.prm` … `test5.prm` (Table 4), with
the refinement levels of the paper's scaling figures (`p1r8-p1r7`, `p2r7-p4r5`,
…). Core counts default to the paper's $48, 96, 192, 240, 288, 384, 480, 768$.

```bash
# on the cluster (`prepare` and `run` need nothing but a Python 3; only `plot`
# needs numpy/matplotlib, so plot on a laptop after copying the .json files back).
# Submit from a fresh login shell: `env | grep SLURM` should print nothing
# (a leftover SLURM_CPUS_PER_TASK from an earlier salloc breaks srun).
# --modules: the line(s) that load the compiler/MPI/libraries deal.II was built
# with, e.g. --modules "$(grep '^module load' ~/.bashrc)".

# 1. smoke test (small and fast; --tag keeps it apart from the real sweep's
#    files). Levels lowered by 3 keep every mesh non-degenerate on 96 ranks.
python3 scalability.py prepare --tests 1,2,3 --cores 48,96 --levels-down 3 --tag smoke \
       --account <account> --partition <partition> \
       --modules "module load <mpi> <libraries>" --time-limit 00:20:00
bash results/scalability_smoke/submit_all.sh
# check results/scalability_smoke/test{1,2,3}_np{48,96}.json exist and parse

# 2. the real sweep (Table 4 sizes), smallest core count first to learn the
#    run time and memory (sacct -j <id> --format=JobID,Elapsed,MaxRSS,State)
python3 scalability.py prepare --tests 1,2,3 --cores 48 --account <account> \
       --partition <partition> --modules "module load <mpi> <libraries>" --time-limit 08:00:00
bash results/scalability/submit_all.sh
# ... then the remaining core counts (write new scripts, resubmit):
python3 scalability.py prepare --tests 1,2,3 --cores 96,192,240,288,384,480,768 \
       --account <account> --partition <partition> \
       --modules "module load <mpi> <libraries>" --time-limit 04:00:00
bash results/scalability/submit_all.sh

# 3. bring the results back through a git branch (see "Getting results back"
#    below), then plot on a laptop
python scalability.py plot

# laptop check of the pipeline, everything scaled down (timings are meaningless)
python scalability.py run --tests 1,2 --cores 1,2,4 --levels-down 4
python scalability.py plot --tests 1,2
```

### Getting results back

The result files are small (JSON, parameter files, logs), so a dedicated
branch is the simplest way to bring them from the cluster to a laptop, and it
records which code version produced them. `results/` is git-ignored on
`main`, hence `-f`. The cluster needs push access to the repository: a GitHub
personal access token or an SSH key, and `git config user.name` /
`user.email`.

Use a second clone for this rather than switching branches in the working
repository: files tracked on the results branch but not on `main` would be
deleted from disk by `git checkout main`.

```bash
# on the cluster: a second clone only for results (create it once)
git clone https://github.com/filippozacchei/dealii-internodes.git $WORK/internodes-results
cd $WORK/internodes-results
git checkout -b results-galileo100

# ... and every time results should be sent back
cd $WORK/internodes-results
mkdir -p reproduce/results && cp -r $WORK/dealii-internodes/reproduce/results/. reproduce/results/
git add -f reproduce/results
git commit -m "Galileo100 results, code at $(git -C $WORK/dealii-internodes rev-parse --short HEAD)"
git push -u origin results-galileo100

# on the laptop: take the files without switching branches
git fetch origin results-galileo100
git checkout FETCH_HEAD -- reproduce/results
git reset -q reproduce/results               # unstage; the files stay (ignored)
```

Files with the same name (e.g. accuracy cases already run locally) are
overwritten by the cluster's.

The accuracy figures' finest levels (5 and 6) can be run on one node with a
batch script like this (errors do not depend on the machine or the rank
count; the CPU-time figure does, so run all levels on the same machine):

```bash
#!/bin/bash
#SBATCH --job-name=accuracy
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=12:00:00
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --output=accuracy.log
unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
module load <mpi> <libraries>
export INTERNODES_LAUNCHER="srun -n {np}"
cd <path>/dealii-internodes/reproduce
python3 accuracy.py run --max-level 6 --np 48 --timeout 28800
```

The phases are those of the paper: assembly of the interpolation operators
(destination-point search and RBF matrix setup), assembly of the internal
operators (`Step 0`), and Steps 1–4 of the algorithm, each as the maximum
over the ranks of its accumulated wall time. The speed-up is
$T(p_0)/T(p)$ with $p_0$ the smallest core count, and the ideal curve in the
time plot is normalised to 1 s at $p_0$, as in the paper. Note that the
algorithm assembles the subdomain matrices three times per solve, and the
"Assembly Internal Operators" time is the sum over the three.

Tests 4 and 5 (Geometry B, tetrahedra) use, until the Gmsh meshes are
available, the hexahedral half shell split into tetrahedra with placeholder
refinement levels (`Shell refinement`), so their DoF counts differ from
Table 4.

## Configuration files

| File | Test | Geometry | $p_1$ | $p_2$ | Interpolation | $r_f$ |
|---|---|---|---|---|---|---|
| `test1.prm` | 1 | A | 1 | 1 | Lagrange | -- |
| `test2.prm` | 2 | A | 1 | 1 | RL-RBF | 1 |
| `test3.prm` | 3 | A | 2 | 4 | RL-RBF | 1 |
| `test4.prm` | 4 | B | 1 | 1 | RL-RBF | 1 |
| `test5.prm` | 5 | B | 2 | 2 | RL-RBF | 10 |

They can also be run directly, `mpirun -np N ./coupled_diffusion reproduce/test1.prm`.
Tests 1–3 reproduce the master DoF counts of Table 4 exactly ($257^3$).
Geometry-A meshes are a single coarse cell per subdomain, refined globally on
the distributed mesh; Geometry-B meshes come from deal.II's half hyper-shell
split into tetrahedra, since the paper's Gmsh meshes are not available here.
