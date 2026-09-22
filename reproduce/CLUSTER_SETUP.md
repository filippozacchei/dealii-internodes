# Building on a cluster (candi)

One-time setup for the scalability runs of `scalability.py` (see
`reproduce/README.md`). Written for a generic SLURM cluster; adjust the
module names to whatever your site provides (`module avail` to check).

## 0. Check for an existing deal.II first (e.g. from a lifex install)

If you already have `lifex` set up on this cluster, its environment script
most likely already builds and exposes a deal.II with everything this port
needs (MPI, Trilinos, p4est) -- lifex itself requires the same. If so, this
whole candi build can be skipped. Check before doing anything else:

```bash
source /path/to/lifex-env/configuration/enable_lifex.sh   # your own path
env | grep -i deal_ii                 # look for DEAL_II_DIR or similar
ls /path/to/lifex-env/                # candi-based envs usually have a
                                       # sibling deal.II-vX.Y.Z/ next to
                                       # configuration/ (see below)
```

Once you have a candidate `$DEAL_II_DIR` (or a `deal.II-vX.Y.Z` directory),
check it is recent and complete enough:

```bash
grep DEAL_II_PACKAGE_VERSION  $DEAL_II_DIR/include/deal.II/base/config.h
grep -E "DEAL_II_WITH_(TRILINOS|P4EST|MPI) " $DEAL_II_DIR/include/deal.II/base/config.h
```

Needed: version >= 9.5, and `DEAL_II_WITH_TRILINOS`/`DEAL_II_WITH_P4EST`/
`DEAL_II_WITH_MPI` all `#define`d to 1 (not commented out to 0). If that's
the case, skip straight to step 3 below with this `DEAL_II_DIR`. If it's
missing, too old, or the environment no longer works (as can happen after
cluster software updates), fall back to steps 1-2.

Tell me what these print and I'll say which path to take.

## 1. Load a compiler and MPI, on a login/build node

candi needs to compile deal.II and its dependencies (Trilinos, p4est) from
source, which takes CPU and disk. Most clusters either allow this on the
login node in moderation, or want it done in an interactive job:

```bash
module avail 2>&1 | grep -iE "gcc|openmpi|mpich|cmake"   # find what's on offer
module load gcc/<version> openmpi/<version> cmake         # or your site's names

# if the login node disallows a long compile, get an interactive allocation
# first (flags vary by site -- check `sinfo`/`sacctmgr` or your site's docs):
salloc --account=<account> --partition=<partition> --time=03:00:00 --ntasks=1 --cpus-per-task=8
```

Build on a filesystem with real disk quota (`$WORK`, `$SCRATCH`, `$CINECA_SCRATCH`,
...), not `$HOME`, which is usually small.

## 2. Build deal.II with candi

```bash
cd $WORK   # or wherever you decided to build
git clone https://github.com/dealii/candi.git
cd candi
./candi.sh --packages="p4est trilinos dealii" \
           --prefix=$WORK/dealii-candi \
           -j 8 \
           --yes
```

- `--prefix` is where everything gets installed; `-j` is the parallel build
  job count (match `--cpus-per-task` above). No `-j` given defaults to a
  small number, so pass it explicitly for a faster build.
- This installs Trilinos and p4est along with deal.II — no separate METIS
  build is needed (the port's `MeshHandler` uses p4est / the z-order
  partitioner, not METIS).
- Expect on the order of 1-3 hours depending on the node and `-j`. It is safe
  to rerun if interrupted; candi skips packages it already built.
- If the compute nodes have no internet access (common), make sure this runs
  on a node that does (usually the login node), since candi downloads
  tarballs of each package.

## 3. Build dealii-internodes against it

```bash
cd /path/to/dealii-internodes
mkdir build && cd build
cmake -DDEAL_II_DIR=$WORK/dealii-candi/deal.II-v9.7.0 -DCMAKE_BUILD_TYPE=Release ..
make -j 8 coupled_diffusion
```

`cmake` stops with a clear error if the deal.II found lacks Trilinos or
p4est support — if that happens, double check `--packages` above included
both.

## 4. Check the executable runs under the cluster's launcher

Before queuing anything from `scalability.py`, confirm the binary actually
starts under `srun` (or your launcher) with the modules loaded, e.g. in the
same interactive allocation as step 1:

```bash
module load openmpi/<version>   # whatever MPI candi built deal.II against
srun -n 4 ./build/examples/coupled_diffusion/coupled_diffusion \
     reproduce/coupled_diffusion.prm
```

It should print "GMRES converged in N iterations." and a broken H1 error in
a few seconds. If this works, `scalability.py prepare` (see
`reproduce/README.md`, smoke test first) will work too — its batch scripts
run the exact same command.

## Notes

- `scalability.py prepare --modules "..."` is a single line inserted
  verbatim into each batch script before the run command; pass all the
  `module load` commands needed to reproduce the environment of steps 3-4,
  separated by `&&` or newlines within the quoted string, e.g.
  `--modules "module load gcc/<v> openmpi/<v>"$'\n'"module load cmake"`.
- `--launcher` (default `srun`) is whatever your site uses to start an MPI
  program inside a batch script (`srun`, or `mpirun` on some systems).
- If Trilinos' AMG (ML/MueLu) needs a specific BLAS/LAPACK on your cluster,
  candi normally picks up whatever `module load`ed compiler/MKL is visible;
  if the candi build fails at the Trilinos step, that's the first thing to
  check (`--platform=<file>` lets you point candi at a site-specific
  configuration if your cluster already has one contributed upstream).
