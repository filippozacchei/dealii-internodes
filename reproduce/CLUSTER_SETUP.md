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

Needed in principle: version >= 9.5, and `DEAL_II_WITH_TRILINOS`/
`DEAL_II_WITH_P4EST`/`DEAL_II_WITH_MPI` all `#define`d to 1. If that's the
case, skip straight to step 3 below with this `DEAL_II_DIR`.

**If the version is older (e.g. a candi-based lifex environment often has
something like deal.II-9.3.x):** don't discard it outright -- a `cmake`+
`make` attempt costs a few minutes, against the 1-3 hours of a fresh candi
build, so it's worth just trying first, *especially* for Tests 1-3, which
are purely hexahedral and so don't touch this port's simplex/tetrahedra
code path (the part most likely to be fragile on an older deal.II; simplex
support was still experimental in the 9.3-9.4 era). A handful of newer
DoFTools/AffineConstraints/TimerOutput call signatures used here might not
exist on an older deal.II either, but those are the kind of thing that
either isn't hit by the hex-only path, or shows up as a clear, easily-fixed
compile error rather than a silent problem:

```bash
source /path/to/lifex-env/configuration/enable_lifex.sh   # sets DEAL_II_DIR,
                                                            # LD_LIBRARY_PATH etc.
cd /path/to/dealii-internodes
mkdir build-old && cd build-old
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8 coupled_diffusion
```

Two outcomes:
- **`cmake` itself refuses** (version check, or missing Trilinos/p4est) --
  paste the message back; that's a clear "use candi instead" (steps 1-4).
- **`cmake` succeeds but `make` fails** -- paste the *first* compiler error
  (not the whole log); it will name the exact API that changed, which is
  usually a small, mechanical fix rather than a reason to rebuild deal.II.
- **Both succeed** -- run it against one of the small hex cases in
  `reproduce/` or `examples/coupled_diffusion/coupled_diffusion.prm` and
  check the printed error/iteration count against the ones recorded in the
  top-level README, to make sure an older deal.II hasn't silently changed
  any numerics.

If any of this points to "rebuild", fall back to steps 1-4, and use the
Intel oneAPI toolchain from your `.bashrc` (`intel/oneapi-2021`,
`intelmpi/oneapi-2021`) rather than GNU+OpenMPI -- it's already proven to
build this exact dependency stack on this cluster (that's what built the
`lifex-env` you just listed), and `CC`/`CXX` are already set to the right
MPI compiler wrappers once those modules are loaded (`export CC=${MPICC}`
etc., as in your `.bashrc`). Use a **separate `--prefix`** from `lifex-env`
so this build cannot disturb whatever still depends on it, e.g.
`--prefix=$WORK/dealii-internodes-candi`, and load the modules directly
rather than sourcing `enable_lifex.sh` (which would point `DEAL_II_DIR` back
at the old 9.3.1 install).

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
