# Building on a cluster (candi)

One-time setup for the scalability runs of `scalability.py` (see
`reproduce/README.md`). Written for a generic SLURM cluster; adjust the
module names to whatever your site provides (`module avail` to check).

**Recommended path: steps 1-4**, a full, from-scratch candi build with a
generic compiler/MPI, exactly following the dependency list in the
top-level README. This is what makes the numbers representative of what a
reviewer with no site-specific knowledge -- and no pre-existing environment
-- would get by following that README, so it's the one to use for anything
that ends up reported. Step 0 below is only a quick, optional pre-check
for wiring problems (does the code even compile/link/run at all), not a
substitute for it.

## 0 (optional, quick check only). An existing deal.II, e.g. from a lifex install

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

Either way, once this quick check has satisfied your curiosity, do the real
build in steps 1-4 below for anything you intend to report -- in a
**separate `--prefix`** from `lifex-env` so it cannot disturb whatever else
still depends on that (e.g. `--prefix=$WORK/dealii-internodes-candi`), and
loading modules directly rather than sourcing `enable_lifex.sh` (which
would point `DEAL_II_DIR` back at the old 9.3.1 install).

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

Prefer a generic GCC + OpenMPI/MPICH here over a vendor toolchain (e.g. Intel
oneAPI), even if the latter is already set up in your own `.bashrc` and known
to work on this cluster: the point of this build is to match exactly what
the top-level README documents ("MPI (e.g. OpenMPI or MPICH)"), which is
what an unfamiliar reviewer would actually have on hand. Fall back to a
vendor toolchain only if no generic one is available on this cluster at
all.

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

If this stops with "Your operating system could not be automatically
recognised", candi's OS detection (from `/etc/os-release`) doesn't have a
matching platform file -- check `cat /etc/os-release` and compare its `ID`/
`VERSION_ID` against `deal.II-toolchain/platforms/{supported,contributed}/`,
then add `--platform=deal.II-toolchain/platforms/supported/<name>.platform`
to the command above. On **CentOS 8** (as on CINECA's Galileo100, `ID=centos`
`VERSION_ID=8`): there is no `centos8.platform` upstream, but
`almalinux8.platform` is the same RHEL-8-family OS, so use
`--platform=deal.II-toolchain/platforms/supported/almalinux8.platform`. If
the build later fails specifically at the Trilinos step on a parmetis
version-detection error, that platform file has a known one-line fix
(`TRILINOS_PARMETIS_CONFOPTS ... HAVE_PARMETIS_VERSION_4_0_3`) present but
commented out -- uncomment it and rerun.

**If Trilinos fails to find BLAS/LAPACK** ("Did not find a lib in the lib
set 'blas blas_win32 openblas'"): this happens when BLAS/LAPACK come from
Intel MKL (e.g. a `module load mkl/...`) rather than a standalone `libblas`,
which candi's Trilinos step doesn't look for automatically -- it has to be
told. candi already has first-class support for this (`deal.II-toolchain/
packages/trilinos.package` has an `MKL=ON` branch that sets the right
library names and Trilinos compile-flag workarounds), toggled in
`candi.cfg`, which `candi.sh` always `source`s (so this must be edited in
the file, not just exported in the shell):

```bash
echo $MKLROOT                                                    # sanity check
ls $MKLROOT/lib/intel64 | grep -E "^libmkl_(core|sequential|intel_lp64)\.so"

cd $WORK/candi   # your candi checkout
sed -i 's/^MKL=OFF/MKL=ON/' candi.cfg
sed -i 's|^# MKL_DIR=|MKL_DIR=$MKLROOT/lib/intel64|' candi.cfg    # $MKLROOT left
                                                                   # unexpanded on
                                                                   # purpose; candi.sh
                                                                   # resolves it at run time
```

Then rerun `./candi.sh` -- but **not** the exact same command: by default
candi refetches and rebuilds *every* package named in `--packages` from
scratch on every invocation, even ones that already succeeded (there is no
automatic checkpointing -- an earlier version of this note claimed there
was; it doesn't). Mark packages that already built successfully with a
`once:` prefix so they're actually skipped (this checks for a
`candi_successful_build` marker file candi writes in that package's build
directory on success, e.g.
`$WORK/dealii-candi/tmp/build/p4est-2.8.7/candi_successful_build`) --
otherwise every retry of a later step pays for the full Trilinos build
again:

```bash
./candi.sh --packages="once:p4est once:trilinos dealii" \
           --prefix=$WORK/dealii-candi -j 8 --yes \
           --platform=deal.II-toolchain/platforms/supported/almalinux8.platform
```

**If Kokkos (a Trilinos dependency) refuses to configure with "Compiler not
supported ... Intel: not supported"**: recent Kokkos versions dropped
support for the classic Intel compiler (`icc`/`icpc`) for C++17 entirely --
only `IntelLLVM` (`icx`/`icpx`) or GCC >= 8.2.0 work. If a GCC module is
also loaded (as in a typical CINECA `.bashrc`), Intel MPI conventionally
ships GCC-wrapped compiler drivers (`mpicc`/`mpicxx`/`mpif90`) alongside the
Intel-wrapped ones (`mpiicc`/`mpiicpc`/`mpiifort`) -- same MPI library, a
different underlying compiler -- so this is usually a one-line swap rather
than restarting with a different MPI module entirely:

```bash
which mpicc mpicxx mpif90
mpicc -show                          # confirm it wraps gcc, not icc
```

If those check out, use `cluster_env.sh` (this directory) rather than
exporting `CC`/`CXX`/`FC` ad hoc: it's a small, explicit, reusable script,
and the *same* compiler is needed consistently at every later step too
(building `dealii-internodes` in step 3, running it in step 4, and any
batch job `scalability.py` submits), since mixing a GCC-built Trilinos/
deal.II with Intel-compiled application code risks a C++ ABI mismatch.

**If this account is dedicated to this project** (not shared with other
work that might want the Intel classic compiler on purpose), it's also
reasonable to just make the swap permanent in `.bashrc` instead, mirroring
the block already there:

```bash
cp ~/.bashrc ~/.bashrc.bak-$(date +%Y%m%d)
sed -i \
  -e 's|^export CC=\${MPICC}|export CC=mpicc|' \
  -e 's|^export CXX=\${MPICXX}|export CXX=mpicxx|' \
  -e 's|^export FC=\${MPIFC}|export FC=mpif90|' \
  -e 's|^export FF=\${MPIF77}|export FF=mpif90|' \
  -e 's|^export F77=\${MPIF77}|export F77=mpif90|' \
  -e 's|^export F90=\${MPIF90}|export F90=mpif90|' \
  ~/.bashrc
```

then log out and back in. With that done, `cluster_env.sh` becomes
redundant (harmless if sourced anyway -- it just re-exports the same
values) since every new shell already has the right compiler by default.

```bash
source reproduce/cluster_env.sh      # sets CC=mpicc, CXX=mpicxx, FC=mpif90
                                       # (skip if already permanent in .bashrc)

# CMake caches the detected compiler in the build directory from the failed
# attempt(s); clearing it forces a fresh configure with the new compiler
# (safe -- it's candi's scratch build dir, not the final install prefix):
rm -rf $WORK/dealii-candi/tmp/build/trilinos-release-16-2-0
```

Then rerun the same `./candi.sh ...` command.

**If Trilinos then fails on a *different* TPL you never asked for** (e.g.
SCALAPACK, ParMETIS, MUMPS) **and the library paths in the log point outside
your candi `--prefix`** (e.g. into a `lifex-env`-style directory): some
other environment already on this machine -- typically sourced
unconditionally from `.bashrc` on every login, as CINECA's lifex
environments often are -- is exporting `..._DIR` variables that
`trilinos.package` picks up and force-enables TPLs for. Check what's
actually set and strip it before building:

```bash
env | grep -iE "_dir=|_lib|scalapack|parmetis|mumps" | sort
unset PARMETIS_DIR SCALAPACK_DIR MUMPS_DIR PETSC_DIR SLEPC_DIR ADOLC_DIR ARPACK_DIR TRILINOS_DIR P4EST_DIR
```

then rerun `./candi.sh ...` again. This port needs neither ParMETIS nor
ScaLAPACK (p4est's own z-order partitioner is used, not METIS), so with
these unset Trilinos should simply stop trying to enable them rather than
needing them pointed anywhere.

`unset` only fixes the current shell, though, and the leak comes back on
every new login. **More robust:** comment out (don't delete, so it's a
one-line revert if `lifex` itself is still needed on this cluster for
something else) the line in `~/.bashrc` that sources the lifex environment,
then log out and back in for a genuinely clean shell:

```bash
cp ~/.bashrc ~/.bashrc.bak-$(date +%Y%m%d)
sed -i 's|^source .*/enable_lifex\.sh|# &|' ~/.bashrc
tail -3 ~/.bashrc                              # confirm it's commented, not deleted
# log out, log back in, then:
env | grep -iE "_dir=|_lib|scalapack|parmetis|mumps" | sort    # should now show nothing
                                                                 # from lifex-env
```

The rest of `.bashrc` (compiler/MPI/MKL/boost/hdf5/cmake modules) still
loads normally on login, so nothing else about the environment changes --
only the lifex-specific package paths stop leaking in. The `candi.cfg`
`MKL=ON` edit from before is unaffected (it lives in the candi checkout, not
the shell), so rerunning `./candi.sh ...` should pick up right where it
left off.

- `--prefix` is where everything gets installed; `-j` is the parallel build
  job count (match `--cpus-per-task` above). No `-j` given defaults to a
  small number, so pass it explicitly for a faster build.
- This installs Trilinos and p4est along with deal.II — no separate METIS
  build is needed (the port's `MeshHandler` uses p4est / the z-order
  partitioner, not METIS).
- Expect on the order of 1-3 hours depending on the node and `-j`. It is
  safe to rerun if interrupted, but only packages named with a `once:`
  prefix are actually skipped on a rerun (see below) -- a plain package
  name is refetched and rebuilt from scratch every time.
- If the compute nodes have no internet access (common), make sure this runs
  on a node that does (usually the login node), since candi downloads
  tarballs of each package.

**If deal.II's own configure fails with "Could not find any suitable mpi
library!" / "Could NOT find MPI_CXX (missing: MPI_CXX_LIB_NAMES
MPI_CXX_HEADER_DIR MPI_CXX_WORKS)"** -- even though `mpicc`/`mpicxx`/`mpif90`
all work fine standalone and Trilinos just built successfully with the same
compilers: this is CMake's `FindMPI` failing to *manually* locate MPI's
headers/libraries for C++ specifically. It happens because candi's
`USE_DEAL_II_CMAKE_MPI_COMPILER=ON` (a workaround for a different, genuine
deal.II/CMake issue, #11478 -- candi's own comment above it admits "this
currently is not reliable enough to enable by default", yet it *is* ON by
default) `unset`s `CXX`/`CC` before configuring deal.II, so
`CMAKE_CXX_COMPILER` ends up a plain `g++` rather than the `mpicxx` wrapper.
With a plain compiler, `FindMPI` has to search for MPI's headers/libraries
itself -- and on a cluster with two mount-point names for the same install
(here, CINECA's `/g100/prod/...` and `/cineca/prod/...` both leading to the
same physical Intel MPI), its C-language search happened to land on one
alias while its C++-language search only tries the other, which doesn't
have the file CMake is looking for at that exact path. Confirmed by
tracing `deal.II-toolchain/packages/dealii.package` and
`cmake/configure/configure_10_mpi.cmake`/`cmake/modules/FindDEAL_II_MPI.cmake`
in the deal.II source, and by CMake's own `--debug-find` trace (added to
`DEAL_II_CONFOPTS`) showing the C++ `find_path` candidate list never
contained the directory that has `mpi.h`, while the C search's did.

Two things that looked plausible but did **not** fix it, so they aren't
worth retrying: passing explicit absolute-path
`-D MPI_C_COMPILER=... -D MPI_CXX_COMPILER=... -D MPI_Fortran_COMPILER=...`
via `DEAL_II_CONFOPTS` (candi's existing `USE_DEAL_II_CMAKE_MPI_COMPILER=ON`
mechanism already does something equivalent with bare names, and the
underlying alias-search bug persists regardless of absolute vs. bare); and
exporting `MPI_HOME` (an environment hint `FindMPI` is documented to read,
but it made no difference here).

The actual fix: turn that workaround **off**, so `CMAKE_CXX_COMPILER` stays
the `mpicxx` wrapper. When the C++ compiler *is* the MPI wrapper, `FindMPI`
recognizes this immediately and succeeds with empty
`MPI_CXX_INCLUDE_DIRS`/`MPI_CXX_LIBRARIES` -- correct, not a bug, since the
wrapper already bakes those flags into every compile/link (confirmed with a
plain `mpicxx test.cpp -o test && ./test`, and with a tiny standalone CMake
project doing only `find_package(MPI REQUIRED COMPONENTS CXX)`, which is
worth reaching for early in place of iterating on the full, much slower
deal.II configure -- it reports the same result in a couple of seconds):

```bash
cd $WORK/candi
sed -i 's/^USE_DEAL_II_CMAKE_MPI_COMPILER=ON/USE_DEAL_II_CMAKE_MPI_COMPILER=OFF/' candi.cfg
grep USE_DEAL_II_CMAKE_MPI_COMPILER candi.cfg    # confirm it now says OFF

rm -rf $WORK/dealii-candi/tmp/build/deal.II-v9.8.0

./candi.sh --packages="once:p4est once:trilinos dealii" \
           --prefix=$WORK/dealii-candi -j 8 --yes \
           --platform=deal.II-toolchain/platforms/supported/almalinux8.platform
```

## 3. Build dealii-internodes against it

If step 2 needed the `cluster_env.sh` compiler swap (Kokkos/Intel), `source`
it again here first -- this build must use the same compiler deal.II/
Trilinos were built with.

```bash
source reproduce/cluster_env.sh   # only if step 2 needed it; harmless otherwise
ls $WORK/dealii-candi/                          # find the actual deal.II-vX.Y.Z name
                                                 # candi fetches the latest tagged
                                                 # release when no version is pinned,
                                                 # so don't assume it matches this guide
cd /path/to/dealii-internodes
mkdir build && cd build
cmake -DDEAL_II_DIR=$WORK/dealii-candi/deal.II-vX.Y.Z -DCMAKE_BUILD_TYPE=Release ..
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
module load <whatever MPI module candi built deal.II against>
source reproduce/cluster_env.sh   # only if step 2 needed it; harmless otherwise
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
  If step 2 needed `cluster_env.sh`, add
  `$'\n'"source reproduce/cluster_env.sh"` too, so the batch job uses the
  same compiler the executable was actually built and linked with.
- `--launcher` (default `srun`) is whatever your site uses to start an MPI
  program inside a batch script (`srun`, or `mpirun` on some systems).
- If the candi build fails at the Trilinos step (needed for the AMG/ML
  preconditioner), see the BLAS/LAPACK and parmetis notes under step 2
  above; `--platform=<file>` (also step 2) covers most other OS-specific
  quirks.
