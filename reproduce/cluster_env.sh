# Source this before every step that builds or runs dealii-internodes on a
# cluster where CC/CXX/FC default to the classic Intel compiler wrappers
# (e.g. Galileo100's .bashrc: export CC=${MPICC}, which resolves to
# mpiicc). Recent Kokkos (a Trilinos dependency) does not support that
# compiler for C++17 -- see CLUSTER_SETUP.md. These are the GCC-wrapped
# equivalents from the *same* Intel MPI installation (same MPI library and
# transport, different underlying compiler).
#
# Not folded into .bashrc by default: CC/CXX/FC are the default compiler
# for anything built on this login, not just this project, and once
# deal.II/Trilinos are built with one compiler, this port must be built
# (step 3) and run (step 4, and any scalability.py batch job) with the
# *same* one for ABI compatibility -- so source this explicitly at each of
# those points rather than relying on a background default, UNLESS you
# know this account is dedicated to this project, in which case putting
# the same export lines directly in .bashrc is a reasonable simplification
# (see CLUSTER_SETUP.md) -- in that case sourcing this script is harmless
# but redundant (it just re-exports the same values).
#
#   source reproduce/cluster_env.sh
#
export CC=mpicc
export CXX=mpicxx
export FC=mpif90
