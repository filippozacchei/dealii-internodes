# Source this before every step that builds or runs dealii-internodes on a
# cluster where CC/CXX/FC default to the classic Intel compiler wrappers
# (e.g. Galileo100's .bashrc: export CC=${MPICC}, which resolves to
# mpiicc). Recent Kokkos (a Trilinos dependency) does not support that
# compiler for C++17 -- see CLUSTER_SETUP.md. These are the GCC-wrapped
# equivalents from the *same* Intel MPI installation (same MPI library and
# transport, different underlying compiler).
#
# Deliberately not folded into .bashrc: CC/CXX/FC are the default compiler
# for anything built on this login, not just this project, and once
# deal.II/Trilinos are built with one compiler, this port must be built
# (step 3) and run (step 4, and any scalability.py batch job) with the
# *same* one for ABI compatibility -- so source this explicitly at each of
# those points rather than relying on a background default.
#
#   source reproduce/cluster_env.sh
#
export CC=mpicc
export CXX=mpicxx
export FC=mpif90
