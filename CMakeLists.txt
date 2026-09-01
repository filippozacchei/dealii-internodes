## ---------------------------------------------------------------------
##
## dealii-internodes: a deal.II-only implementation of the INTERNODES
## method for non-conforming interface coupling.
##
## ---------------------------------------------------------------------

CMAKE_MINIMUM_REQUIRED(VERSION 3.13)

FIND_PACKAGE(deal.II 9.5 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to CMake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()

# The Schur-complement preconditioner uses Trilinos' algebraic multigrid
# (ML), and the subdomain meshes are parallel::distributed::Triangulation
# objects, which requires p4est.
IF(NOT DEAL_II_WITH_TRILINOS)
  MESSAGE(FATAL_ERROR "\n"
    "*** deal.II was not configured with Trilinos support, which "
    "dealii-internodes requires (TrilinosWrappers linear algebra and the "
    "algebraic multigrid preconditioner). ***\n"
    )
ENDIF()
IF(NOT DEAL_II_WITH_P4EST)
  MESSAGE(FATAL_ERROR "\n"
    "*** deal.II was not configured with p4est support, which "
    "dealii-internodes requires (parallel::distributed::Triangulation). ***\n"
    )
ENDIF()

PROJECT(dealii-internodes CXX)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include)

FILE(GLOB INTERNODES_SOURCES "${CMAKE_SOURCE_DIR}/source/*.cpp")

ADD_LIBRARY(internodes ${INTERNODES_SOURCES})
DEAL_II_SETUP_TARGET(internodes)

ADD_SUBDIRECTORY(examples)
