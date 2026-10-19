ADD_EXECUTABLE(coupled_diffusion coupled_diffusion.cpp)
TARGET_LINK_LIBRARIES(coupled_diffusion internodes)
DEAL_II_SETUP_TARGET(coupled_diffusion)

# Copy the default parameter file next to the built executable, so
# `./coupled_diffusion` works out of the box from the build directory.
CONFIGURE_FILE(
  ${CMAKE_CURRENT_SOURCE_DIR}/coupled_diffusion.prm
  ${CMAKE_CURRENT_BINARY_DIR}/coupled_diffusion.prm
  COPYONLY
  )

# Record the git commit the binary was built from (see cmake/git_version.cmake)
# in the results file -- an out-of-date binary once silently produced a whole
# set of results that did not match the source.
ADD_CUSTOM_TARGET(internodes_git_version
  COMMAND ${CMAKE_COMMAND} -DSRC=${CMAKE_SOURCE_DIR}
          -DOUT=${CMAKE_CURRENT_BINARY_DIR}/git_version.h
          -P ${CMAKE_SOURCE_DIR}/cmake/git_version.cmake
  BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/git_version.h
  )
ADD_DEPENDENCIES(coupled_diffusion internodes_git_version)
TARGET_INCLUDE_DIRECTORIES(coupled_diffusion PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
