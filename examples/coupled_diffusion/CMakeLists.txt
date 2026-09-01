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
