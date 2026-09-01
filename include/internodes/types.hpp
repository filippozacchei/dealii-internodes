#ifndef INTERNODES_TYPES_HPP
#define INTERNODES_TYPES_HPP

namespace internodes
{
  /**
   * @brief Geometrical space dimension.
   *
   * The original lifex-based implementation configured this as a CMake
   * build-time constant (LIFEX_DIM). All the paper's numerical results are
   * 3D, so it is fixed here; templating this on the dimension (as deal.II
   * tutorials typically do) is a natural follow-up if 2D support is ever
   * needed.
   */
  constexpr unsigned int dim = 3;
} // namespace internodes

#endif // INTERNODES_TYPES_HPP
