#ifndef INTERNODES_RTREE_HANDLER_HPP
#define INTERNODES_RTREE_HANDLER_HPP

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/point.h>

// Registers dealii::Point/BoundingBox with boost::geometry's trait system
// (used internally by deal.II's own RTree utilities). The original lifex
// code relied on this being pulled in transitively through lifex's
// aggregating "source/core.hpp" header; ported here as an explicit,
// self-documenting dependency instead of an implicit one.
#include <deal.II/numerics/rtree.h>

#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>

#include "internodes/types.hpp"

#include <utility>
#include <vector>

namespace bgi = boost::geometry::index;
namespace bg  = boost::geometry;

namespace internodes
{
  using namespace dealii;

  /**
   * @brief Nearest-neighbor search structure over a fixed set of support
   * points, used to accelerate RL-RBF interpolation stencil construction
   * (see interface_DoFHandler_RBF).
   *
   * Wraps a boost::geometry R-tree over the given points; @ref query
   * returns all support points within a given radius of a destination
   * point.
   */
  class RTreeHandler
  {
  public:
    RTreeHandler(const std::vector<Point<dim>> &support_points_global,
                 double                          radius);

    std::vector<std::pair<Point<dim>, unsigned int>>
    query(Point<dim> point) const;

  private:
    using RTree =
      bgi::rtree<std::pair<Point<dim>, unsigned int>, bgi::quadratic<16>>;

    void
    initialize_rtree();

    const std::vector<Point<dim>> &support_points_global;
    double                          radius;
    RTree                           rtree;
  };
} // namespace internodes

#endif // INTERNODES_RTREE_HANDLER_HPP
