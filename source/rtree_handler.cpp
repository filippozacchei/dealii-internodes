#include "internodes/rtree_handler.hpp"

#include <algorithm>

namespace internodes
{
  RTreeHandler::RTreeHandler(
    const std::vector<Point<dim>> &support_points_global,
    double                          radius)
    : support_points_global(support_points_global)
    , radius(radius)
  {
    initialize_rtree();
  }

  void
  RTreeHandler::initialize_rtree()
  {
    for (unsigned int j = 0; j < support_points_global.size(); ++j)
      {
        rtree.insert(std::make_pair(support_points_global[j], j));
      }
  }

  std::vector<std::pair<Point<dim>, unsigned int>>
  RTreeHandler::query(Point<dim> point) const
  {
    Point<dim> point_min, point_max;
    for (unsigned int d = 0; d < dim; ++d)
      {
        point_min[d] = point[d] - radius;
        point_max[d] = point[d] + radius;
      }

    const BoundingBox<dim> local_box(std::make_pair(point_min, point_max));

    // Broad-phase query: all support points whose bounding box intersects
    // the (square) box of the given radius around `point`.
    std::vector<std::pair<Point<dim>, unsigned int>> result_n;
    result_n.reserve(support_points_global.size());
    rtree.query(bgi::intersects(local_box), std::back_inserter(result_n));

    // Narrow-phase check: the broad-phase box query over-selects points in
    // the box's corners that lie outside the actual (circular/spherical)
    // radius; filter those out with an exact distance check.
    result_n.erase(std::remove_if(result_n.begin(),
                                   result_n.end(),
                                   [&point, this](const auto &entry) {
                                     return point.distance_square(
                                              entry.first) >
                                            this->radius * this->radius;
                                   }),
                   result_n.end());
    return result_n;
  }
} // namespace internodes
