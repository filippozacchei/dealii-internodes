#ifndef INTERNODES_UTILITIES_HPP
#define INTERNODES_UTILITIES_HPP

#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>

#include <algorithm>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace internodes
{
  using namespace dealii;

  /// Shared MPI communicator. The original lifex-based code used a global
  /// Core::mpi_comm singleton managed by lifex::CoreModel; ported here as a
  /// plain constant since this project has no such application framework.
  inline const MPI_Comm mpi_comm = MPI_COMM_WORLD;

  /// Shared timer, printing a summary of wall-clock times for named scopes
  /// (TimerOutput::Scope) on program exit. Replaces lifex's global
  /// `timer_output` object.
  inline TimerOutput timer_output(mpi_comm,
                                   std::cout,
                                   TimerOutput::summary,
                                   TimerOutput::wall_times);

  /// @return whether @p value is contained in @p container.
  template <typename T>
  bool
  contains(const std::set<T> &container, const T &value)
  {
    return container.find(value) != container.end();
  }

  /// Merges a map that is only partially filled on each MPI process (with
  /// disjoint keys across processes -- each key is set by exactly one rank,
  /// e.g. because it corresponds to a destination point found in a locally
  /// owned cell) into the same, fully-populated map on every process.
  ///
  /// @tparam Key    map key type; must be trivially copyable (e.g. an
  ///                unsigned integral type).
  /// @tparam Value  map value type; must be trivially copyable.
  template <typename Key, typename Value>
  std::map<Key, Value>
  compute_map_union(const std::map<Key, Value> &local_map,
                     const MPI_Comm             &comm)
  {
    const std::vector<std::map<Key, Value>> all_maps =
      Utilities::MPI::all_gather(comm, local_map);

    std::map<Key, Value> result;
    for (const auto &rank_map : all_maps)
      result.insert(rank_map.begin(), rank_map.end());

    return result;
  }
} // namespace internodes

#endif // INTERNODES_UTILITIES_HPP
