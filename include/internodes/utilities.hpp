#ifndef INTERNODES_UTILITIES_HPP
#define INTERNODES_UTILITIES_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>

#include <algorithm>
#include <iostream>
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

  /// Parallel-conditional output stream: only rank 0 actually prints,
  /// avoiding duplicated output across MPI processes. Replaces lifex's
  /// global `pcout` object.
  ///
  /// Implemented as a function returning a function-local static, rather
  /// than a plain namespace-scope global, because its initializer calls
  /// Utilities::MPI::this_mpi_process(), which requires MPI_Init() to have
  /// already run. A namespace-scope `inline` variable would be constructed
  /// at static-initialization time -- before main() and its
  /// MPI_InitFinalize -- which is undefined behavior. A function-local
  /// static is instead constructed on first call, which in practice only
  /// ever happens from within main(), after MPI has been initialized.
  inline ConditionalOStream &
  pcout()
  {
    static ConditionalOStream instance(
      std::cout, Utilities::MPI::this_mpi_process(mpi_comm) == 0);
    return instance;
  }

  /// Shared timer, printing a summary of wall-clock times for named scopes
  /// (TimerOutput::Scope) on program exit. Replaces lifex's global
  /// `timer_output` object. Construct-on-first-use for the same MPI
  /// initialization-order reason as pcout() above (hence defined after it:
  /// it uses pcout() itself, see below).
  inline TimerOutput &
  timer_output()
  {
    // TimerOutput::never, not ::summary: with ::summary the destructor
    // auto-prints via an MPI collective (to gather per-rank timings), but
    // this object is a function-local static and so is destroyed at
    // program-exit time -- *after* main()'s own MPI_InitFinalize object
    // (an ordinary local variable, destroyed when main() returns) has
    // already called MPI_Finalize. Doing an MPI call after that point
    // aborts on strict MPI implementations (observed with Intel MPI/MPICH;
    // OpenMPI tolerates it silently, which is why this went unnoticed in
    // local testing). Callers must explicitly call
    // timer_output().print_summary() before main() returns instead.
    //
    // pcout(), not std::cout: TimerOutput has two MPI_Comm-taking
    // constructor overloads -- one for a raw std::ostream, where *every*
    // rank writes the summary (all ranks compute it, since it's a
    // collective, but each also prints its own copy -- observed as the
    // same table N times over, interleaved, for N ranks), and one for a
    // ConditionalOStream, which is how deal.II restricts printing to a
    // single rank. pcout() is exactly that.
    static TimerOutput instance(mpi_comm,
                                 pcout(),
                                 TimerOutput::never,
                                 TimerOutput::wall_times);
    return instance;
  }

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
