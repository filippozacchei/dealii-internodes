#ifndef INTERNODES_MESH_HANDLER_HPP
#define INTERNODES_MESH_HANDLER_HPP

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping_q1.h>

#include "internodes/types.hpp"
#include "internodes/utilities.hpp"

#include <memory>

namespace internodes
{
  using namespace dealii;

  /**
   * @brief Thin wrapper around a distributed triangulation for one
   * subdomain.
   *
   * This is a deliberately minimal replacement for lifex::utils::MeshHandler
   * (which additionally handles parameter-file-driven mesh
   * generation/import). Construction of the triangulation itself (via
   * GridGenerator, GridIn, or a subdivided/merged mesh) is left to the
   * caller; MeshHandler here only owns the triangulation and provides the
   * linear (affine, Q1) mapping used throughout the interface/RBF DoF
   * handlers, matching the affine-mapping assumption already present in the
   * original interface_DoFHandler::setup_destination_points (which notes
   * "this only works exactly for linear mappings").
   */
  class MeshHandler
  {
  public:
    explicit MeshHandler(const MPI_Comm &comm = internodes::mpi_comm)
      : triangulation(comm)
    {}

    /// @return the underlying triangulation (mutable, for mesh generation).
    parallel::distributed::Triangulation<dim> &
    get()
    {
      return triangulation;
    }

    /// @return the underlying triangulation (read-only).
    const parallel::distributed::Triangulation<dim> &
    get() const
    {
      return triangulation;
    }

    /// @return a newly allocated linear (Q1) mapping. Deliberately returns
    /// a fresh unique_ptr on each call, matching the original MeshHandler
    /// interface, so that call sites can hold their own mapping object
    /// without lifetime coupling to the MeshHandler.
    std::unique_ptr<Mapping<dim>>
    get_linear_mapping() const
    {
      return std::make_unique<MappingQ1<dim>>();
    }

  private:
    parallel::distributed::Triangulation<dim> triangulation;
  };
} // namespace internodes

#endif // INTERNODES_MESH_HANDLER_HPP
