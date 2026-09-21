#ifndef INTERNODES_MESH_HANDLER_HPP
#define INTERNODES_MESH_HANDLER_HPP

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/tria_base.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/reference_cell.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_description.h>

#include "internodes/types.hpp"

#include <set>
#include "internodes/utilities.hpp"

#include <memory>
#include <vector>

namespace internodes
{
  using namespace dealii;

  /**
   * @brief Owns the parallel triangulation of one subdomain, and provides
   * the cell-type-dependent finite element, quadrature, and mapping objects
   * used throughout the interface handlers and SubProblem.
   *
   * Deliberately minimal replacement for lifex::utils::MeshHandler (which
   * additionally handles parameter-file-driven mesh generation/import).
   *
   * **Why the triangulation is held through a base-class pointer.** The
   * paper's two geometries need two different parallel triangulation
   * types: Geometry-A is hexahedral, which deal.II distributes with
   * p4est (parallel::distributed::Triangulation); Geometry-B is
   * tetrahedral, and p4est only supports quadrilateral/hexahedral forests,
   * so simplex meshes must instead use
   * parallel::fullydistributed::Triangulation. Both derive from
   * parallel::DistributedTriangulationBase, which is what is stored here
   * (matching the original lifex::utils::MeshHandler), and which is all a
   * DoFHandler needs.
   *
   * Usage: build the mesh as an ordinary *serial* dealii::Triangulation
   * (identically on every MPI rank, via GridGenerator/GridIn), then hand it
   * to create(), which picks the right parallel type from the mesh's cell
   * type and distributes it.
   */
  class MeshHandler
  {
  public:
    explicit MeshHandler(const MPI_Comm &comm = internodes::mpi_comm)
      : comm(comm)
    {}

    /// Distributes @p serial_tria -- a complete serial triangulation,
    /// identical on every rank -- across the MPI processes. Only meshes
    /// made entirely of hexahedra or entirely of tetrahedra are supported.
    ///
    /// Boundary ids are preserved from the caller's point of view: the ids
    /// of the mesh passed in are the ids to use everywhere else in the
    /// library. (For tetrahedral meshes the ids stored internally are
    /// shifted by one; see to_internal_boundary_id().)
    void
    create(const Triangulation<dim> &serial_tria)
    {
      if (is_hex(serial_tria))
        {
          auto tria =
            std::make_unique<parallel::distributed::Triangulation<dim>>(comm);
          tria->copy_triangulation(serial_tria);
          triangulation = std::move(tria);
        }
      else
        {
          // p4est cannot hold simplices, so use a fully distributed
          // triangulation. Partition with the z-order partitioner, which
          // (unlike GridTools::partition_triangulation) does not need METIS.
          Triangulation<dim> partitioned;
          partitioned.copy_triangulation(serial_tria);

          // A fully distributed triangulation gives the faces on the
          // partition boundary (those facing cells that are not stored
          // locally) the default boundary id 0. A genuine boundary with id
          // 0 -- the deal.II default, and what GridGenerator colorizing
          // often produces -- would then be indistinguishable from them on
          // ghost cells, and DoFTools::extract_boundary_dofs() and
          // VectorTools::interpolate_boundary_values() would pick up
          // spurious "boundary" DoFs. Free up id 0 by shifting the ids of
          // all genuine boundary faces by one.
          for (const auto &cell : partitioned.active_cell_iterators())
            for (const auto &face : cell->face_iterators())
              if (face->at_boundary())
                face->set_boundary_id(face->boundary_id() + 1);
          boundary_id_shift = 1;

          GridTools::partition_triangulation_zorder(
            Utilities::MPI::n_mpi_processes(comm), partitioned);

          const auto description =
            TriangulationDescription::Utilities::create_description_from_triangulation(
              partitioned, comm);

          auto tria =
            std::make_unique<parallel::fullydistributed::Triangulation<dim>>(comm);
          tria->create_triangulation(description);
          triangulation = std::move(tria);
        }
    }

    /// Translates a boundary id of the mesh passed to create() into the id
    /// under which the same boundary is stored in get(). The identity for
    /// hexahedral meshes.
    types::boundary_id
    to_internal_boundary_id(const types::boundary_id id) const
    {
      return id + boundary_id_shift;
    }

    std::set<types::boundary_id>
    to_internal_boundary_ids(const std::set<types::boundary_id> &ids) const
    {
      std::set<types::boundary_id> result;
      for (const auto id : ids)
        result.insert(to_internal_boundary_id(id));
      return result;
    }

    /// @return the underlying triangulation. create() must have been
    /// called first.
    Triangulation<dim> &
    get()
    {
      Assert(triangulation, ExcMessage("MeshHandler::create() not called."));
      return *triangulation;
    }
    const Triangulation<dim> &
    get() const
    {
      Assert(triangulation, ExcMessage("MeshHandler::create() not called."));
      return *triangulation;
    }

    /// @return whether the mesh consists only of quadrilateral (2D) /
    /// hexahedral (3D) cells.
    bool
    is_hex() const
    {
      return is_hex(get());
    }
    static bool
    is_hex(const Triangulation<dim> &tria)
    {
      const std::vector<ReferenceCell> &reference_cells =
        tria.get_reference_cells();
      Assert(reference_cells.size() == 1,
             ExcMessage("Mixed-cell-type meshes are not supported."));
      return reference_cells[0].is_hyper_cube();
    }

    /// @return whether the mesh consists only of triangular (2D) /
    /// tetrahedral (3D) cells.
    bool
    is_tet() const
    {
      return is_tet(get());
    }
    static bool
    is_tet(const Triangulation<dim> &tria)
    {
      const std::vector<ReferenceCell> &reference_cells =
        tria.get_reference_cells();
      Assert(reference_cells.size() == 1,
             ExcMessage("Mixed-cell-type meshes are not supported."));
      return reference_cells[0].is_simplex();
    }

    /// @return the standard Lagrange finite element space of the given
    /// @p degree for this mesh's cell type (FE_Q for hex, FE_SimplexP for
    /// tet).
    std::unique_ptr<FiniteElement<dim>>
    get_fe_lagrange(const unsigned int degree) const
    {
      if (is_hex())
        return std::make_unique<FE_Q<dim>>(degree);
      else
        return std::make_unique<FE_SimplexP<dim>>(degree);
    }

    /// @return the standard Gauss quadrature rule with @p n_points points
    /// per direction, for this mesh's cell type (QGauss for hex,
    /// QGaussSimplex for tet).
    template <unsigned int dim_quad = dim>
    std::unique_ptr<Quadrature<dim_quad>>
    get_quadrature_gauss(const unsigned int n_points) const
    {
      if (is_hex())
        return std::make_unique<QGauss<dim_quad>>(n_points);
      else
        return std::make_unique<QGaussSimplex<dim_quad>>(n_points);
    }

    /// @return a mapping of the given @p degree for this mesh's cell type
    /// (MappingQ for hex; MappingFE over an FE_SimplexP for tet, since
    /// MappingQ is only defined for hypercube reference cells).
    std::unique_ptr<Mapping<dim>>
    get_mapping(const unsigned int degree) const
    {
      if (is_hex())
        return std::make_unique<MappingQ<dim>>(degree);
      else
        // MappingFE clones the element it is given, so passing a temporary
        // is fine (and, unlike a function-local static, correct for any
        // degree, not just the first one requested).
        return std::make_unique<MappingFE<dim>>(FE_SimplexP<dim>(degree));
    }

    /// @return the (bi-/tri-)linear mapping for this mesh's cell type.
    /// Returned fresh on every call so callers can own their mapping.
    std::unique_ptr<Mapping<dim>>
    get_linear_mapping() const
    {
      return get_mapping(1);
    }

  private:
    MPI_Comm comm;
    std::unique_ptr<parallel::DistributedTriangulationBase<dim>> triangulation;
    types::boundary_id boundary_id_shift = 0;
  };
} // namespace internodes

#endif // INTERNODES_MESH_HANDLER_HPP
