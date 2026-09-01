#ifndef INTERNODES_MESH_HANDLER_HPP
#define INTERNODES_MESH_HANDLER_HPP

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/reference_cell.h>

#include "internodes/types.hpp"
#include "internodes/utilities.hpp"

#include <memory>
#include <vector>

namespace internodes
{
  using namespace dealii;

  /**
   * @brief Thin wrapper around a distributed triangulation for one
   * subdomain.
   *
   * Deliberately minimal replacement for lifex::utils::MeshHandler (which
   * additionally handles parameter-file-driven mesh generation/import).
   * Construction of the triangulation itself (via GridGenerator, GridIn, or
   * a subdivided/merged mesh) is left to the caller; MeshHandler here owns
   * the triangulation and provides the element-type-dependent finite
   * element, quadrature, and mapping objects used throughout the
   * interface/RBF DoF handlers and SubProblem -- ported directly from
   * lifex::utils::MeshHandler's own is_hex()/get_fe_lagrange()/
   * get_quadrature_gauss()/get_linear_mapping(), which all dispatch on
   * whether the mesh is hypercube (quad/hex) or simplex (tri/tet) cells.
   * Both cases occur in the paper's own tests (structured hexahedral
   * meshes, and hybrid tetrahedral-hexahedral couplings), so both are
   * supported here, not just the hex case.
   */
  class MeshHandler
  {
  public:
    explicit MeshHandler(const MPI_Comm &comm = internodes::mpi_comm)
      : triangulation(comm)
    {}

    parallel::distributed::Triangulation<dim> &
    get()
    {
      return triangulation;
    }
    const parallel::distributed::Triangulation<dim> &
    get() const
    {
      return triangulation;
    }

    /// @return whether the mesh consists only of quadrilateral (2D) /
    /// hexahedral (3D) cells.
    bool
    is_hex() const
    {
      return is_hex(triangulation);
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
      return is_tet(triangulation);
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
        {
          static const FE_SimplexP<dim> fe(degree);
          return std::make_unique<MappingFE<dim>>(fe);
        }
    }

    /// @return the (bi-/tri-)linear mapping for this mesh's cell type.
    /// Equivalent to get_mapping(1), returned fresh on every call so
    /// callers can own their mapping object independently.
    std::unique_ptr<Mapping<dim>>
    get_linear_mapping() const
    {
      return get_mapping(1);
    }

  private:
    parallel::distributed::Triangulation<dim> triangulation;
  };
} // namespace internodes

#endif // INTERNODES_MESH_HANDLER_HPP
