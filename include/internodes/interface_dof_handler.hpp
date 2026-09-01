#ifndef INTERNODES_INTERFACE_DOF_HANDLER_HPP
#define INTERNODES_INTERFACE_DOF_HANDLER_HPP

#include <deal.II/base/quadrature.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include "internodes/mesh_handler.hpp"
#include "internodes/types.hpp"
#include "internodes/utilities.hpp"

#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

namespace internodes
{
  using namespace dealii;

  /**
   * @brief Handler for the interface degrees of freedom of one subdomain,
   * for geometrically conforming interfaces (Lagrange interpolation).
   *
   * Given a subdomain's DoFHandler and the boundary id marking its interface
   * with the other subdomain, this class:
   * - separates the subdomain's DoFs into "interface" and "internal" sets,
   *   each with its own local numbering (used to extract/assemble the
   *   $A_{k,k}$, $A_{k,\overline\Gamma_k}$, etc. matrix blocks -- see the
   *   paper's Eq. for the block system);
   * - builds the interface mass matrix $M_{\Gamma_k}$;
   * - interpolates a vector of interface DoF values onto an arbitrary set of
   *   destination points on the interface of the *other* subdomain, via
   *   Lagrange interpolation -- this implements the $R_{12}$/$R_{21}$
   *   interpolation operators for geometrically conforming interfaces.
   *
   * See interface_dof_handler_rbf.hpp for the geometrically non-conforming
   * (RL-RBF interpolation) counterpart.
   */
  class InterfaceDoFHandler
  {
  public:
    InterfaceDoFHandler(
      const std::shared_ptr<const DoFHandler<dim>> &dof_handler,
      const std::shared_ptr<const MeshHandler>     &triangulation,
      const std::set<types::boundary_id>           &interface_id,
      const std::set<types::boundary_id>           &neumann_id,
      const std::set<types::boundary_id>           &dirichlet_id);

    InterfaceDoFHandler() = default;

    virtual ~InterfaceDoFHandler() = default;

    /// Locates, for each of the given points (assumed to lie on this
    /// subdomain's interface), the owning cell and the interface DoFs/shape
    /// function values needed to evaluate a finite element function there.
    /// Must be called before interpolate().
    virtual void
    setup_destination_points(const std::vector<Point<dim>> &points);

    /// Interpolates the finite element function with interface-DoF
    /// coefficients @p src onto the destination points passed to the last
    /// call of setup_destination_points(), filling @p dst (indexed by
    /// destination point number, not by DoF).
    virtual void
    interpolate(TrilinosWrappers::MPI::Vector       &dst,
                const TrilinosWrappers::MPI::Vector &src,
                const std::vector<Point<dim>>       &points) const;

    /// Assembles the interface mass matrix $M_{\Gamma_k}$, i.e.
    /// $(M_{\Gamma_k})_{ij} = \int_{\Gamma_k} \mu_j \mu_i \, d\sigma$ over
    /// the interface faces, in the interface-local DoF numbering.
    void
    interfaceMassMatrix(SparsityPattern              &sp,
                        TrilinosWrappers::SparseMatrix &mass_matrix,
                        Quadrature<dim - 1>            &face_quadrature_formula) const;

    /// @return the local (0-based) index, among the faces of @p cell, of
    /// the interface face containing vertex @p v; or `cell->n_faces()` if
    /// no interface face of @p cell contains @p v.
    unsigned int
    cellFace_index(const Point<dim>                            &v,
                  const DoFHandler<dim>::active_cell_iterator &cell) const;

    bool
    is_at_interface(unsigned int global_dof) const;

    /// Global-to-interface-local and interface-local-to-global DoF maps.
    /// @{
    types::global_dof_index
    interface_local_dof(unsigned int global_dof) const;
    types::global_dof_index
    interface_global_dof(unsigned int local_dof) const;
    types::global_dof_index
    internal_local_dof(unsigned int global_dof) const;
    types::global_dof_index
    internal_global_dof(unsigned int local_dof) const;
    /// @}

    const std::vector<Point<dim>> &
    support_points() const
    {
      return support_points_;
    }

    std::vector<Point<dim>>
    support_points_global() const
    {
      return support_points_global_;
    }

    IndexSet
    owned_dofs() const
    {
      return owned_dofs_;
    }
    IndexSet
    internal_dofs() const
    {
      return internal_dofs_;
    }
    IndexSet
    internal_dofs_global() const
    {
      return internal_dofs_total_indexSet;
    }
    IndexSet
    relevant_dofs() const
    {
      return relevant_dofs_;
    }
    IndexSet
    interface_dofs() const
    {
      return interface_dofs_;
    }
    IndexSet
    interface_dofs_global() const
    {
      return interface_dofs_total_indexSet;
    }
    /// Interface/internal DoF IndexSets in the *local-to-the-interface*
    /// numbering used for the block system -- these are what get passed to
    /// Trilinos vector/matrix reinit() calls.
    IndexSet
    internal_dofs_owned() const
    {
      return internal_parallelPartitioning;
    }
    IndexSet
    interface_dofs_owned() const
    {
      return interface_parallelPartitioning;
    }
    IndexSet
    internal_dofs_relevant() const
    {
      return internal_relevantParallelPartitioning;
    }
    IndexSet
    interface_dofs_relevant() const
    {
      return interface_relevantParallelPartitioning;
    }

    const std::shared_ptr<const DoFHandler<dim>> &
    dof_handler() const
    {
      return dof_handler_;
    }

    std::unique_ptr<Mapping<dim>>
    mapping() const
    {
      return triangulation_->get_linear_mapping();
    }

    const std::set<types::boundary_id> &
    interface_id() const
    {
      return interface_id_;
    }
    const std::set<types::boundary_id> &
    Neumann_id() const
    {
      return Neumann_id_;
    }
    const std::set<types::boundary_id> &
    Dirichlet_id() const
    {
      return Dirichlet_id_;
    }

  protected:
    std::shared_ptr<const DoFHandler<dim>> dof_handler_;
    std::shared_ptr<const MeshHandler>     triangulation_;

    IndexSet interface_dofs_;
    IndexSet interface_dofs_all_;
    IndexSet interface_parallelPartitioning;
    IndexSet interface_relevantParallelPartitioning;

    IndexSet internal_dofs_;
    IndexSet internal_parallelPartitioning;
    IndexSet internal_relevantParallelPartitioning;

    IndexSet owned_dofs_;
    IndexSet relevant_dofs_;

    std::vector<IndexSet> interface_dofs_total;
    std::vector<IndexSet> internal_dofs_total;
    IndexSet               interface_dofs_total_indexSet;
    IndexSet               internal_dofs_total_indexSet;

    std::vector<std::vector<Point<dim>>> support_points_total;
    std::vector<Point<dim>>              support_points_;
    std::vector<Point<dim>>              support_points_global_;
    std::vector<Point<dim>>              reference_points_;

    std::set<types::boundary_id> interface_id_;
    std::set<types::boundary_id> Neumann_id_;
    std::set<types::boundary_id> Dirichlet_id_;

    /// destination point index -> (interface-local DoF indices, shape
    /// function values) needed to evaluate a finite element function at
    /// that point.
    std::map<types::global_dof_index,
             std::pair<std::vector<types::global_dof_index>, std::vector<double>>>
      destination_points_map;

  private:
    void
    DoFs_values(const types::global_dof_index &i, Vector<double> &rhs_vector) const;
  };
} // namespace internodes

#endif // INTERNODES_INTERFACE_DOF_HANDLER_HPP
