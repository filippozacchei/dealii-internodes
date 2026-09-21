#ifndef INTERNODES_SUB_PROBLEM_BASE_HPP
#define INTERNODES_SUB_PROBLEM_BASE_HPP

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include "internodes/interface_dof_handler.hpp"
#include "internodes/interface_dof_handler_rbf.hpp"
#include "internodes/mesh_handler.hpp"
#include "internodes/types.hpp"

#include <map>
#include <memory>
#include <set>
#include <string>

namespace internodes
{
  using namespace dealii;

  /**
   * @brief One subdomain of a two-subdomain INTERNODES coupled problem.
   *
   * Together with MultiDomainProblem, this class holds everything needed to
   * discretize and assemble one subdomain's Galerkin problem and to extract
   * the algebraic blocks ($A_{k,k}$, $A_{k,\overline\Gamma_k}$, etc. --
   * see the paper's block system) needed for the Schur-complement coupling.
   * SubProblemBase is the abstract base: it owns the mesh, finite element
   * space, DoFHandler, interface handler, and assembly/BC machinery common
   * to any PDE; the bilinear form itself (localBilinearForm()) is supplied
   * by a concrete subclass -- see sub_problem_diffusion_reaction.hpp for the
   * one used in the paper's numerical results ($Lu = -\Delta u + u$).
   */
  class SubProblemBase
  {
  public:
    SubProblemBase() = default;

    /**
     * @param mesh        subdomain triangulation, already populated (via
     *                     GridGenerator/GridIn) by the caller.
     * @param fe_degree    Lagrange finite element degree.
     * @param dirichlet_fun  Dirichlet boundary data.
     * @param neumann_fun    Neumann boundary data (its gradient is used, per
     *                       the co-normal derivative -- see assembly()).
     * @param forcing_term   forcing term $f$.
     * @param boundary_tags  map with keys "Dirichlet", "Neumann",
     *                       "Interface", each giving the set of boundary ids
     *                       with that role.
     * @param coefficients   named PDE coefficients (used by the concrete
     *                       subclass's localBilinearForm()).
     * @param rbf_radius     if > 0, use RL-RBF interpolation with this
     *                       support radius for the interface handler
     *                       instead of Lagrange interpolation.
     * @param rbf_mode       RBF kernel, if rbf_radius > 0.
     */
    SubProblemBase(
      const std::shared_ptr<MeshHandler>                         &mesh,
      unsigned int                                                 fe_degree,
      const std::shared_ptr<Function<dim>>                        &dirichlet_fun,
      const std::shared_ptr<Function<dim>>                        &neumann_fun,
      const std::shared_ptr<Function<dim>>                        &forcing_term,
      const std::map<std::string, std::set<types::boundary_id>>  &boundary_tags,
      const std::map<std::string, double>                         &coefficients,
      double                                                        rbf_radius = 0.,
      InterfaceDoFHandlerRBF::Mode rbf_mode = InterfaceDoFHandlerRBF::Mode::Wendland)
      : slice(mesh)
      , fe_degree(fe_degree)
      , dof_handler(std::make_shared<DoFHandler<dim>>())
      // Translate the ids of the mesh handed to MeshHandler::create() into
      // the ids stored in the distributed triangulation (see
      // MeshHandler::to_internal_boundary_id()).
      , dirichlet_ids(mesh->to_internal_boundary_ids(boundary_tags.at("Dirichlet")))
      , neumann_ids(mesh->to_internal_boundary_ids(boundary_tags.at("Neumann")))
      , interface_id(mesh->to_internal_boundary_ids(boundary_tags.at("Interface")))
      , fun(dirichlet_fun)
      , fun_neumann(neumann_fun)
      , forcing_term(forcing_term)
      , coefficients(coefficients)
    {
      setupSystem(rbf_radius, rbf_mode);
    }

    virtual ~SubProblemBase() = default;

    /// Local bilinear form $a_k(\phi_j, \phi_i)$ at quadrature point @p q,
    /// supplied by a concrete PDE model.
    virtual double
    localBilinearForm(const FEValues<dim> &fe_values,
                      unsigned int         i,
                      unsigned int         j,
                      unsigned int         q) = 0;

    /// Assembles the full local matrix/rhs and splits them into the
    /// internal/interface blocks ($A_{k,k}$, $A_{k,\overline\Gamma_k}$,
    /// $A_{\overline\Gamma_k,k}$, $A_{\overline\Gamma_k,\overline\Gamma_k}$,
    /// $\mathbf f_k$, $\mathbf f_{\overline\Gamma_k}$) used by the Schur
    /// complement solver. @p intermediate is unused here (kept for
    /// interface parity with the original -- some subclasses may use it to
    /// skip recomputation of state that only depends on the mesh, not on
    /// the current forcing/BC data, when called repeatedly with zeroed data
    /// during the Schur mat-vec).
    virtual void
    assembly_global(bool intermediate = false);

    /// (Re-)initializes the algebraic multigrid preconditioner for the
    /// internal-internal block $A_{k,k}$.
    void
    assembly_preconditioner();

    /// Sets the Dirichlet-constrained entries of @p internal_solution -- a
    /// vector indexed by the *internal-block-local* numbering, as returned
    /// by the block solves -- to their prescribed values.
    ///
    /// This replaces `constraints_dirichlet.distribute()`: that function
    /// requires a vector in the DoFHandler's own (full) numbering, and only
    /// appeared to work on internal-block vectors in older deal.II because
    /// it did not check the size, relying on the two numberings happening
    /// to coincide (true in serial, not in general). Since the constraints
    /// here are pure Dirichlet (no constraint entries / hanging nodes),
    /// distribute() amounts to exactly this assignment.
    void
    apply_dirichlet_to_internal(TrilinosWrappers::MPI::Vector &internal_solution) const;

    /// Replaces the forcing term and boundary data (used by
    /// InternodesSchurComplement to zero out the problem data before
    /// applying the homogeneous Schur complement operator, then restore it
    /// afterwards).
    void
    set_data(const std::shared_ptr<Function<dim>> &new_forcing_term,
             const std::shared_ptr<Function<dim>> &new_dirichlet,
             const std::shared_ptr<Function<dim>> &new_neumann)
    {
      forcing_term = new_forcing_term;
      fun          = new_dirichlet;
      fun_neumann  = new_neumann;
    }

    // -- Block accessors: A_{internal,internal}, A_{internal,interface},
    // A_{interface,internal}, A_{interface,interface} --
    TrilinosWrappers::SparseMatrix &
    M_in_in()
    {
      return matrix.block(0, 0);
    }
    TrilinosWrappers::SparseMatrix &
    M_in_gamma()
    {
      return matrix.block(0, 1);
    }
    TrilinosWrappers::SparseMatrix &
    M_gamma_in()
    {
      return matrix.block(1, 0);
    }
    TrilinosWrappers::SparseMatrix &
    M_gamma_gamma()
    {
      return matrix.block(1, 1);
    }

    /// Triangulation (owned jointly with whoever else holds a reference,
    /// e.g. the caller who built it).
    std::shared_ptr<MeshHandler> slice;

    unsigned int                          fe_degree;
    std::unique_ptr<FiniteElement<dim>>   fe;
    std::unique_ptr<Quadrature<dim>>      quadrature_formula;
    std::unique_ptr<Quadrature<dim - 1>>  face_quadrature_formula;
    std::shared_ptr<DoFHandler<dim>>      dof_handler;

    std::set<types::boundary_id> dirichlet_ids;
    std::set<types::boundary_id> neumann_ids;
    std::set<types::boundary_id> interface_id;

    /// Interface DoF handler -- either InterfaceDoFHandler (Lagrange) or
    /// InterfaceDoFHandlerRBF (RL-RBF), chosen in setupSystem() based on
    /// whether an RBF radius was given.
    std::shared_ptr<InterfaceDoFHandler> interface_dofHandler_ptr;

    /// Interface mass matrix $M_{\Gamma_k}$.
    TrilinosWrappers::SparseMatrix M_gamma;
    SparsityPattern                 sp_M_gamma;

    std::shared_ptr<Function<dim>> fun;          ///< Dirichlet data.
    std::shared_ptr<Function<dim>> fun_neumann;  ///< Neumann data.
    std::shared_ptr<Function<dim>> forcing_term; ///< Forcing term $f$.

    std::map<std::string, double> coefficients;

    AffineConstraints<double> constraints_dirichlet;

    TrilinosWrappers::PreconditionAMG preconditioner_in_in;

    BlockSparsityPattern              sp;
    TrilinosWrappers::BlockSparseMatrix matrix;

    IndexSet owned_dofs;
    IndexSet relevant_dofs;

    TrilinosWrappers::MPI::Vector      rhs_gamma; ///< $\mathbf f_{\overline\Gamma_k}$.
    TrilinosWrappers::MPI::Vector      rhs_in;    ///< $\mathbf f_k$.
    TrilinosWrappers::MPI::BlockVector rhs;

  private:
    void
    setupSystem(double radius, InterfaceDoFHandlerRBF::Mode mode);

    /// Assembles the (unsplit) local matrix/rhs over the whole subdomain --
    /// see resolutionAlgorithm.tex's Algorithm 2 in the paper for the exact
    /// quadrature-loop structure this follows.
    virtual void
    assembly(bool intermediate = false);
  };
} // namespace internodes

#endif // INTERNODES_SUB_PROBLEM_BASE_HPP
