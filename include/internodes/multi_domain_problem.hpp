#ifndef INTERNODES_MULTI_DOMAIN_PROBLEM_HPP
#define INTERNODES_MULTI_DOMAIN_PROBLEM_HPP

#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_vector.h>

#include "internodes/sub_problem_base.hpp"

#include <memory>
#include <vector>

namespace internodes
{
  /**
   * @brief Couples a master and a slave SubProblem into one INTERNODES
   * problem, and implements the intergrid ($Q_{21}$) and residual-transfer
   * ($Q_{12}$) operators between them.
   *
   * Per the paper's convention: the Dirichlet trace is interpolated from
   * the master interface to the slave ($Q_{21}$, via
   * InterfaceDoFHandler::interpolate() called directly on the master's
   * handler elsewhere -- see InternodesSchurComplement), while the residual
   * is interpolated from slave to master ($Q_{12} = M_{\Gamma_1} R_{12}
   * M_{\Gamma_2}^{-1}$, implemented here in interpolateResidual()).
   */
  class MultiDomainProblem
  {
  public:
    MultiDomainProblem() = default;

    MultiDomainProblem(const std::shared_ptr<SubProblemBase> &master,
                       const std::shared_ptr<SubProblemBase> &slave)
      : master(master)
      , slave(slave)
      , point_to_interpolate(master->interface_dofHandler_ptr->support_points_global())
      , residual_master(master->interface_dofHandler_ptr->interface_dofs_owned())
      , residual_slave(slave->interface_dofHandler_ptr->interface_dofs_owned())
    {
      setup_system();
    }

    void
    set_data(const std::shared_ptr<Function<dim>> &new_forcing_term,
             const std::shared_ptr<Function<dim>> &new_dirichlet,
             const std::shared_ptr<Function<dim>> &new_neumann)
    {
      master->set_data(new_forcing_term, new_dirichlet, new_neumann);
      slave->set_data(new_forcing_term, new_dirichlet, new_neumann);
    }

    /// $\mathbf{dst} = Q_{12}\,\mathbf{src} = M_{\Gamma_1} R_{12}
    /// M_{\Gamma_2}^{-1}\,\mathbf{src}$: interpolates a residual vector
    /// living on the slave interface onto the master interface.
    void
    interpolateResidual(TrilinosWrappers::MPI::Vector       &dst,
                        const TrilinosWrappers::MPI::Vector &src) const;

    std::shared_ptr<SubProblemBase> master;
    std::shared_ptr<SubProblemBase> slave;

    unsigned int size_internal_slave   = 0;
    unsigned int size_internal_master  = 0;
    unsigned int size_interface_slave  = 0;
    unsigned int size_interface_master = 0;

    /// Master interface support points, cached for repeated use as the
    /// destination points of the slave->master interpolation.
    std::vector<Point<dim>> point_to_interpolate;

    mutable TrilinosWrappers::MPI::Vector residual_master;
    mutable TrilinosWrappers::MPI::Vector residual_slave;

    /// Preconditioner for the slave's interface mass matrix $M_{\Gamma_2}$,
    /// used to apply $M_{\Gamma_2}^{-1}$ in interpolateResidual().
    /// Initialized externally by InternodesSchurComplement::stepP().
    ///
    /// Jacobi (diagonal) is used deliberately rather than algebraic
    /// multigrid: a mass matrix is spectrally equivalent to its diagonal
    /// (the diagonally scaled condition number is bounded independently of
    /// the mesh size), so Jacobi-preconditioned CG converges in a few
    /// iterations, whereas AMG with a direct (Amesos-KLU) coarse solver --
    /// what the original lifex-based code used here -- proved fragile on
    /// small parallel interface matrices (NumericFactorization failures on
    /// tetrahedral meshes).
    TrilinosWrappers::PreconditionJacobi precond;

  private:
    void
    setup_system();
  };
} // namespace internodes

#endif // INTERNODES_MULTI_DOMAIN_PROBLEM_HPP
