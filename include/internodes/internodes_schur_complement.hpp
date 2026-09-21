#ifndef INTERNODES_INTERNODES_SCHUR_COMPLEMENT_HPP
#define INTERNODES_INTERNODES_SCHUR_COMPLEMENT_HPP

#include <deal.II/base/subscriptor.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include "internodes/multi_domain_problem.hpp"

#include <memory>

namespace internodes
{
  /**
   * @brief Implements the matrix-free preconditioned-GMRES solution of the
   * INTERNODES coupled problem via the Schur complement, exactly as
   * described in the paper's "Schur complement algorithm" section
   * (Algorithms 5-7): vmult() is the Schur operator's matrix-vector
   * product, solve() runs the full four-step procedure (Step 1: local
   * solves with zero interface data; Step 2: interface residual $\chi$;
   * Step 3: solve $S\lambda=\chi$; Step 4: local solves with the
   * interface solution as boundary data).
   *
   * Inherits from Subscriptor so it can be passed by reference to deal.II's
   * SolverGMRES as the matrix argument (only vmult() is required).
   */
  class InternodesSchurComplement : public Subscriptor
  {
  public:
    InternodesSchurComplement(const std::shared_ptr<MultiDomainProblem> &pb,
                              const SolverControl                        &solver_control)
      : sol_omega_master(pb->master->interface_dofHandler_ptr->internal_dofs_owned())
      , sol_omega_slave(pb->slave->interface_dofHandler_ptr->internal_dofs_owned())
      , lambda_master(pb->master->interface_dofHandler_ptr->interface_dofs_owned())
      , lambda_slave(pb->slave->interface_dofHandler_ptr->interface_dofs_owned())
      , res_master(pb->master->interface_dofHandler_ptr->interface_dofs_owned())
      , res_slave(pb->slave->interface_dofHandler_ptr->interface_dofs_owned())
      , res_master_internal(pb->master->interface_dofHandler_ptr->internal_dofs_owned())
      , res_slave_internal(pb->slave->interface_dofHandler_ptr->internal_dofs_owned())
      , problem(pb)
      , solver_control(solver_control)
      , linear_solver(this->solver_control)
    {}

    const TrilinosWrappers::MPI::Vector &
    sol_master() const
    {
      return sol_omega_master;
    }
    const TrilinosWrappers::MPI::Vector &
    sol_slave() const
    {
      return sol_omega_slave;
    }
    const TrilinosWrappers::MPI::Vector &
    lambda_master_() const
    {
      return lambda_master;
    }
    const TrilinosWrappers::MPI::Vector &
    lambda_slave_() const
    {
      return lambda_slave;
    }

    void
    set_lambda(TrilinosWrappers::MPI::Vector &l)
    {
      lambda_master = l;
      lambda_master.compress(VectorOperation::insert);
      problem->master->interface_dofHandler_ptr->interpolate(
        lambda_slave,
        lambda_master,
        problem->slave->interface_dofHandler_ptr->support_points_global());
    }

    unsigned int
    get_n_iterations()
    {
      return solver_control.last_step();
    }

    /// Enables (default) or disables the Dirichlet-Neumann Schur
    /// preconditioner in the interface GMRES solve. Disabling it (identity
    /// preconditioner) only serves to measure what the preconditioner buys.
    void
    set_use_schur_preconditioner(const bool use)
    {
      use_preconditioner = use;
    }

    /// Matrix-free application of the Schur complement operator $S$ (paper
    /// Algorithm 5): $S\mathbf p_{\overline\Gamma_1} \to \mathbf r$.
    void
    vmult(TrilinosWrappers::MPI::Vector &r, const TrilinosWrappers::MPI::Vector &l) const;

    /// Solves $A_{k,k}\mathbf u = \mathbf f_k -
    /// A_{k,\overline\Gamma_k}\mathbf l$ for the given subdomain (paper
    /// Eqs. \eqref{eq:localsystem_f}/\eqref{eq:localsystem_lambda},
    /// depending on whether the subdomain's forcing data is currently the
    /// real data or has been zeroed out -- see solve()).
    void
    solve_subproblem(const TrilinosWrappers::MPI::Vector &l,
                     TrilinosWrappers::MPI::Vector       &u,
                     TrilinosWrappers::MPI::Vector       &pb_rhs,
                     const std::shared_ptr<SubProblemBase> &pb) const;

    /// $\mathbf r_{\overline\Gamma_k} = A_{\overline\Gamma_k,k}\mathbf u +
    /// A_{\overline\Gamma_k,\overline\Gamma_k}\mathbf l -
    /// \mathbf f_{\overline\Gamma_k}$.
    void
    normalDerivative(const TrilinosWrappers::MPI::Vector   &l,
                     const TrilinosWrappers::MPI::Vector   &u,
                     TrilinosWrappers::MPI::Vector         &res,
                     const std::shared_ptr<SubProblemBase> &pb) const;

    /// Runs the full four-step INTERNODES solution procedure (paper's
    /// "Solution phase"), storing the internal solutions in
    /// sol_omega_master/slave and the interface solution in
    /// lambda_master/slave.
    void
    solve();

    /// $\mathbf r = \mathbf r_{\overline\Gamma_1} +
    /// Q_{12}\mathbf r_{\overline\Gamma_2}$, where
    /// $\mathbf r_{\overline\Gamma_k}$ is normalDerivative() of solution
    /// @p s_1 / @p s_2 on the master/slave.
    void
    ComputeResidual(TrilinosWrappers::MPI::Vector &r,
                    TrilinosWrappers::MPI::Vector   s_1,
                    TrilinosWrappers::MPI::Vector   s_2);

    // -- Named steps of solve(), broken out individually for timing
    // (TimerOutput::Scope) purposes -- mirrors the paper's Algorithm 7
    // structure. --
    void
    step0(bool intermediate = false);
    void
    stepP();
    void
    step_rhs(TrilinosWrappers::MPI::Vector       &rhs_temp,
             const TrilinosWrappers::MPI::Vector &lambda_temp,
             const TrilinosWrappers::MPI::Vector &rhs_in_temp,
             const TrilinosWrappers::SparseMatrix &M_in_temp) const;
    void
    step_linear_solver_subproblem(
      SolverCG<TrilinosWrappers::MPI::Vector>   &solver,
      const TrilinosWrappers::SparseMatrix      &M,
      TrilinosWrappers::MPI::Vector             &sol,
      const TrilinosWrappers::MPI::Vector       &rhs,
      const TrilinosWrappers::PreconditionAMG   &preconditioner) const;
    void
    step1(TrilinosWrappers::MPI::Vector &uf_master,
          TrilinosWrappers::MPI::Vector &uf_slave,
          TrilinosWrappers::MPI::Vector &pb_rhs_master,
          TrilinosWrappers::MPI::Vector &pb_rhs_slave);
    void
    step2(TrilinosWrappers::MPI::Vector &chi,
          TrilinosWrappers::MPI::Vector &residual,
          TrilinosWrappers::MPI::Vector &uf_master,
          TrilinosWrappers::MPI::Vector &uf_slave);
    void
    step3(TrilinosWrappers::MPI::Vector &lambda_, TrilinosWrappers::MPI::Vector &chi);
    void
    step4(TrilinosWrappers::MPI::Vector &ulambda_master,
          TrilinosWrappers::MPI::Vector &ulambda_slave,
          TrilinosWrappers::MPI::Vector &pb_rhs_master,
          TrilinosWrappers::MPI::Vector &pb_rhs_slave);

    friend class SchurPreconditioner;

  private:
    mutable TrilinosWrappers::MPI::Vector sol_omega_master;
    mutable TrilinosWrappers::MPI::Vector sol_omega_slave;
    mutable TrilinosWrappers::MPI::Vector lambda_master;
    mutable TrilinosWrappers::MPI::Vector lambda_slave;
    mutable TrilinosWrappers::MPI::Vector res_master;
    mutable TrilinosWrappers::MPI::Vector res_slave;
    mutable TrilinosWrappers::MPI::Vector res_master_internal;
    mutable TrilinosWrappers::MPI::Vector res_slave_internal;

    std::shared_ptr<MultiDomainProblem> problem;

    SolverControl                            solver_control;
    SolverGMRES<TrilinosWrappers::MPI::Vector> linear_solver;
    bool                                     use_preconditioner = true;
  };
} // namespace internodes

#endif // INTERNODES_INTERNODES_SCHUR_COMPLEMENT_HPP
