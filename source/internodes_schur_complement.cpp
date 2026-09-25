#include "internodes/internodes_schur_complement.hpp"
#include "internodes/schur_preconditioner.hpp"

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

namespace internodes
{
  void
  InternodesSchurComplement::step0(bool intermediate)
  {
    TimerOutput::Scope timer_section(timer_output(), "Step 0: assembly (master/slave)");
    problem->master->assembly_global(intermediate);
    problem->slave->assembly_global(intermediate);
  }

  void
  InternodesSchurComplement::stepP()
  {
    TimerOutput::Scope timer_section(timer_output(), "Step 0: assembly (preconditioners)");
    problem->master->assembly_preconditioner();
    problem->slave->assembly_preconditioner();
    problem->precond.initialize(problem->slave->M_gamma);
  }

  void
  InternodesSchurComplement::step_rhs(TrilinosWrappers::MPI::Vector        &rhs_temp,
                                      const TrilinosWrappers::MPI::Vector  &lambda_temp,
                                      const TrilinosWrappers::MPI::Vector  &rhs_in_temp,
                                      const TrilinosWrappers::SparseMatrix &M_in_temp) const
  {
    TimerOutput::Scope timer_section(timer_output(), "step_rhs: f_k - A_{k,Gamma_k} lambda_k");
    M_in_temp.residual(rhs_temp, lambda_temp, rhs_in_temp);
  }

  void
  InternodesSchurComplement::step_linear_solver_subproblem(
    SolverCG<TrilinosWrappers::MPI::Vector> &solver,
    const TrilinosWrappers::SparseMatrix    &M,
    TrilinosWrappers::MPI::Vector           &sol,
    const TrilinosWrappers::MPI::Vector     &rhs,
    const TrilinosWrappers::PreconditionAMG &preconditioner) const
  {
    TimerOutput::Scope timer_section(timer_output(), "subdomain linear solve");
    solver.solve(M, sol, rhs, preconditioner);
  }

  void
  InternodesSchurComplement::step1(TrilinosWrappers::MPI::Vector &uf_master,
                                   TrilinosWrappers::MPI::Vector &uf_slave,
                                   TrilinosWrappers::MPI::Vector &pb_rhs_master,
                                   TrilinosWrappers::MPI::Vector &pb_rhs_slave)
  {
    TimerOutput::Scope timer_section(timer_output(), "Step 1: u^(f) on master/slave");
    solve_subproblem(lambda_master, uf_master, pb_rhs_master, problem->master);
    solve_subproblem(lambda_slave, uf_slave, pb_rhs_slave, problem->slave);
  }

  void
  InternodesSchurComplement::step2(TrilinosWrappers::MPI::Vector &chi,
                                   TrilinosWrappers::MPI::Vector &residual,
                                   TrilinosWrappers::MPI::Vector &uf_master,
                                   TrilinosWrappers::MPI::Vector &uf_slave)
  {
    TimerOutput::Scope timer_section(timer_output(), "Step 2: interface residual chi");
    ComputeResidual(residual, uf_master, uf_slave);
    chi -= residual;
  }

  void
  InternodesSchurComplement::step3(TrilinosWrappers::MPI::Vector &lambda_,
                                   TrilinosWrappers::MPI::Vector &chi)
  {
    if (!use_preconditioner)
      {
        TimerOutput::Scope timer_section(timer_output(), "Step 3b: solve S lambda = chi");
        linear_solver.solve(*this, lambda_, chi, PreconditionIdentity());
        return;
      }

    SchurPreconditioner preconditioner_schur;
    {
      TimerOutput::Scope timer_section(timer_output(), "Step 3a: initialize Schur preconditioner");
      preconditioner_schur.initialize(this->problem);
    }
    preconditioner_schur.assembly();
    {
      TimerOutput::Scope timer_section(timer_output(), "Step 3b: solve S lambda = chi");
      linear_solver.solve(*this, lambda_, chi, preconditioner_schur);
    }
  }

  void
  InternodesSchurComplement::step4(TrilinosWrappers::MPI::Vector &ulambda_master,
                                   TrilinosWrappers::MPI::Vector &ulambda_slave,
                                   TrilinosWrappers::MPI::Vector &pb_rhs_master,
                                   TrilinosWrappers::MPI::Vector &pb_rhs_slave)
  {
    TimerOutput::Scope timer_section(timer_output(), "Step 4: u^(lambda) on master/slave");
    solve_subproblem(lambda_master, ulambda_master, pb_rhs_master, problem->master);
    solve_subproblem(lambda_slave, ulambda_slave, pb_rhs_slave, problem->slave);
  }

  void
  InternodesSchurComplement::solve()
  {
    pcout() << "Step 0: assembling system matrices" << std::endl;
    lambda_master = 0.;
    lambda_slave  = 0.;

    this->step0();
    this->stepP();

    pcout() << "Step 1: solving for u^(f)" << std::endl;
    TrilinosWrappers::MPI::Vector uf_master(
      problem->master->interface_dofHandler_ptr->internal_dofs_owned());
    TrilinosWrappers::MPI::Vector uf_slave(
      problem->slave->interface_dofHandler_ptr->internal_dofs_owned());
    this->step1(uf_master, uf_slave, res_master_internal, res_slave_internal);

    pcout() << "Step 2: computing interface residual chi" << std::endl;
    TrilinosWrappers::MPI::Vector chi(
      problem->master->interface_dofHandler_ptr->interface_dofs_owned());
    TrilinosWrappers::MPI::Vector residual(
      problem->master->interface_dofHandler_ptr->interface_dofs_owned());
    step2(chi, residual, uf_master, uf_slave);

    pcout() << "Step 3: solving S*lambda = chi" << std::endl;
    // Zero out the problem data before applying the homogeneous Schur
    // complement operator (see this class's vmult(), which relies on
    // solve_subproblem() picking up whatever forcing/BC data is currently
    // set on `problem` -- see paper Eq. \eqref{eq:localsystem_lambda}: the
    // Step-3 matrix-vector product uses A_{k,k}u = -A_{k,Gamma_k}l, with
    // *zero* forcing, unlike Steps 1/4 which use the real data).
    //
    // The matrices do not depend on the data (only the right-hand sides and
    // the values of the Dirichlet constraints do), so they are not assembled
    // again: the constraints are rebuilt with the zero data and the
    // right-hand sides are set to zero, which is exactly what a zero-data
    // assembly produces. (The algorithm used to assemble the matrices three
    // times per solve, once here and once more to restore the data below.)
    const std::shared_ptr<Function<dim>> rhs_real       = problem->master->forcing_term;
    const std::shared_ptr<Function<dim>> dirichlet_real = problem->master->fun;
    const std::shared_ptr<Function<dim>> neumann_real   = problem->master->fun_neumann;

    const auto null_rhs = std::make_shared<Functions::ZeroFunction<dim>>();
    const auto null_dirichlet = std::make_shared<Functions::ZeroFunction<dim>>();
    const auto null_neumann = std::make_shared<Functions::ZeroFunction<dim>>();

    // Right-hand sides with the real data, put back at the end.
    const TrilinosWrappers::MPI::BlockVector master_rhs_real       = problem->master->rhs;
    const TrilinosWrappers::MPI::Vector      master_rhs_in_real    = problem->master->rhs_in;
    const TrilinosWrappers::MPI::Vector      master_rhs_gamma_real = problem->master->rhs_gamma;
    const TrilinosWrappers::MPI::BlockVector slave_rhs_real        = problem->slave->rhs;
    const TrilinosWrappers::MPI::Vector      slave_rhs_in_real     = problem->slave->rhs_in;
    const TrilinosWrappers::MPI::Vector      slave_rhs_gamma_real  = problem->slave->rhs_gamma;

    {
      TimerOutput::Scope timer_section(timer_output(),
                                       "Step 0: homogeneous-phase data update");
      problem->set_data(null_rhs, null_dirichlet, null_neumann);
      problem->master->update_constraints();
      problem->slave->update_constraints();
      problem->master->set_zero_rhs();
      problem->slave->set_zero_rhs();
    }

    TrilinosWrappers::MPI::Vector lambda_(
      problem->master->interface_dofHandler_ptr->interface_dofs_owned());
    {
      TimerOutput::Scope timer_section(timer_output(), "Step 3: interface solve (total)");
      step3(lambda_, chi);
    }
    this->set_lambda(lambda_);

    pcout() << "Step 4: solving for u^(lambda)" << std::endl;
    TrilinosWrappers::MPI::Vector ulambda_master(
      problem->master->interface_dofHandler_ptr->internal_dofs_owned());
    TrilinosWrappers::MPI::Vector ulambda_slave(
      problem->slave->interface_dofHandler_ptr->internal_dofs_owned());
    step4(ulambda_master, ulambda_slave, res_master_internal, res_slave_internal);

    sol_omega_master = 0.;
    sol_omega_slave  = 0.;
    sol_omega_master.add(1.0, uf_master, 1.0, ulambda_master);
    sol_omega_slave.add(1.0, uf_slave, 1.0, ulambda_slave);

    // Restore the real problem data, constraints and right-hand sides,
    // leaving `problem` in the same state solve() found it in.
    {
      TimerOutput::Scope timer_section(timer_output(),
                                       "Step 0: homogeneous-phase data update");
      problem->set_data(rhs_real, dirichlet_real, neumann_real);
      problem->master->update_constraints();
      problem->slave->update_constraints();
      problem->master->rhs       = master_rhs_real;
      problem->master->rhs_in    = master_rhs_in_real;
      problem->master->rhs_gamma = master_rhs_gamma_real;
      problem->slave->rhs        = slave_rhs_real;
      problem->slave->rhs_in     = slave_rhs_in_real;
      problem->slave->rhs_gamma  = slave_rhs_gamma_real;
    }
  }

  void
  InternodesSchurComplement::ComputeResidual(TrilinosWrappers::MPI::Vector &residual,
                                             TrilinosWrappers::MPI::Vector  s_master,
                                             TrilinosWrappers::MPI::Vector  s_slave)
  {
    this->normalDerivative(lambda_master, s_master, res_master, problem->master);
    this->normalDerivative(lambda_slave, s_slave, res_slave, problem->slave);

    problem->interpolateResidual(residual, res_slave);
    residual.add(1.0, res_master);
  }

  void
  InternodesSchurComplement::vmult(TrilinosWrappers::MPI::Vector       &residual_,
                                   const TrilinosWrappers::MPI::Vector &lambda_) const
  {
    // One call = one Schur mat-vec (Algorithm "Matrix-Vector Product for the
    // Schur Complement System" in the paper), i.e. one GMRES iteration of
    // Step 3. Timed in three pieces matching that algorithm's own steps
    // (Q21 interpolation; the two subdomain solves A_{k,k} u_k = ...; the
    // residual r_{Gamma_k} and its Q12 transfer), so that Step 3's own cost
    // -- as opposed to the same operations' cost when they occur in Steps
    // 1/2/4 -- can be broken down into "solving the subdomain systems" vs.
    // "computing/transferring the residual", the split asserted qualitatively
    // just above Algorithm~\ref{algo:schur-mvp} in the paper.
    {
      TimerOutput::Scope timer_section(timer_output(), "Step 3 (GMRES): Q21 interpolation");
      problem->master->interface_dofHandler_ptr->interpolate(
        lambda_slave,
        lambda_,
        problem->slave->interface_dofHandler_ptr->support_points_global());
    }

    {
      TimerOutput::Scope timer_section(timer_output(), "Step 3 (GMRES): subdomain solves");
      solve_subproblem(lambda_, sol_omega_master, res_master_internal, problem->master);
      solve_subproblem(lambda_slave, sol_omega_slave, res_slave_internal, problem->slave);
    }

    {
      TimerOutput::Scope timer_section(timer_output(), "Step 3 (GMRES): residual computation");
      normalDerivative(lambda_, sol_omega_master, res_master, problem->master);
      normalDerivative(lambda_slave, sol_omega_slave, res_slave, problem->slave);

      problem->interpolateResidual(residual_, res_slave);
      residual_.add(1.0, res_master);
    }
  }

  void
  InternodesSchurComplement::solve_subproblem(
    const TrilinosWrappers::MPI::Vector   &lambda_,
    TrilinosWrappers::MPI::Vector         &solution_,
    TrilinosWrappers::MPI::Vector         &pb_rhs,
    const std::shared_ptr<SubProblemBase> &subPb) const
  {
    step_rhs(pb_rhs, lambda_, subPb->rhs_in, subPb->M_in_gamma());

    // The default tolerances (InnerSolverTolerances) are near machine
    // precision, far tighter than the outer GMRES's 1e-8: they can be
    // relaxed through MultiDomainProblem::set_solver_tolerances().
    const CGTolerance &tolerances = problem->tolerances.subdomain;
    const unsigned int max_iterations = 1000000;
    ReductionControl control(max_iterations,
                             tolerances.tolerance,
                             tolerances.reduction,
                             false,
                             false);

    SolverCG<TrilinosWrappers::MPI::Vector> linear_solver_subproblem(control);
    step_linear_solver_subproblem(linear_solver_subproblem,
                                  subPb->M_in_in(),
                                  solution_,
                                  pb_rhs,
                                  subPb->preconditioner_in_in);
    record_cg_solve("subdomain", control.last_step());

    subPb->apply_dirichlet_to_internal(solution_);
  }

  void
  InternodesSchurComplement::normalDerivative(
    const TrilinosWrappers::MPI::Vector   &lambda_,
    const TrilinosWrappers::MPI::Vector   &solution_,
    TrilinosWrappers::MPI::Vector         &res_,
    const std::shared_ptr<SubProblemBase> &subPb) const
  {
    TimerOutput::Scope timer_section(timer_output(), "  normal derivative");
    res_ = 0.;
    res_ -= subPb->rhs_gamma;
    subPb->M_gamma_gamma().vmult_add(res_, lambda_);
    subPb->M_gamma_in().vmult_add(res_, solution_);
  }
} // namespace internodes
