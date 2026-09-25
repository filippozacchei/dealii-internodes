#include "internodes/multi_domain_problem.hpp"

#include <deal.II/lac/solver_cg.h>

namespace internodes
{
  void
  MultiDomainProblem::setup_system()
  {
    size_internal_slave =
      slave->interface_dofHandler_ptr->internal_dofs_owned().n_elements();
    size_internal_master =
      master->interface_dofHandler_ptr->internal_dofs_owned().n_elements();
    size_interface_slave =
      slave->interface_dofHandler_ptr->interface_dofs_owned().n_elements();
    size_interface_master =
      master->interface_dofHandler_ptr->interface_dofs_owned().n_elements();

    // Each handler interpolates onto the *other* subdomain's interface, so
    // it needs the interpolation weights of the destination points owned by
    // this rank on that side only.
    slave->interface_dofHandler_ptr->setup_destination_points(
      master->interface_dofHandler_ptr->support_points_global(),
      master->interface_dofHandler_ptr->interface_dofs_owned());
    master->interface_dofHandler_ptr->setup_destination_points(
      slave->interface_dofHandler_ptr->support_points_global(),
      slave->interface_dofHandler_ptr->interface_dofs_owned());
  }

  void
  MultiDomainProblem::interpolateResidual(TrilinosWrappers::MPI::Vector       &dst,
                                          const TrilinosWrappers::MPI::Vector &src) const
  {
    residual_slave  = 0;
    residual_master = 0;

    const unsigned int max_iters = 1000000;
    ReductionControl control(max_iters,
                             tolerances.interface_mass.tolerance,
                             tolerances.interface_mass.reduction,
                             false,
                             false);
    SolverCG<TrilinosWrappers::MPI::Vector> solver(control);
    {
      TimerOutput::Scope timer_section(timer_output(), "  interpolateResidual: solve M_gamma");
      solver.solve(slave->M_gamma, residual_slave, src, precond);
    }
    record_cg_solve("interface_mass", control.last_step());

    slave->interface_dofHandler_ptr->interpolate(residual_master,
                                                  residual_slave,
                                                  point_to_interpolate);

    master->M_gamma.vmult(dst, residual_master);
  }
} // namespace internodes
