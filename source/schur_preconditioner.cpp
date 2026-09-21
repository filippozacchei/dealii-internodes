#include "internodes/schur_preconditioner.hpp"

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparsity_tools.h>

namespace internodes
{
  void
  SchurPreconditioner::initialize(const std::shared_ptr<MultiDomainProblem> &pb)
  {
    this->problem = pb;

    auto       &dof_handler   = *(problem->master->dof_handler);
    auto       &owned_dofs    = problem->master->owned_dofs;
    auto       &relevant_dofs = problem->master->relevant_dofs;

    relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

    DynamicSparsityPattern dsp(relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                                owned_dofs,
                                                mpi_comm,
                                                relevant_dofs);

    matrix.reinit(owned_dofs, dsp, mpi_comm);
    rhs.reinit(owned_dofs, mpi_comm);
    dst_global.reinit(owned_dofs, mpi_comm);

    const auto &interface_dofs = problem->master->interface_dofHandler_ptr->interface_dofs();
    const auto &owned_interface_dofs =
      problem->master->interface_dofHandler_ptr->interface_dofs_owned();

    for (auto it = interface_dofs.begin(); it != interface_dofs.end(); ++it)
      interface_indices.push_back(*it);
    for (auto it = owned_interface_dofs.begin(); it != owned_interface_dofs.end(); ++it)
      owned_indices.push_back(*it);
  }

  void
  SchurPreconditioner::vmult(TrilinosWrappers::MPI::Vector       &dst,
                             const TrilinosWrappers::MPI::Vector &src) const
  {
    for (unsigned int i = 0; i < owned_indices.size(); ++i)
      rhs[interface_indices[i]] = src[owned_indices[i]];
    rhs.compress(VectorOperation::insert);

    const double       tolerance = 1e-13;
    const double       reduction = 1e-11;
    const unsigned int max_iters = 1000000;
    ReductionControl control(max_iters, tolerance, reduction, false, false);
    SolverCG<TrilinosWrappers::MPI::Vector> solver(control);
    solver.solve(matrix, dst_global, rhs, preconditioner_problem);

    for (unsigned int i = 0; i < owned_indices.size(); ++i)
      dst[owned_indices[i]] = dst_global[interface_indices[i]];
  }

  void
  SchurPreconditioner::assembly()
  {
    TimerOutput::Scope timer_section(timer_output(), "Assembly Preconditioner");

    matrix = 0;
    rhs    = 0;

    const auto        &dof_handler = *(problem->master->interface_dofHandler_ptr->dof_handler());
    const auto        &fe          = dof_handler.get_fe();
    const auto        &quadrature  = *(problem->master->quadrature_formula);
    const auto          mapping     = problem->master->interface_dofHandler_ptr->mapping();

    FEValues<dim> fe_values(*mapping,
                            fe,
                            quadrature,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        cell->get_dof_indices(dof_indices);
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs    = 0;

        for (unsigned int q = 0; q < n_q_points; ++q)
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              // Same bilinear form as the master subproblem's own primal
              // problem, as the paper specifies -- see this class's doc
              // comment for why this now calls through polymorphically
              // rather than hardcoding mass+stiffness.
              cell_matrix(i, j) +=
                problem->master->localBilinearForm(fe_values, i, j, q);

        problem->master->constraints_dirichlet.distribute_local_to_global(
          cell_matrix, cell_rhs, dof_indices, matrix, rhs);
      }

    matrix.compress(VectorOperation::add);
    rhs.compress(VectorOperation::add);

    TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
    amg_data.higher_order_elements = false;
    amg_data.aggregation_threshold = 1e-3;
    amg_data.smoother_sweeps       = 2;
    amg_data.smoother_type         = "Chebyshev";
    amg_data.coarse_type           = "Amesos-KLU";

    preconditioner_problem.initialize(matrix, amg_data);
  }
} // namespace internodes
