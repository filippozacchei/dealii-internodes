#include "internodes/sub_problem_base.hpp"

#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/numerics/vector_tools.h>

namespace internodes
{
  void
  SubProblemBase::setupSystem(double radius, InterfaceDoFHandlerRBF::Mode mode)
  {
    TimerOutput::Scope setup_timer(timer_output(),
                                   "setup: subproblem (DoFs, interface, matrices)");
    fe = slice->get_fe_lagrange(fe_degree);

    if (slice->is_hex())
      {
        quadrature_formula      = std::make_unique<QGauss<dim>>(fe_degree + 1);
        face_quadrature_formula = std::make_unique<QGauss<dim - 1>>(fe_degree + 1);
      }
    else // tet
      {
        quadrature_formula = slice->get_quadrature_gauss(fe_degree + 1);
        face_quadrature_formula =
          std::make_unique<QGaussSimplex<dim - 1>>(fe_degree + 1);
      }

    dof_handler->reinit(slice->get());
    dof_handler->distribute_dofs(*fe);
    DoFRenumbering::Cuthill_McKee(*dof_handler);
    owned_dofs = dof_handler->locally_owned_dofs();

    // Renumber DoFs so that all interface DoFs come last -- this is what
    // makes the subsequent block split (block 0 = internal, block 1 =
    // interface) a contiguous range rather than a scattered set.
    TimerOutput::Scope renumber_timer(timer_output(), "setup: interface-DoFs-last renumbering");
    const IndexSet interface_dofs_pre(
      DoFTools::extract_boundary_dofs(*dof_handler, ComponentMask(), interface_id));

    std::vector<bool>     selected_dofs(dof_handler->n_dofs());
    std::vector<IndexSet> interface_dofs_total_pre =
      Utilities::MPI::all_gather(mpi_comm, interface_dofs_pre);
    for (unsigned int i = 0; i < dof_handler->n_dofs(); ++i)
      for (const IndexSet &j : interface_dofs_total_pre)
        if (j.is_element(i))
          selected_dofs[i] = true;

    std::vector<types::global_dof_index> new_numbers(dof_handler->n_dofs());
    std::vector<types::global_dof_index> new_numbers_owned(
      dof_handler->locally_owned_dofs().n_elements());
    DoFRenumbering::compute_sort_selected_dofs_back(new_numbers,
                                                     *dof_handler,
                                                     selected_dofs);
    for (unsigned int i = 0; i < dof_handler->n_dofs(); ++i)
      if (dof_handler->locally_owned_dofs().is_element(i))
        new_numbers_owned[dof_handler->locally_owned_dofs().index_within_set(i)] =
          new_numbers[i];
    dof_handler->renumber_dofs(new_numbers_owned);
    renumber_timer.stop();

    owned_dofs = dof_handler->locally_owned_dofs();
    relevant_dofs = DoFTools::extract_locally_relevant_dofs(*dof_handler);

    if (radius > 0.)
      interface_dofHandler_ptr =
        std::make_shared<InterfaceDoFHandlerRBF>(dof_handler,
                                                  slice,
                                                  interface_id,
                                                  neumann_ids,
                                                  dirichlet_ids,
                                                  radius,
                                                  mode);
    else
      interface_dofHandler_ptr = std::make_shared<InterfaceDoFHandler>(
        dof_handler, slice, interface_id, neumann_ids, dirichlet_ids);

    interface_dofHandler_ptr->interfaceMassMatrix(sp_M_gamma,
                                                   M_gamma,
                                                   *face_quadrature_formula);

    const std::vector<IndexSet> relevant_dofs_block = {
      interface_dofHandler_ptr->internal_dofs_relevant(),
      interface_dofHandler_ptr->interface_dofs_relevant()};

    BlockDynamicSparsityPattern dsp(relevant_dofs_block);
    DoFTools::make_sparsity_pattern(*dof_handler, dsp, constraints_dirichlet);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                                owned_dofs,
                                                mpi_comm,
                                                relevant_dofs);

    const std::vector<IndexSet> owned_dofs_block = {
      interface_dofHandler_ptr->internal_dofs_owned(),
      interface_dofHandler_ptr->interface_dofs_owned()};

    matrix.reinit(owned_dofs_block, dsp, mpi_comm);
    rhs.reinit(owned_dofs_block);
  }

  void
  SubProblemBase::assembly_global(bool intermediate)
  {
    constraints_dirichlet.clear();
    // Note: reinit() with the locally relevant DoFs is required by modern
    // deal.II for correct parallel behavior; the original lifex-based code
    // relied on lifex::utils::BCHandler doing this internally.
    constraints_dirichlet.reinit(owned_dofs, relevant_dofs);

    for (const auto &id : dirichlet_ids)
      VectorTools::interpolate_boundary_values(*dof_handler,
                                                id,
                                                *fun,
                                                constraints_dirichlet);
    constraints_dirichlet.close();

    this->assembly(intermediate);

    rhs_in    = rhs.block(0);
    rhs_gamma = rhs.block(1);
  }

  void
  SubProblemBase::apply_dirichlet_to_internal(
    TrilinosWrappers::MPI::Vector &internal_solution) const
  {
    const InterfaceDoFHandler &idh = *interface_dofHandler_ptr;
    for (const types::global_dof_index global_dof : idh.internal_dofs())
      if (constraints_dirichlet.is_constrained(global_dof))
        internal_solution[idh.internal_local_dof(global_dof)] =
          constraints_dirichlet.get_inhomogeneity(global_dof);
    internal_solution.compress(VectorOperation::insert);
  }

  void
  SubProblemBase::assembly_preconditioner()
  {
    TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
    amg_data.higher_order_elements = false;
    amg_data.aggregation_threshold = 1e-3;
    amg_data.smoother_sweeps       = 2;
    amg_data.smoother_type         = "Chebyshev";
    amg_data.coarse_type           = "Amesos-KLU";

    preconditioner_in_in.initialize(M_in_in(), amg_data);
  }

  void
  SubProblemBase::assembly(bool /*intermediate*/)
  {
    matrix = 0;
    rhs    = 0;

    const auto ptr_mapping = interface_dofHandler_ptr->mapping();
    FEValues<dim> fe_values(*ptr_mapping,
                            interface_dofHandler_ptr->dof_handler()->get_fe(),
                            *quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const auto ptr_mapping_face = interface_dofHandler_ptr->mapping();
    FEFaceValues<dim> fe_face_values(*ptr_mapping_face,
                                     interface_dofHandler_ptr->dof_handler()->get_fe(),
                                     *face_quadrature_formula,
                                     update_values | update_quadrature_points |
                                       update_normal_vectors | update_JxW_values);

    const unsigned int dofs_per_cell =
      interface_dofHandler_ptr->dof_handler()->get_fe().dofs_per_cell;
    const unsigned int n_q_points      = quadrature_formula->size();
    const unsigned int n_face_q_points = face_quadrature_formula->size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    for (const auto &cell :
         interface_dofHandler_ptr->dof_handler()->active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        cell->get_dof_indices(dof_indices);
        fe_values.reinit(cell);

        cell_matrix = 0;
        cell_rhs    = 0;

        for (unsigned int q = 0; q < n_q_points; ++q)
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                cell_matrix(i, j) += localBilinearForm(fe_values, i, j, q);

              cell_rhs(i) += forcing_term->value(fe_values.quadrature_point(q)) *
                             fe_values.shape_value(i, q) * fe_values.JxW(q);
            }

        for (const auto &face : cell->face_iterators())
          if (face->at_boundary() &&
              neumann_ids.find(face->boundary_id()) != neumann_ids.cend())
            {
              fe_face_values.reinit(cell, face);
              for (unsigned int q = 0; q < n_face_q_points; ++q)
                {
                  const double neumann_value =
                    fun_neumann->gradient(fe_face_values.quadrature_point(q)) *
                    fe_face_values.normal_vector(q);

                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    cell_rhs(i) += neumann_value *
                                   fe_face_values.shape_value(i, q) *
                                   fe_face_values.JxW(q);
                }
            }

        constraints_dirichlet.distribute_local_to_global(
          cell_matrix, cell_rhs, dof_indices, matrix, rhs);
      }

    matrix.compress(VectorOperation::add);
    rhs.compress(VectorOperation::add);
  }
} // namespace internodes
