#include "internodes/interface_dof_handler.hpp"

#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

#include <algorithm>

namespace internodes
{
  InterfaceDoFHandler::InterfaceDoFHandler(
    const std::shared_ptr<const DoFHandler<dim>> &dof_handler,
    const std::shared_ptr<const MeshHandler>     &triangulation,
    const std::set<types::boundary_id>           &interface_id,
    const std::set<types::boundary_id>           &neumann_id,
    const std::set<types::boundary_id>           &dirichlet_id)
    : dof_handler_(dof_handler)
    , triangulation_(triangulation)
    , interface_dofs_(DoFTools::extract_boundary_dofs(*dof_handler_,
                                                       ComponentMask(),
                                                       interface_id) &
                       dof_handler->locally_owned_dofs())
    , internal_dofs_(dof_handler_->n_dofs())
    , support_points_(interface_dofs_.n_elements())
    , reference_points_(interface_dofs_.n_elements())
    , interface_id_(interface_id)
    , Neumann_id_(neumann_id)
    , Dirichlet_id_(dirichlet_id)
  {
    std::map<types::global_dof_index, Point<dim>> support_points_dof_handler;
    DoFTools::map_dofs_to_support_points(*(this->mapping()),
                                          *dof_handler_,
                                          support_points_dof_handler,
                                          ComponentMask());

    owned_dofs_ = dof_handler->locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(*dof_handler_, relevant_dofs_);

    internal_dofs_ = owned_dofs_;
    internal_dofs_.subtract_set(interface_dofs_);

    // Build interface-local and internal-local DoF numberings, valid across
    // all processes: every process independently computes the same global
    // ordering (all owned interface/internal DoFs, gathered from every
    // rank, in rank order), then keeps only the entries it actually owns
    // (interface_parallelPartitioning/internal_parallelPartitioning) or can
    // see (the "relevant" counterparts).
    std::vector<unsigned int> interface_owned, internal_owned;
    interface_dofs_.fill_index_vector(interface_owned);
    internal_dofs_.fill_index_vector(internal_owned);

    const int n_interface_local = interface_owned.size();
    const int n_internal_local  = internal_owned.size();
    const int n_interface_global =
      Utilities::MPI::sum(n_interface_local, mpi_comm);
    const int n_internal_global =
      Utilities::MPI::sum(n_internal_local, mpi_comm);

    interface_dofs_all_.set_size(n_interface_global);
    interface_dofs_all_.add_range(0, n_interface_global);

    interface_parallelPartitioning.set_size(n_interface_global);
    internal_parallelPartitioning.set_size(n_internal_global);
    interface_relevantParallelPartitioning.set_size(n_interface_global);
    internal_relevantParallelPartitioning.set_size(n_internal_global);

    const std::vector<std::vector<unsigned int>> internal_owned_total_vec =
      Utilities::MPI::all_gather(mpi_comm, internal_owned);
    const std::vector<std::vector<unsigned int>> interface_owned_total_vec =
      Utilities::MPI::all_gather(mpi_comm, interface_owned);

    std::vector<unsigned int> internal_owned_total, interface_owned_total;
    for (const auto &v : internal_owned_total_vec)
      internal_owned_total.insert(internal_owned_total.end(), v.begin(), v.end());
    for (const auto &v : interface_owned_total_vec)
      interface_owned_total.insert(interface_owned_total.end(), v.begin(), v.end());

    for (unsigned int i = 0; i < internal_owned_total.size(); ++i)
      if (owned_dofs_.is_element(internal_owned_total[i]))
        internal_parallelPartitioning.add_index(i);
    for (unsigned int i = 0; i < interface_owned_total.size(); ++i)
      if (owned_dofs_.is_element(interface_owned_total[i]))
        interface_parallelPartitioning.add_index(i);
    for (unsigned int i = 0; i < internal_owned_total.size(); ++i)
      if (relevant_dofs_.is_element(internal_owned_total[i]))
        internal_relevantParallelPartitioning.add_index(i);
    for (unsigned int i = 0; i < interface_owned_total.size(); ++i)
      if (relevant_dofs_.is_element(interface_owned_total[i]))
        interface_relevantParallelPartitioning.add_index(i);

    internal_dofs_total  = Utilities::MPI::all_gather(mpi_comm, internal_dofs_);
    interface_dofs_total = Utilities::MPI::all_gather(mpi_comm, interface_dofs_);

    interface_dofs_total_indexSet.set_size(interface_dofs_.size());
    internal_dofs_total_indexSet.set_size(internal_dofs_.size());
    for (const auto &set : internal_dofs_total)
      internal_dofs_total_indexSet.add_indices(set);
    for (const auto &set : interface_dofs_total)
      interface_dofs_total_indexSet.add_indices(set);

    for (std::size_t i = 0; i < interface_dofs_.n_elements(); ++i)
      support_points_[i] =
        support_points_dof_handler[interface_dofs_.nth_index_in_set(i)];

    support_points_total = Utilities::MPI::all_gather(mpi_comm, support_points_);
    support_points_global_.reserve(interface_dofs_total.size());
    for (const auto &points : support_points_total)
      support_points_global_.insert(support_points_global_.end(),
                                     points.begin(),
                                     points.end());
  }

  unsigned int
  InterfaceDoFHandler::cellFace_index(
    const Point<dim>                            &v,
    const DoFHandler<dim>::active_cell_iterator &cell) const
  {
    for (unsigned int face = 0; face < cell->n_faces(); ++face)
      {
        if (interface_id_.find(cell->face(face)->boundary_id()) !=
            interface_id_.end())
          for (unsigned int p = 0; p < cell->face(face)->n_vertices(); ++p)
            if (cell->face(face)->vertex(p) == v)
              return face;
      }
    return cell->n_faces();
  }

  void
  InterfaceDoFHandler::DoFs_values(const types::global_dof_index &i,
                                    Vector<double>                &rhs_vector) const
  {
    rhs_vector = 0.0;
    const auto point_data = destination_points_map.find(i);
    if (point_data != destination_points_map.end())
      {
        const auto &data = point_data->second;
        rhs_vector.add(data.first, data.second);
      }
  }

  void
  InterfaceDoFHandler::setup_destination_points(
    const std::vector<Point<dim>> &points)
  {
    TimerOutput::Scope timer_section(timer_output(), "Setup destination points");

    destination_points_map.clear();

    const auto         ptr_mapping = this->mapping();
    const unsigned int dofs_per_cell =
      this->dof_handler()->get_fe().n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<bool> found(points.size(), false);

    for (const auto &cell : dof_handler_->active_cell_iterators())
      {
        if (!cell->is_locally_owned() || !cell->at_boundary())
          continue;

        bool at_interface = false;
        for (unsigned int face = 0; face < cell->n_faces(); ++face)
          if (cell->face(face)->at_boundary() &&
              contains(interface_id_, cell->face(face)->boundary_id()))
            {
              at_interface = true;
              break;
            }
        if (!at_interface)
          continue;

        for (unsigned int i = 0; i < points.size(); ++i)
          {
            if (found[i] || !cell->point_inside(points[i]))
              continue;

            found[i] = true;
            cell->get_dof_indices(local_dof_indices);

            // Note: exact only for linear (affine) mappings -- see
            // MeshHandler's documentation.
            const Point<dim> unit_p =
              cell->real_to_unit_cell_affine_approximation(points[i]);

            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              if (is_at_interface(local_dof_indices[j]))
                {
                  destination_points_map[i].first.push_back(
                    interface_local_dof(local_dof_indices[j]));
                  destination_points_map[i].second.push_back(
                    dof_handler()->get_fe().shape_value(j, unit_p));
                }
          }
      }

    destination_points_map = compute_map_union(destination_points_map, mpi_comm);
  }

  void
  InterfaceDoFHandler::interpolate(TrilinosWrappers::MPI::Vector       &dst,
                                    const TrilinosWrappers::MPI::Vector &src,
                                    const std::vector<Point<dim>>       &points) const
  {
    Vector<double> src_data(src);
    Vector<double> rhs(this->interface_dofs_owned().size());

    for (std::size_t i = 0; i < points.size(); ++i)
      if (dst.locally_owned_elements().is_element(i))
        {
          DoFs_values(i, rhs);
          dst(i) = rhs * src_data;
        }

    dst.compress(VectorOperation::insert);
  }

  bool
  InterfaceDoFHandler::is_at_interface(unsigned int i) const
  {
    return interface_dofs_total_indexSet.is_element(i);
  }

  void
  InterfaceDoFHandler::interfaceMassMatrix(
    SparsityPattern                &sp_,
    TrilinosWrappers::SparseMatrix &mass_matrix,
    Quadrature<dim - 1>             &face_quadrature_formula) const
  {
    const unsigned int n_face_q_points = face_quadrature_formula.size();
    const unsigned int dofs_per_face =
      this->dof_handler()->get_fe().n_dofs_per_cell();

    const unsigned int      n_interface(interface_dofs_owned().size());
    DynamicSparsityPattern dsp(n_interface, n_interface);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_face);

    const auto        ptr_mapping = this->mapping();
    FEFaceValues<dim> fe_face_values(*ptr_mapping,
                                      this->dof_handler()->get_fe(),
                                      face_quadrature_formula,
                                      update_values | update_quadrature_points |
                                        update_JxW_values);

    FullMatrix<double> face_matrix(dofs_per_face, dofs_per_face);

    for (const auto &cell : this->dof_handler()->active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;
        for (const auto &face : cell->face_iterators())
          if (interface_id_.find(face->boundary_id()) != interface_id_.end())
            {
              fe_face_values.reinit(cell, face);
              cell->get_dof_indices(local_dof_indices);
              for (unsigned int i = 0; i < dofs_per_face; ++i)
                if (is_at_interface(local_dof_indices[i]))
                  for (unsigned int j = 0; j < dofs_per_face; ++j)
                    if (is_at_interface(local_dof_indices[j]))
                      dsp.add(interface_local_dof(local_dof_indices[i]),
                              interface_local_dof(local_dof_indices[j]));
            }
      }

    dsp.compress();
    SparsityTools::distribute_sparsity_pattern(dsp,
                                                interface_dofs_owned(),
                                                mpi_comm,
                                                interface_dofs_relevant());
    sp_.copy_from(dsp);

    mass_matrix.reinit(interface_dofs_owned(),
                        interface_dofs_owned(),
                        dsp,
                        mpi_comm);

    for (const auto &cell : this->dof_handler()->active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        face_matrix = 0.;
        for (const auto &face : cell->face_iterators())
          if (interface_id_.find(face->boundary_id()) != interface_id_.end())
            {
              fe_face_values.reinit(cell, face);
              for (unsigned int q = 0; q < n_face_q_points; ++q)
                for (unsigned int i = 0; i < dofs_per_face; ++i)
                  for (unsigned int j = 0; j < dofs_per_face; ++j)
                    face_matrix(i, j) += fe_face_values.shape_value(i, q) *
                                          fe_face_values.shape_value(j, q) *
                                          fe_face_values.JxW(q);

              cell->get_dof_indices(local_dof_indices);
              for (unsigned int i = 0; i < dofs_per_face; ++i)
                if (is_at_interface(local_dof_indices[i]))
                  for (unsigned int j = 0; j < dofs_per_face; ++j)
                    if (is_at_interface(local_dof_indices[j]))
                      mass_matrix.add(interface_local_dof(local_dof_indices[i]),
                                       interface_local_dof(local_dof_indices[j]),
                                       face_matrix(i, j));
            }
      }
    mass_matrix.compress(VectorOperation::add);
  }

  types::global_dof_index
  InterfaceDoFHandler::interface_local_dof(unsigned int i) const
  {
    return interface_dofs_total_indexSet.index_within_set(i);
  }
  types::global_dof_index
  InterfaceDoFHandler::interface_global_dof(unsigned int i) const
  {
    return interface_dofs_total_indexSet.nth_index_in_set(i);
  }
  types::global_dof_index
  InterfaceDoFHandler::internal_local_dof(unsigned int i) const
  {
    return internal_dofs_total_indexSet.index_within_set(i);
  }
  types::global_dof_index
  InterfaceDoFHandler::internal_global_dof(unsigned int i) const
  {
    return internal_dofs_total_indexSet.nth_index_in_set(i);
  }
} // namespace internodes
