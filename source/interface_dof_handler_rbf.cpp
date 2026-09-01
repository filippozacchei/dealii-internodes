#include "internodes/interface_dof_handler_rbf.hpp"

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparsity_tools.h>

#include <cmath>

namespace internodes
{
  InterfaceDoFHandlerRBF::InterfaceDoFHandlerRBF(
    const std::shared_ptr<const DoFHandler<dim>> &dof_handler,
    const std::shared_ptr<const MeshHandler>     &triangulation,
    const std::set<types::boundary_id>           &interface_id,
    const std::set<types::boundary_id>           &neumann_id,
    const std::set<types::boundary_id>           &dirichlet_id,
    double                                          radius_,
    Mode                                            mode_)
    : InterfaceDoFHandler(dof_handler,
                          triangulation,
                          interface_id,
                          neumann_id,
                          dirichlet_id)
    , radius(radius_)
    , mode(mode_)
    , rtree(std::make_shared<RTreeHandler>(support_points_global_, radius_))
  {
    rhs.reinit(this->interface_dofs_owned().size());
    temp_data.reinit(this->interface_dofs_owned().size());
    temp.reinit(this->interface_dofs_owned());

    pcout() << "Assembling RBF sparsity pattern..." << std::endl;
    const unsigned int n_interface(interface_dofs_owned().size());
    DynamicSparsityPattern dsp(n_interface, n_interface);

    for (std::size_t i = 0; i < interface_dofs_owned().n_elements(); ++i)
      {
        const Point<dim>   point = this->support_points()[i];
        const unsigned int idx_i = interface_dofs_owned().nth_index_in_set(i);

        const auto results = rtree->query(point);
        for (const auto &entry : results)
          dsp.add(idx_i, entry.second);
      }
    dsp.compress();
    SparsityTools::distribute_sparsity_pattern(dsp,
                                                interface_dofs_owned(),
                                                mpi_comm,
                                                interface_dofs_relevant());
    sp.copy_from(dsp);
    Phi.reinit(interface_dofs_owned(), sp, mpi_comm);

    pcout() << "Assembling RBF matrix Phi..." << std::endl;
    for (std::size_t i = 0; i < interface_dofs_owned().n_elements(); ++i)
      {
        const Point<dim>   point = this->support_points()[i];
        const unsigned int idx_i = interface_dofs_owned().nth_index_in_set(i);

        const auto results = rtree->query(point);
        DoFs_values(point, rhs, results);

        for (const auto &entry : results)
          if (sp.exists(idx_i, entry.second))
            Phi.set(idx_i, entry.second, rhs(entry.second));
      }
    Phi.compress(VectorOperation::insert);

    pcout() << "Factoring Phi and computing RL-RBF scaling factors..."
             << std::endl;
    TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
    amg_data.higher_order_elements = true;
    amg_data.smoother_sweeps       = 1;
    amg_data.smoother_type         = "Chebyshev";
    amg_data.coarse_type           = "Amesos-KLU";
    amg_data.aggregation_threshold = 1e-2;
    preconditioner_Phi.initialize(Phi, amg_data);

    // Rescaling: solve Phi * scaling_factors = 1 (the paper's g(x) = 1 in
    // the rescaled interpolant, Eq. before \eqref{eq:interp}).
    ReductionControl               control(100000, 1e-12, 1e-10);
    SolverCG<TrilinosWrappers::MPI::Vector> solver(control);
    TrilinosWrappers::MPI::Vector  ones(this->interface_dofs_owned());
    ones.add(1.);

    scaling_factors.reinit(this->interface_dofs_owned());
    solver.solve(Phi, scaling_factors, ones, preconditioner_Phi);

    scaling_factor_vectors = scaling_factors;
  }

  double
  InterfaceDoFHandlerRBF::evaluate_rbf(const Point<dim> &distance) const
  {
    const double norm_p_over_r = distance.norm() / radius;

    if (mode == Mode::Wendland)
      return std::pow(std::max(0., 1. - norm_p_over_r), 4) *
             (1 + 4 * norm_p_over_r);
    else if (mode == Mode::Gaussian)
      return std::exp(-norm_p_over_r * norm_p_over_r);
    else // Mode::IMQ
      return 1.0 / (radius * std::sqrt(norm_p_over_r * norm_p_over_r + 1));
  }

  void
  InterfaceDoFHandlerRBF::DoFs_values(
    const Point<dim>                                        &p,
    Vector<double>                                            &rhs_vector,
    const std::vector<std::pair<Point<dim>, unsigned int>> &results) const
  {
    rhs_vector = 0;
    for (const auto &entry : results)
      {
        Point<dim> node_distance = p;
        node_distance -= this->support_points_global()[entry.second];
        rhs_vector[entry.second] = evaluate_rbf(node_distance);
      }
  }

  void
  InterfaceDoFHandlerRBF::setup_destination_points(
    const std::vector<Point<dim>> &points)
  {
    // Neighbor queries against the (globally gathered) rtree don't depend
    // on which rank owns which geometry, so the destination point list is
    // simply load-balanced by splitting it into contiguous chunks across
    // ranks -- unlike the Lagrange case, there's no need to iterate locally
    // owned cells.
    const unsigned int my_rank = Utilities::MPI::this_mpi_process(mpi_comm);
    const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(mpi_comm);
    const unsigned int n_local = points.size() / n_ranks;
    const unsigned int start   = my_rank * n_local;
    const unsigned int end =
      (my_rank == n_ranks - 1) ? points.size() : start + n_local;

    for (std::size_t i = start; i < end; ++i)
      {
        const Point<dim> point   = points[i];
        const auto       results = rtree->query(point);
        DoFs_values(point, rhs, results);

        for (const auto &entry : results)
          if (rhs(entry.second) > 1e-15)
            {
              destination_points_map[i].first.push_back(entry.second);
              destination_points_map[i].second.push_back(rhs(entry.second));
            }
      }

    destination_points_map = compute_map_union(destination_points_map, mpi_comm);
  }

  void
  InterfaceDoFHandlerRBF::interpolate(TrilinosWrappers::MPI::Vector       &dst,
                                       const TrilinosWrappers::MPI::Vector &src,
                                       const std::vector<Point<dim>>       &points) const
  {
    // Step 1: temp = Phi^{-1} * src (the source-side RBF coefficients gamma
    // in the paper's notation).
    ReductionControl               control(100000, 1e-12, 1e-10);
    SolverCG<TrilinosWrappers::MPI::Vector> solver(control);
    solver.solve(Phi, temp, src, preconditioner_Phi);
    temp_data = temp;

    // Step 2: evaluate the rescaled interpolant at each destination point.
    // (Loop bound is points.size(), matching setup_destination_points();
    // in practice this equals dst's global size, since `points` is always
    // the full set of destination-interface support points.)
    for (std::size_t i = 0; i < points.size(); ++i)
      {
        if (!dst.locally_owned_elements().is_element(i))
          continue;

        rhs = 0.0;
        const auto point_data = destination_points_map.find(i);
        if (point_data != destination_points_map.end())
          {
            const auto &data = point_data->second;
            rhs.add(data.first, data.second);
          }

        const double scale_factor = scaling_factor_vectors * rhs;
        const double value        = temp_data * rhs;
        dst(i)                    = value / scale_factor;
      }
    dst.compress(VectorOperation::insert);
  }
} // namespace internodes
