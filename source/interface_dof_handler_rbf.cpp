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
  {
    // The sub-timers below split this setup into its parts (they are nested
    // inside this scope, so their times are included in it).
    TimerOutput::Scope timer_section(timer_output(),
                                     "setup: RBF operators (Phi, AMG, scaling)");

    {
      TimerOutput::Scope rtree_timer(timer_output(), "  RBF setup: R-tree");
      rtree = std::make_shared<RTreeHandler>(support_points_global_, radius_);
    }

    temp_data.reinit(this->interface_dofs_owned().size());
    temp.reinit(this->interface_dofs_owned());

    // interface_dofs_owned() returns a copy: take it once.
    const IndexSet owned_interface = this->interface_dofs_owned();

    pcout() << "Assembling RBF sparsity pattern..." << std::endl;
    {
      TimerOutput::Scope pattern_timer(timer_output(), "  RBF setup: sparsity pattern");

      const unsigned int n_interface(owned_interface.size());
      DynamicSparsityPattern dsp(n_interface,
                                 n_interface,
                                 interface_dofs_relevant());

      std::vector<types::global_dof_index> columns;
      for (std::size_t i = 0; i < owned_interface.n_elements(); ++i)
        {
          const Point<dim>   point = this->support_points()[i];
          const unsigned int idx_i = owned_interface.nth_index_in_set(i);

          const auto results = rtree->query(point);
          columns.clear();
          for (const auto &entry : results)
            columns.push_back(entry.second);
          dsp.add_entries(idx_i, columns.begin(), columns.end());
        }
      dsp.compress();
      SparsityTools::distribute_sparsity_pattern(dsp,
                                                  owned_interface,
                                                  mpi_comm,
                                                  interface_dofs_relevant());
      sp.copy_from(dsp);
      Phi.reinit(owned_interface, sp, mpi_comm);
    }

    pcout() << "Assembling RBF matrix Phi..." << std::endl;
    {
      TimerOutput::Scope assembly_timer(timer_output(), "  RBF setup: Phi assembly");

      // One row at a time: the row's nonzeros are the kernel evaluated at
      // the support points within the radius, exactly the columns of the
      // pattern above.
      std::vector<types::global_dof_index> columns;
      std::vector<double>                  values;
      for (std::size_t i = 0; i < owned_interface.n_elements(); ++i)
        {
          const Point<dim>   point = this->support_points()[i];
          const unsigned int idx_i = owned_interface.nth_index_in_set(i);

          const auto results = rtree->query(point);
          columns.clear();
          values.clear();
          for (const auto &entry : results)
            {
              Point<dim> node_distance = point;
              node_distance -= this->support_points_global()[entry.second];
              columns.push_back(entry.second);
              values.push_back(evaluate_rbf(node_distance));
            }
          Phi.set(idx_i, columns, values);
        }
      Phi.compress(VectorOperation::insert);
    }

    pcout() << "Factoring Phi and computing RL-RBF scaling factors..."
             << std::endl;
    {
      TimerOutput::Scope amg_timer(timer_output(), "  RBF setup: AMG initialization");

      TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
      amg_data.higher_order_elements = true;
      amg_data.smoother_sweeps       = 1;
      amg_data.smoother_type         = "Chebyshev";
      amg_data.coarse_type           = "Amesos-KLU";
      amg_data.aggregation_threshold = 1e-2;
      preconditioner_Phi.initialize(Phi, amg_data);
    }

    // Rescaling: solve Phi * scaling_factors = 1 (the paper's g(x) = 1 in
    // the rescaled interpolant, Eq. before \eqref{eq:interp}).
    {
      TimerOutput::Scope scaling_timer(timer_output(), "  RBF setup: scaling solve");

      ReductionControl               control(100000, 1e-12, 1e-10);
      SolverCG<TrilinosWrappers::MPI::Vector> solver(control);
      TrilinosWrappers::MPI::Vector  ones(this->interface_dofs_owned());
      ones.add(1.);

      scaling_factors.reinit(this->interface_dofs_owned());
      solver.solve(Phi, scaling_factors, ones, preconditioner_Phi);

      scaling_factor_vectors = scaling_factors;
    }
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
  InterfaceDoFHandlerRBF::setup_destination_points(
    const std::vector<Point<dim>> &points,
    const IndexSet                &destination_owned)
  {
    TimerOutput::Scope timer_section(timer_output(), "Setup destination points");

    destination_points_map.clear();
    destination_owned_ = destination_owned;

    // The weights of a destination point only depend on the source support
    // points, which every rank holds (and on the R-tree over them): each
    // rank computes them for the points of the destination vector it owns
    // and keeps them, without any communication. (They used to be computed
    // for a slice of all the points and then gathered on every rank -- for
    // Test 3 of the paper about 160 MB per rank.)
    std::vector<types::global_dof_index> indices;
    std::vector<double>                  weights;
    for (const types::global_dof_index i : destination_owned_)
      {
        const Point<dim> &point   = points[i];
        const auto        results = rtree->query(point);

        indices.clear();
        weights.clear();
        for (const auto &entry : results)
          {
            Point<dim> node_distance = point;
            node_distance -= this->support_points_global()[entry.second];
            const double weight = evaluate_rbf(node_distance);
            if (weight > 1e-15)
              {
                indices.push_back(entry.second);
                weights.push_back(weight);
              }
          }

        if (!indices.empty())
          destination_points_map[i] = std::make_pair(indices, weights);
      }
  }

  void
  InterfaceDoFHandlerRBF::interpolate(TrilinosWrappers::MPI::Vector       &dst,
                                       const TrilinosWrappers::MPI::Vector &src,
                                       const std::vector<Point<dim>> & /*points*/) const
  {
    AssertThrow(dst.locally_owned_elements() == destination_owned_,
                ExcMessage("The destination vector must own exactly the entries "
                           "passed as destination_owned to "
                           "setup_destination_points()."));

    // Step 1: temp = Phi^{-1} * src (the source-side RBF coefficients gamma
    // in the paper's notation).
    {
      TimerOutput::Scope timer_section(timer_output(), "  interpolate: solve Phi");
      ReductionControl   control(100000, phi_tolerance, phi_reduction);
      SolverCG<TrilinosWrappers::MPI::Vector> solver(control);
      solver.solve(Phi, temp, src, preconditioner_Phi);
      record_cg_solve("rbf", control.last_step());
      temp_data = temp;
    }
    TimerOutput::Scope timer_section(timer_output(), "  interpolate: evaluate at points");

    // Step 2: evaluate the rescaled interpolant at each destination point
    // owned here: value = (weights . gamma) / (weights . scaling_factors),
    // with the sparse weights of the point.
    for (const types::global_dof_index i : destination_owned_)
      {
        double     value        = 0.;
        double     scale_factor = 0.;
        const auto point_data   = destination_points_map.find(i);
        if (point_data != destination_points_map.end())
          {
            const auto &indices = point_data->second.first;
            const auto &weights = point_data->second.second;
            for (std::size_t k = 0; k < indices.size(); ++k)
              {
                value += weights[k] * temp_data[indices[k]];
                scale_factor += weights[k] * scaling_factor_vectors[indices[k]];
              }
          }
        dst(i) = value / scale_factor;
      }
    dst.compress(VectorOperation::insert);
  }
} // namespace internodes
