#include "internodes/interface_dof_handler_rbf.hpp"

#include <deal.II/lac/dynamic_sparsity_pattern.h>
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
    TimerOutput::Scope timer_section(timer_output(),
                                     "setup: RBF operators (Phi, direct factor, scaling)");
    rhs.reinit(this->interface_dofs_owned().size());
    temp_data.reinit(this->interface_dofs_owned().size());
    temp.reinit(this->interface_dofs_owned());

    pcout() << "Assembling RBF sparsity pattern..." << std::endl;
    const unsigned int n_interface(interface_dofs_owned().size());
    DynamicSparsityPattern dsp(n_interface,
                               n_interface,
                               interface_dofs_relevant());

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

    pcout() << "Factoring Phi (direct solve) and computing RL-RBF scaling "
               "factors..."
             << std::endl;
    solver_Phi.initialize(Phi);

    // Rescaling: solve Phi * scaling_factors = 1 (the paper's g(x) = 1 in
    // the rescaled interpolant, Eq. before \eqref{eq:interp}).
    TrilinosWrappers::MPI::Vector  ones(this->interface_dofs_owned());
    ones.add(1.);

    scaling_factors.reinit(this->interface_dofs_owned());
    solver_Phi.solve(scaling_factors, ones);

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
    const std::vector<Point<dim>> &points,
    const IndexSet                &destination_owned)
  {
    TimerOutput::Scope timer_section(timer_output(), "Setup destination points");

    destination_points_map.clear();
    destination_owned_ = destination_owned;

    // Unlike the Lagrange base class, a destination point's RBF weights
    // don't depend on which rank's cell contains it -- only on the
    // (globally replicated) source support points and the R-tree over
    // them, which every rank already holds. So each rank computes the
    // weights of exactly the destination points it owns and keeps them,
    // with no communication at all (it used to compute a load-balanced
    // slice of *all* points and then all_gather the result onto every
    // rank -- hundreds of MB per rank at large support radii, and a cost
    // that doesn't shrink with more ranks since the full map has to land
    // everywhere regardless of how many there are).
    for (const types::global_dof_index i : destination_owned_)
      {
        const Point<dim> point   = points[i];
        const auto       results = rtree->query(point);

        for (const auto &entry : results)
          {
            Point<dim> node_distance = point;
            node_distance -= this->support_points_global()[entry.second];
            const double weight = evaluate_rbf(node_distance);
            if (weight > 1e-15)
              {
                destination_points_map[i].first.push_back(entry.second);
                destination_points_map[i].second.push_back(weight);
              }
          }
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
    // in the paper's notation). Reuses the factorization computed once in
    // the constructor -- solve() does not refactor.
    {
      TimerOutput::Scope timer_section(timer_output(), "  interpolate: solve Phi");
      solver_Phi.solve(temp, src);
      temp_data = temp;
    }
    TimerOutput::Scope timer_section(timer_output(), "  interpolate: evaluate at points");

    // Step 2: evaluate the rescaled interpolant at each destination point
    // owned here: value = (weights . gamma) / (weights . scaling_factors),
    // using the sparse weights computed in setup_destination_points()
    // directly (no dense O(n_interface) pass per point).
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
