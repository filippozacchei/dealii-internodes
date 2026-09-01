#ifndef INTERNODES_SUB_PROBLEM_DIFFUSION_REACTION_HPP
#define INTERNODES_SUB_PROBLEM_DIFFUSION_REACTION_HPP

#include "internodes/sub_problem_base.hpp"

namespace internodes
{
  /**
   * @brief SubProblemBase specialization for the reaction-diffusion model
   * problem used in the paper's numerical results:
   * $$-\Delta u + u = f.$$
   */
  class SubProblemDiffusionReaction : public SubProblemBase
  {
  public:
    SubProblemDiffusionReaction() : SubProblemBase() {}

    SubProblemDiffusionReaction(
      const std::shared_ptr<MeshHandler>                         &mesh,
      unsigned int                                                 fe_degree,
      const std::shared_ptr<Function<dim>>                        &dirichlet_fun,
      const std::shared_ptr<Function<dim>>                        &neumann_fun,
      const std::shared_ptr<Function<dim>>                        &forcing_term,
      const std::map<std::string, std::set<types::boundary_id>>  &boundary_tags,
      const std::map<std::string, double>                         &coefficients,
      double                                                        rbf_radius = 0.,
      InterfaceDoFHandlerRBF::Mode rbf_mode = InterfaceDoFHandlerRBF::Mode::Wendland)
      : SubProblemBase(mesh,
                       fe_degree,
                       dirichlet_fun,
                       neumann_fun,
                       forcing_term,
                       boundary_tags,
                       coefficients,
                       rbf_radius,
                       rbf_mode)
    {}

    /// $a_k(\phi_j, \phi_i) = \int_{\Omega_k} (\nabla\phi_j\cdot\nabla\phi_i +
    /// \phi_j\phi_i)\,d\Omega$, evaluated at quadrature point @p q.
    double
    localBilinearForm(const FEValues<dim> &fe_values,
                      unsigned int         i,
                      unsigned int         j,
                      unsigned int         q) override;
  };
} // namespace internodes

#endif // INTERNODES_SUB_PROBLEM_DIFFUSION_REACTION_HPP
