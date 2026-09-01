#include "internodes/sub_problem_diffusion_reaction.hpp"

namespace internodes
{
  double
  SubProblemDiffusionReaction::localBilinearForm(const FEValues<dim> &fe_values,
                                                  unsigned int         i,
                                                  unsigned int         j,
                                                  unsigned int         q)
  {
    return (scalar_product(fe_values.shape_grad(i, q), fe_values.shape_grad(j, q)) +
            fe_values.shape_value(i, q) * fe_values.shape_value(j, q)) *
           fe_values.JxW(q);
  }
} // namespace internodes
