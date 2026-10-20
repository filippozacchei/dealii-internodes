#ifndef INTERNODES_AMG_SETTINGS_HPP
#define INTERNODES_AMG_SETTINGS_HPP

#include <deal.II/lac/trilinos_precondition.h>

#include <cstdlib>
#include <string>

namespace internodes
{
  /// ML AMG settings for the volume (subdomain and Schur-preconditioner)
  /// solves. Experimental switch INTERNODES_VOLUME_AMG:
  ///   lifex (default)  aggregation_threshold 1e-3, no higher-order flag
  ///                    (lifex's active settings)
  ///   ho_agg           higher_order_elements + elliptic, aggregation 0.02.
  ///                    Fewer CG iterations for P2 (Test 5: -16-20%), but on
  ///                    Galileo100 each iteration costs 25-50% more at 768
  ///                    ranks (coarser hierarchy, coarse levels spread over
  ///                    many ranks); iteration counts for P1 (Tests 1, 4)
  ///                    and P2/P4 (Test 3) unchanged at full scale. Net:
  ///                    faster at 48-192 ranks, slower at 768 in every test
  ///                    -- it hurts strong scaling, so it is not the default.
  ///   ho               ho_agg with aggregation 1e-3 (same as lifex)
  ///   ho_w             ho with W-cycles (slower)
  inline dealii::TrilinosWrappers::PreconditionAMG::AdditionalData
  volume_amg_data()
  {
    const char *const env     = std::getenv("INTERNODES_VOLUME_AMG");
    const std::string variant = env ? env : "lifex";

    dealii::TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
    amg_data.smoother_type = "Chebyshev";
    amg_data.coarse_type   = "Amesos-KLU";
    if (variant == "lifex")
      {
        amg_data.higher_order_elements = false;
        amg_data.aggregation_threshold = 1e-3;
        amg_data.smoother_sweeps       = 2;
      }
    else
      {
        amg_data.elliptic              = true;
        amg_data.higher_order_elements = true;
        amg_data.smoother_sweeps       = 2;
        amg_data.aggregation_threshold = (variant == "ho_agg") ? 2e-2 : 1e-3;
        amg_data.w_cycle               = (variant == "ho_w");
      }
    return amg_data;
  }
} // namespace internodes

#endif
