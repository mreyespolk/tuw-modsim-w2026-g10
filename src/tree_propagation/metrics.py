import numpy as np
from scipy import stats

from tree_propagation.models import SeedShadowMetrics, SeedShadowResult


def evaluate_seed_shadow(result: SeedShadowResult) -> SeedShadowMetrics:
    """Computes empirical dispersal metrics and KS goodness-of-fit against Exp(r_bar)."""
    mean_per_instance = (
        result.total_seeds / result.num_instances if result.num_instances > 0 else float("nan")
    )

    if result.is_empty:
        return SeedShadowMetrics(
            total_seeds=0,
            mean_seeds_per_instance=mean_per_instance,
            theoretical_mean_radius=result.theoretical_mean_radius,
            empirical_mean_radius=float("nan"),
            rel_error_pct=float("nan"),
            ks_statistic=float("nan"),
            ks_pvalue=float("nan"),
        )

    empirical_mean_radius = float(np.mean(result.distances))
    rel_error_pct = abs(empirical_mean_radius - result.theoretical_mean_radius) / result.theoretical_mean_radius * 100.0
    ks_res = stats.kstest(result.distances, "expon", args=(0, result.theoretical_mean_radius))

    return SeedShadowMetrics(
        total_seeds=result.total_seeds,
        mean_seeds_per_instance=mean_per_instance,
        theoretical_mean_radius=result.theoretical_mean_radius,
        empirical_mean_radius=empirical_mean_radius,
        rel_error_pct=rel_error_pct,
        ks_statistic=float(ks_res.statistic),
        ks_pvalue=float(ks_res.pvalue),
    )