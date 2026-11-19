from collections.abc import Mapping

import pandas as pd

from tree_propagation.dispersal import compute_mean_dispersal_distance
from tree_propagation.environment import EnvironmentState
from tree_propagation.tree_config import SPECIES_TRAIT_LABELS, TreeConfig
from tree_propagation.wind_config import WindRose


def windrose_to_dataframe(wind_rose: WindRose) -> pd.DataFrame:
    return pd.DataFrame(
        [{direction.name: prob for direction, prob in wind_rose.probabilities.items()}],
        index=["Probability"],
    )

def build_species_trait_table(
    species_config: Mapping[str, TreeConfig],
    reference_height_class: int = 4,
    environment: EnvironmentState | None = None,
) -> pd.DataFrame:
    """Builds a tabular DataFrame of raw and derived species traits for tables and plots."""
    env = environment if environment is not None else EnvironmentState()
    rows: list[dict[str, object]] = []

    for cfg in species_config.values():
        k_eff = min(reference_height_class, cfg.maximal_height_class)
        r_bar = compute_mean_dispersal_distance(
            config=cfg,
            height_class=k_eff,
            environment=env,
        )

        row: dict[str, object] = {
            "species_name": cfg.species_name,
            "dispersal_type": cfg.dispersal_type.value,
        }
        for trait_key in SPECIES_TRAIT_LABELS:
            if trait_key == "mean_dispersal_radius":
                row[trait_key] = float(r_bar)
            else:
                row[trait_key] = getattr(cfg, trait_key)

        rows.append(row)

    return pd.DataFrame(rows)