import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import tree_propagation.visualization.plotly as vpl
from tree_propagation.consts import DEFAULT_HEIGHT_CLASS, DEFAULT_RANDOM_SEED
from tree_propagation.environment import DEFAULT_WIND_STRENGTH, EnvironmentState
from tree_propagation.metrics import evaluate_seed_shadow
from tree_propagation.models import SeedShadowMetrics, SeedShadowResult
from tree_propagation.simulation import (
    DEFAULT_SEED_DISPERSAL_SIMULATION_INSTANCES,
    SeedDispersalSimulation,
    SeedDispersalSimulationParams,
)
from tree_propagation.species_config import (
    SPECIES_TRAIT_LABELS,
    SpeciesConfig,
    load_species_configs_from_csv,
)
from tree_propagation.tables import build_species_trait_table, windrose_to_dataframe
from tree_propagation.types import DispersalType
from tree_propagation.visualization.common import compute_plot_extent
from tree_propagation.wind_config import DEFAULT_WIND_ROSE, WindRose

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = APP_DIR / "data" / "tree_species_config.csv"
INSTANCE_OPTIONS = [1, 5, 10, 25, 50, 100, 250, 500, 1_000, 5_000, 10_000, 100_000]
WIND_STRENGTH_OPTIONS = [1, 5, 10, 20, 50]

COMPARISON_SLOTS = 3
DEFAULT_DISPERSAL_ORDER = (DispersalType.WIND, DispersalType.GRAVITY, DispersalType.ANIMAL)


@dataclass(frozen=True)
class SidebarControls:
    """Global sidebar settings."""
    seed: int
    wind_strength: int
    num_instances: int
    share_axes: bool


@st.cache_data
def get_species_configs(csv_path: str) -> dict[str, SpeciesConfig]:
    return load_species_configs_from_csv(csv_path)


def format_species_properties(config: SpeciesConfig):
    """Formats a SpeciesConfig dataclass into a styled two-column property table."""
    rows = [(k, "—" if v is None else str(v)) for k, v in asdict(config).items()]
    df = pd.DataFrame(rows, columns=["Property", "Value"])
    return df.style.map(lambda _: "font-weight: bold", subset=pd.IndexSlice[:, ["Property"]])  # type: ignore


def _randomize_seed() -> None:
    st.session_state.seed = int(np.random.default_rng().integers(0, 1_000_000))


def _default_species_indices(species_names: list[str], configs: dict[str, SpeciesConfig]) -> list[int]:
    """Picks one default species per dispersal type (Wind, Gravity, Animal), with safe fallback."""
    indices: list[int] = []
    for target_type in DEFAULT_DISPERSAL_ORDER:
        match = next(
            (i for i, name in enumerate(species_names) if configs[name].dispersal_type == target_type),
            None,
        )
        if match is not None and match not in indices:
            indices.append(match)

    for i in range(len(species_names)):
        if len(indices) >= COMPARISON_SLOTS:
            break
        if i not in indices:
            indices.append(i)
    return indices


def render_sidebar() -> SidebarControls:
    st.session_state.setdefault("seed", DEFAULT_RANDOM_SEED)

    with st.sidebar:
        st.header("Global Parameters")
        seed_col, btn_col = st.columns([0.6, 0.4], vertical_alignment="bottom")
        with seed_col:
            st.number_input(label="Seed", min_value=0, step=1, key="seed")
        with btn_col:
            st.button("New seed", on_click=_randomize_seed, width="stretch")

        wind_strength = st.select_slider(
            "Wind Strength (s)",
            options=WIND_STRENGTH_OPTIONS,
            value=DEFAULT_WIND_STRENGTH,
            help="Scales wind-dispersed species; gravity and animal species are wind-invariant.",
        )
        num_instances = st.select_slider(
            "Simulation Instances (N)",
            options=INSTANCE_OPTIONS,
            value=DEFAULT_SEED_DISPERSAL_SIMULATION_INSTANCES,
        )
        share_axes = st.checkbox(
            "Share axis scale across columns",
            value=True,
            help="Locks 2D and distance axes to the largest dispersal reach across the 3 columns.",
        )

    return SidebarControls(
        seed=int(st.session_state["seed"]),
        wind_strength=wind_strength,
        num_instances=num_instances,
        share_axes=share_axes,
    )


def render_metrics_bar(metrics: SeedShadowMetrics, compact: bool = False) -> None:
    if compact:
        r1_c1, r1_c2 = st.columns(2)
        r1_c1.metric(
            "Total Seeds",
            f"{metrics.total_seeds:,}",
            f"{metrics.mean_seeds_per_instance:.1f} / run",
            delta_color="off",
            delta_arrow="off",
        )
        r1_c2.metric("KS-Test p-value", f"{metrics.ks_pvalue:.3f}")

        r2_c1, r2_c2 = st.columns(2)
        r2_c1.metric("Theoretical Mean (r̄)", f"{metrics.theoretical_mean_radius:.2f} m")
        r2_c2.metric(
            "Simulated Mean",
            f"{metrics.empirical_mean_radius:.2f} m",
            f"{metrics.rel_error_pct:.2f}% err",
            delta_color="off",
            delta_arrow="off",
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Total Seeds Dispersed",
            f"{metrics.total_seeds:,}",
            f"avg {metrics.mean_seeds_per_instance:.1f} / run",
            delta_color="off",
            delta_arrow="off",
        )
        c2.metric("Theoretical Mean (r̄)", 
                f"{metrics.theoretical_mean_radius:.3f} m",
                delta_color="off",
                delta_arrow="off",
                )
        c3.metric(
            "Simulated Mean",
            f"{metrics.empirical_mean_radius:.3f} m",
            f"{metrics.rel_error_pct:.2f}% err",
            delta_color="off",
            delta_arrow="off",
        )
        c4.metric("KS-Test p-value", f"{metrics.ks_pvalue:.3f}")


@st.cache_data(
    show_spinner="Simulating seed shadow…",
    max_entries=16,
    hash_funcs={
        SeedDispersalSimulationParams: lambda p: (
            p.config,
            p.environment_state.wind_strength,
            tuple(p.environment_state.wind_rose.probabilities.items()),
            p.height_class,
            p.num_instances,
            p.seed,
        )
    },
)
def run_seed_shadow(params: SeedDispersalSimulationParams) -> tuple[SeedShadowResult, SeedShadowMetrics]:
    """Runs and evaluates a seed-shadow simulation, cached on its immutable params."""
    sim = SeedDispersalSimulation(params)
    result = sim.run()
    return result, evaluate_seed_shadow(result)


def render_seed_shadow_tab(configs: dict[str, SpeciesConfig], controls: SidebarControls) -> None:
    species_names = list(configs.keys())
    default_indices = _default_species_indices(species_names, configs)
    cols = st.columns(COMPARISON_SLOTS, gap="medium")

    selections: list[tuple[SpeciesConfig, int]] = []
    for slot_idx, col in enumerate(cols):
        with col:
            choice = st.selectbox(
                f"Species {slot_idx + 1}",
                options=species_names,
                index=default_indices[slot_idx],
                key=f"comparison_species_{slot_idx}",
                format_func=lambda name: f"{name} ({configs[name].dispersal_type.value})",
            )
            cfg = configs[choice]
            max_k = cfg.maximal_height_class
            slider_key = f"comparison_height_{slot_idx}"

            if max_k == 1:
                st.caption("Height Class (k): **1** (max K = 1)")
                k_val = 1
            else:
                current_k = int(st.session_state.get(slider_key, DEFAULT_HEIGHT_CLASS))
                st.session_state[slider_key] = max(1, min(current_k, max_k))
                k_val = st.slider(
                    f"Height Class (k) [1–{max_k}]",
                    min_value=1,
                    max_value=max_k,
                    key=slider_key,
                )

            selections.append((cfg, k_val))

    env_state = EnvironmentState(wind_rose=DEFAULT_WIND_ROSE, wind_strength=controls.wind_strength)

    # Run all 3 cached simulations first so shared axis extents can be computed across them
    runs: list[tuple[SpeciesConfig, SeedShadowResult, SeedShadowMetrics]] = []
    for cfg, k_val in selections:
        params = SeedDispersalSimulationParams(
            config=cfg,
            environment_state=env_state,
            height_class=k_val,
            num_instances=controls.num_instances,
            seed=controls.seed,
        )
        result, metrics = run_seed_shadow(params)
        runs.append((cfg, result, metrics))

    if controls.share_axes:
        shared_extent = max(compute_plot_extent(res) for _, res, _ in runs)
        shared_max_dist = max(
            (float(res.distances.max()) for _, res, _ in runs if not res.is_empty),
            default=shared_extent,
        )
    else:
        shared_extent = None
        shared_max_dist = None

    for slot_idx, (col, (cfg, result, metrics)) in enumerate(zip(cols, runs, strict=True)):
        with col:
            render_metrics_bar(metrics, compact=True)

            if result.is_empty:
                st.warning("Zero seeds produced (estimated age k·T is below maturity age A).")
            else:
                st.plotly_chart(
                    vpl.plot_seed_shadow_2d(result=result, max_extent=shared_extent),
                    width="stretch",
                    key=f"shadow_2d_{slot_idx}",
                )
                st.plotly_chart(
                    vpl.plot_distance_distribution(
                        result=result, metrics=metrics, max_dist=shared_max_dist
                    ),
                    width="stretch",
                    key=f"dist_hist_{slot_idx}",
                )
                st.plotly_chart(
                    vpl.plot_directional_transects(
                        result=result,
                        wind_rose=env_state.wind_rose,
                        max_dist=shared_max_dist,
                        cols=1,
                    ),
                    width="stretch",
                    key=f"transects_{slot_idx}",
                )

            with st.expander("Species Traits"):
                st.dataframe(
                    format_species_properties(cfg),
                    column_config={"Property": st.column_config.MarkdownColumn("Property")},
                    hide_index=True,
                )


def render_wind_tab(wind_rose: WindRose) -> None:
    st.plotly_chart(vpl.plot_wind_rose(wind_rose), width="stretch")
    _, col_table, _ = st.columns([0.25, 0.5, 0.25], vertical_alignment="center")
    with col_table:
        st.subheader("Sector Probabilities (Blowing From Direction)")
        st.table(windrose_to_dataframe(wind_rose))
        st.caption(
            "Wind directions denote where wind blows *from*; dispersed seeds travel "
            "180° opposite (downwind) with uniform jitter within each 45° sector."
        )

def render_species_overview_tab(configs: dict[str, SpeciesConfig], controls: SidebarControls) -> None:
    max_k = max(cfg.maximal_height_class for cfg in configs.values())
    ref_k = st.slider(
        "Reference Height Class (k) for Trait Table & Heatmap",
        min_value=1,
        max_value=max_k,
        value=min(DEFAULT_HEIGHT_CLASS, max_k),
        help=("Used to compute Mean Dispersal Radius r̄ " 
              "in the trait matrix and parameter table "
              "(clamped to each species' K)."
        ),
    )

    env_state = EnvironmentState(wind_rose=DEFAULT_WIND_ROSE, wind_strength=controls.wind_strength)
    traits_df = build_species_trait_table(
        configs,
        reference_height_class=ref_k,
        environment=env_state,
    )

    st.plotly_chart(vpl.plot_pioneer_climax_tradeoff(traits_df), width="stretch")
    st.plotly_chart(vpl.plot_species_lifecycle_timelines(traits_df), width="stretch")
    st.plotly_chart(
        vpl.plot_dispersal_reach_by_age(configs, wind_strength=controls.wind_strength),
        width="stretch",
    )
    st.plotly_chart(vpl.plot_growth_over_time(traits_df), width="stretch")
    st.plotly_chart(vpl.plot_species_trait_heatmap(traits_df), width="stretch")

    st.subheader("3D Trait Space Explorer")
    trait_keys = list(SPECIES_TRAIT_LABELS.keys())
    c_x, c_y, c_z = st.columns(3)
    with c_x:
        x_key = st.selectbox("X-Axis Variable", trait_keys, index=0, format_func=SPECIES_TRAIT_LABELS.get)  # type: ignore
    with c_y:
        y_key = st.selectbox("Y-Axis Variable", trait_keys, index=1, format_func=SPECIES_TRAIT_LABELS.get)  # type: ignore
    with c_z:
        z_key = st.selectbox("Z-Axis Variable", trait_keys, index=2, format_func=SPECIES_TRAIT_LABELS.get)  # type: ignore

    st.plotly_chart(
        vpl.plot_species_3d_scatter(traits_df, x_key=x_key, y_key=y_key, z_key=z_key),
        width="stretch",
    )

    st.subheader("Species Parameter Table")
    st.dataframe(
        traits_df.rename(columns={"species_name": "Species", "dispersal_type": "Dispersal", **SPECIES_TRAIT_LABELS}),
        hide_index=True,
        width="stretch",
    )


def main() -> None:
    st.set_page_config(page_title="Seed Dispersal Explorer", layout="wide")
    st.title("Tree Seed Shadow Simulator")

    configs = get_species_configs(str(DEFAULT_CONFIG_PATH))
    if not configs:
        st.error(f"No species configurations found in {DEFAULT_CONFIG_PATH}.")
        return

    controls = render_sidebar()

    tab_sim, tab_wind, tab_species = st.tabs(
        ["Seed Shadow Simulation", "Wind Rose", "Species Overview"]
    )
    with tab_sim:
        render_seed_shadow_tab(configs, controls)
    with tab_wind:
        render_wind_tab(DEFAULT_WIND_ROSE)
    with tab_species:
        render_species_overview_tab(configs, controls)


if __name__ == "__main__":
    main()