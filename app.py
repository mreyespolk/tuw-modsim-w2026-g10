import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import tree_propagation.visualization_plotly as visualization_plotly
from tree_propagation.consts import DEFAULT_HEIGHT_CLASS, DEFAULT_RANDOM_SEED
from tree_propagation.environment import DEFAULT_WIND_STRENGTH, EnvironmentState
from tree_propagation.metrics import evaluate_seed_shadow
from tree_propagation.models import SeedShadowMetrics, SeedShadowResult
from tree_propagation.simulation import (
    DEFAULT_SEED_DISPERSAL_SIMULATION_INSTANCES,
    SeedDispersalSimulation,
    SeedDispersalSimulationParams,
)
from tree_propagation.tree_config import (
    SPECIES_TRAIT_LABELS,
    TreeConfig,
    load_tree_configs_from_csv,
)
from tree_propagation.utils import build_species_trait_table, windrose_to_dataframe
from tree_propagation.wind_config import DEFAULT_WIND_ROSE, WindRose

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = APP_DIR / "data" / "tree_species_config.csv"
INSTANCE_OPTIONS = [1, 5, 10, 25, 50, 100, 250, 500, 1_000, 5_000, 10_000, 100_000]
WIND_STRENGTH_OPTIONS = [1, 5, 10, 20, 50]


@st.cache_data
def get_tree_configs(csv_path: str) -> dict[str, TreeConfig]:
    return load_tree_configs_from_csv(csv_path)


def format_species_properties(config: TreeConfig):
    """Formats a TreeConfig dataclass into a styled two-column property table."""
    rows = [(k, "—" if v is None else str(v)) for k, v in asdict(config).items()]
    df = pd.DataFrame(rows, columns=["Property", "Value"])
    return df.style.map(lambda _: "font-weight: bold", subset=pd.IndexSlice[:, ["Property"]])  # type: ignore


def _randomize_seed() -> None:
    st.session_state.seed = int(np.random.default_rng().integers(0, 1_000_000))


def render_sidebar(configs: dict[str, TreeConfig]) -> SeedDispersalSimulationParams:
    st.session_state.setdefault("seed", DEFAULT_RANDOM_SEED)

    with st.sidebar:
        seed_col, btn_col = st.columns([0.6, 0.4], vertical_alignment="bottom")
        with seed_col:
            st.number_input(label="Seed", min_value=0, step=1, key="seed")
        with btn_col:
            st.button("New seed", on_click=_randomize_seed, width="stretch")

        species = st.selectbox("Species", list(configs.keys()))
        selected_config = configs[species]

        max_k = selected_config.maximal_height_class
        if max_k == 1:
            st.caption("Height Class (k): **1** (max K = 1)")
            height_class = 1
        else:
            current_k = int(st.session_state.get("sidebar_height_class", DEFAULT_HEIGHT_CLASS))
            st.session_state["sidebar_height_class"] = max(1, min(current_k, max_k))
            height_class = st.slider(
                "Height Class (k)",
                min_value=1,
                max_value=max_k,
                key="sidebar_height_class",
            )

        wind_strength = st.select_slider(
            "Wind Strength",
            options=WIND_STRENGTH_OPTIONS,
            value=DEFAULT_WIND_STRENGTH,
        )
        num_instances = st.select_slider(
            "Simulation Instances (N)",
            options=INSTANCE_OPTIONS,
            value=DEFAULT_SEED_DISPERSAL_SIMULATION_INSTANCES,
        )

        st.dataframe(
            format_species_properties(selected_config),
            column_config={"Property": st.column_config.MarkdownColumn("Property")},
            hide_index=True,
        )

    return SeedDispersalSimulationParams(
        config=selected_config,
        environment_state=EnvironmentState(wind_rose=DEFAULT_WIND_ROSE, wind_strength=wind_strength),
        height_class=height_class,
        num_instances=num_instances,
        seed=int(st.session_state["seed"]),
    )


def render_metrics_bar(metrics: SeedShadowMetrics) -> None:
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


def render_seed_shadow_tab(params: SeedDispersalSimulationParams) -> None:
    result, metrics = run_seed_shadow(params)

    render_metrics_bar(metrics)

    if result.is_empty:
        st.warning("Zero seeds were produced across the simulated instances. Increase N or height class.")
    else:
        fig = visualization_plotly.plot_simulated_seed_shadow_and_distance_histogram(result=result, metrics=metrics)
        st.plotly_chart(fig, width="stretch")

        fig_transects = visualization_plotly.plot_directional_transects(
            result=result,
            wind_rose=params.environment_state.wind_rose,
        )
        st.plotly_chart(fig_transects, width="stretch")


def render_wind_tab(wind_rose: WindRose) -> None:
    st.plotly_chart(visualization_plotly.plot_wind_rose(wind_rose), width="stretch")
    _, col_table, _ = st.columns([0.25, 0.5, 0.25], vertical_alignment="center")
    with col_table:
        st.subheader("Sector Probabilities (Blowing From Direction)")
        st.table(windrose_to_dataframe(wind_rose))
        st.caption(
            "Wind directions denote where wind blows *from*; dispersed seeds travel "
            "180° opposite (downwind) with uniform jitter within each 45° sector."
        )

def render_species_overview_tab(
    configs: dict[str, TreeConfig],
    params: SeedDispersalSimulationParams,
) -> None:
    traits_df = build_species_trait_table(
        configs,
        reference_height_class=params.height_class,
        environment=params.environment_state,
    )

    st.plotly_chart(visualization_plotly.plot_pioneer_climax_tradeoff(traits_df), width="stretch")
    st.plotly_chart(visualization_plotly.plot_species_lifecycle_timelines(traits_df), width="stretch")
    st.plotly_chart(
        visualization_plotly.plot_dispersal_reach_by_age(
            configs,
            wind_strength=params.environment_state.wind_strength,
        ),
        width="stretch",
    )
    st.plotly_chart(visualization_plotly.plot_growth_over_time(traits_df), width="stretch")
    st.plotly_chart(visualization_plotly.plot_species_trait_heatmap(traits_df), width="stretch")

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
        visualization_plotly.plot_species_3d_scatter(
            traits_df,
            x_key=x_key,
            y_key=y_key,
            z_key=z_key,
        ),
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

    configs = get_tree_configs(str(DEFAULT_CONFIG_PATH))
    if not configs:
        st.error(f"No species configurations found in {DEFAULT_CONFIG_PATH}.")
        return

    params = render_sidebar(configs)

    tab_sim, tab_wind, tab_species = st.tabs(
        ["Seed Shadow Simulation", "Wind Rose", "Species Overview"]
    )
    with tab_sim:
        render_seed_shadow_tab(params)
    with tab_wind:
        render_wind_tab(params.environment_state.wind_rose)
    with tab_species:
        render_species_overview_tab(configs, params)


if __name__ == "__main__":
    main()