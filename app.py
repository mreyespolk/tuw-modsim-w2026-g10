import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.tree_propagation.consts import DEFAULT_RANDOM_SEED
from src.tree_propagation.environment import EnvironmentState
from src.tree_propagation.metrics import evaluate_seed_shadow
from src.tree_propagation.models import SeedShadowMetrics
from src.tree_propagation.simulation import simulate_seed_shadow
from src.tree_propagation.tree_config import TreeConfig, load_tree_configs
from src.tree_propagation.utils import windrose_to_dataframe
from src.tree_propagation.visualization_plotly import (
    plot_simulated_seed_shadow_and_distance_histogram,
    plot_wind_rose,
)
from src.tree_propagation.wind_config import DEFAULT_WIND_ROSE, WindRose

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = APP_DIR / "data" / "trees_vienna.csv"
INSTANCE_OPTIONS = [1, 5, 10, 25, 50, 100, 250, 500, 1_000, 5_000, 10_000, 100_000]


@dataclass(frozen=True)
class SimulationParams:
    seed: int
    species: str
    height_class: int
    wind_strength: int
    num_instances: int


@st.cache_data
def get_tree_configs(csv_path: str) -> dict[str, TreeConfig]:
    return load_tree_configs(csv_path)


def format_species_properties(config: TreeConfig):
    """Formats a TreeConfig dataclass into a styled two-column property table."""
    rows = [(k, "—" if v is None else str(v)) for k, v in asdict(config).items()]
    df = pd.DataFrame(rows, columns=["Property", "Value"])
    return df.style.map(lambda _: "font-weight: bold", subset=pd.IndexSlice[:, ["Property"]])  # type: ignore


def _randomize_seed() -> None:
    st.session_state.seed = int(np.random.default_rng().integers(0, 1_000_000))


def render_sidebar(configs: dict[str, TreeConfig]) -> SimulationParams:
    st.session_state.setdefault("seed", DEFAULT_RANDOM_SEED)

    with st.sidebar:
        seed_col, btn_col = st.columns([0.6, 0.4], vertical_alignment="bottom")
        with seed_col:
            st.number_input(label="Seed", min_value=0, step=1, key="seed")
        with btn_col:
            st.button("New seed", on_click=_randomize_seed, use_container_width=True)

        species = st.selectbox("Species", list(configs.keys()))
        selected_config = configs[species]

        default_height = min(4, selected_config.maximal_height_class)
        height_class = st.slider("Height Class (k)", 1, selected_config.maximal_height_class, default_height)
        wind_strength = st.slider("Wind Strength", 1, 10, 1, step=1)
        num_instances = st.select_slider(
            "Simulation Instances (N)",
            options=INSTANCE_OPTIONS,
            value=100,
        )

        st.dataframe(
            format_species_properties(selected_config),
            column_config={"Property": st.column_config.MarkdownColumn("Property")},
            hide_index=True,
        )

    return SimulationParams(
        seed=int(st.session_state["seed"]),
        species=species,
        height_class=height_class,
        wind_strength=wind_strength,
        num_instances=num_instances,
    )


def render_wind_rose_panel(wind_rose: WindRose) -> None:
    st.plotly_chart(plot_wind_rose(wind_rose), use_container_width=True)
    st.table(windrose_to_dataframe(wind_rose))


def render_metrics_bar(metrics: SeedShadowMetrics) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Total Seeds Dispersed",
        f"{metrics.total_seeds:,}",
        f"avg {metrics.mean_seeds_per_instance:.1f} / run",
    )
    c2.metric("Theoretical Mean (r̄)", f"{metrics.theoretical_mean_radius:.3f} m")
    c3.metric(
        "Simulated Mean",
        f"{metrics.empirical_mean_radius:.3f} m",
        f"{metrics.rel_error_pct:.2f}% err",
        delta_color="off",
    )
    c4.metric("KS-Test p-value", f"{metrics.ks_pvalue:.3f}")


def render_seed_shadow_tab(config: TreeConfig, params: SimulationParams) -> None:
    rng = np.random.default_rng(params.seed)
    env_state = EnvironmentState(wind_rose=DEFAULT_WIND_ROSE, wind_strength=params.wind_strength)

    result = simulate_seed_shadow(
        config=config,
        height_class=params.height_class,
        num_instances=params.num_instances,
        environment_state=env_state,
        rng=rng,
    )
    metrics = evaluate_seed_shadow(result)

    render_metrics_bar(metrics)

    if result.is_empty:
        st.warning("Zero seeds were produced across the simulated instances. Increase N or height class.")
    else:
        fig = plot_simulated_seed_shadow_and_distance_histogram(result=result, metrics=metrics)
        st.plotly_chart(fig, use_container_width=True)


def render_wind_tab(wind_rose: WindRose) -> None:
    col_chart, col_table = st.columns([0.45, 0.55], vertical_alignment="center")
    with col_chart:
        st.plotly_chart(plot_wind_rose(wind_rose), use_container_width=True)
    with col_table:
        st.subheader("Sector Probabilities (Blowing From)")
        st.table(windrose_to_dataframe(wind_rose))
        st.caption(
            "Wind directions denote where wind blows *from*; dispersed seeds travel "
            "180° opposite (downwind) with uniform jitter within each 45° sector."
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
        render_seed_shadow_tab(configs[params.species], params)
    with tab_wind:
        render_wind_tab(DEFAULT_WIND_ROSE)
    with tab_species:
        st.info("Species comparison charts and trait table coming soon.")


if __name__ == "__main__":
    main()