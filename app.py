import logging
from dataclasses import asdict

import numpy as np
import pandas as pd
import streamlit as st

from src.tree_propagation.environment import EnvironmentState
from src.tree_propagation.metrics import evaluate_seed_shadow
from src.tree_propagation.simulation import simulate_seed_shadow
from src.tree_propagation.tree_config import TreeConfig, load_tree_configs
from src.tree_propagation.utils import windrose_to_dataframe
from src.tree_propagation.visualization import (
    plot_seed_shadow_and_distance,
    plot_wind_rose,
)
from src.tree_propagation.wind_config import DEFAULT_WIND_ROSE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Seed Dispersal Explorer", layout="wide")
st.title("Tree Seed Shadow Simulator")

RANDOM_SEED = 10
rng = np.random.default_rng(RANDOM_SEED)


@st.cache_data
def get_tree_configs(csv_path: str) -> dict[str, TreeConfig]:
    return load_tree_configs(csv_path)


configs = get_tree_configs("./data/trees_vienna.csv")

with st.sidebar:
    species = st.selectbox("Species", list(configs.keys()))
    selected_config = configs[species]

    height_class = st.slider("Height Class (k)", 1, selected_config.maximal_height_class, 4)
    wind_strength = st.slider("Wind Strength", 1, 10, 1, step=1)
    num_instances = st.select_slider(
        "Simulation Instances (N)",
        options=[1, 5, 10, 25, 50, 100, 250, 500, 1_000, 5_000, 10_000, 100_000],
        value=100,
    )

    df = pd.DataFrame(
        [(k, "—" if v is None else str(v)) for k, v in asdict(selected_config).items()],
        columns=["Property", "Value"],
    )
    styled_df = df.style.map(lambda _: "font-weight: bold", subset=pd.IndexSlice[:, ["Property"]])
    st.dataframe(
        styled_df,
        column_config={"Property": st.column_config.MarkdownColumn("Property")},
        hide_index=True,
    )

res = simulate_seed_shadow(
    selected_config,
    height_class=height_class,
    num_instances=num_instances,
    environment_state=EnvironmentState(wind_rose=DEFAULT_WIND_ROSE, wind_strength=wind_strength),
    rng=rng,
)
metrics = evaluate_seed_shadow(res)

col_rose, col_metrics = st.columns([0.42, 0.58], vertical_alignment="center")
with col_rose:
    st.plotly_chart(plot_wind_rose(DEFAULT_WIND_ROSE), use_container_width=True)
    st.table(windrose_to_dataframe(DEFAULT_WIND_ROSE))

with col_metrics:
    c1, c2 = st.columns(2)
    c1.metric(
        "Total Seeds Dispersed",
        f"{metrics.total_seeds:,}",
        f"avg {metrics.mean_seeds_per_instance:.1f} / run",
    )
    c1.metric("Theoretical Mean (r̄)", f"{metrics.theoretical_mean_radius:.3f} m")
    c2.metric("KS-Test p-value", f"{metrics.ks_pvalue:.3f}")
    c2.metric(
        "Simulated Mean",
        f"{metrics.empirical_r_bar:.3f} m",
        f"{metrics.rel_error_pct:.2f}% err",
        delta_color="off",
    )

if not res.is_empty:
    st.plotly_chart(plot_seed_shadow_and_distance(result=res, metrics=metrics), use_container_width=True)
else:
    st.warning("Zero seeds were produced across the simulated instances. Increase N or height class.")

# TO DO: Add tab which shows descriptive information regarding different species
# Plot variables across species and show tabular
