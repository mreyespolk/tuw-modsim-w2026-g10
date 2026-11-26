"""Smoke and consistency tests for the Plotly figure builders.

Run with: python -m pytest -v tests/test_visualization_plotly.py
"""

import dataclasses

import numpy as np
import plotly.graph_objects as go
import pytest

import tree_propagation.visualization.plotly as vpl
from tree_propagation.dispersal import compute_mean_dispersal_distance
from tree_propagation.environment import DEFAULT_ENVIRONMENT_STATE, EnvironmentState
from tree_propagation.metrics import evaluate_seed_shadow
from tree_propagation.simulation import SeedDispersalSimulation, SeedDispersalSimulationParams
from tree_propagation.species_config import SPECIES_TRAIT_LABELS, SpeciesConfig
from tree_propagation.tables import build_species_trait_table
from tree_propagation.types import DispersalType
from tree_propagation.wind_config import DEFAULT_WIND_ROSE


@pytest.fixture(scope="module")
def traits_df(tree_species_config: dict[str, SpeciesConfig]):
    return build_species_trait_table(tree_species_config)


# region Shared helpers
def test_every_dispersal_type_has_a_style_and_facet():
    assert set(vpl.DISPERSAL_STYLES) == set(DispersalType)
    assert set(vpl.FACET_ORDER) == set(DispersalType)


def test_facet_marker_symbols_cycle_independently_per_facet():
    wind, animal = DispersalType.WIND, DispersalType.ANIMAL
    symbols = vpl._facet_marker_symbols([wind, animal, wind, animal, wind])
    s = vpl.MARKER_SYMBOLS
    assert symbols == [s[0], s[0], s[1], s[1], s[2]]
# endregion


# region Smoke tests
def test_plot_wind_rose():
    fig = vpl.plot_wind_rose(DEFAULT_WIND_ROSE)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1 # type: ignore


@pytest.mark.parametrize("num_instances", [0, 50])
def test_plot_seed_shadow_handles_empty_and_non_empty(ahorn_species_config: SpeciesConfig, rng, num_instances: int):
    env_state = DEFAULT_ENVIRONMENT_STATE
    sim_params = SeedDispersalSimulationParams(
        config=ahorn_species_config,
        environment_state=env_state,
        num_instances=num_instances,
    )
    simulation = SeedDispersalSimulation(
        params=sim_params,
        rng=rng
    )
    result = simulation.run()

    fig = vpl.plot_simulated_seed_shadow_and_distance_histogram(result, evaluate_seed_shadow(result))
    assert isinstance(fig, go.Figure)
    trace_names = {t.name for t in fig.data} # type: ignore
    assert ("Seeds" in trace_names) == (not result.is_empty)


@pytest.mark.parametrize(
    "plot_fn",
    [
        vpl.plot_pioneer_climax_tradeoff,
        vpl.plot_species_lifecycle_timelines,
        vpl.plot_species_trait_heatmap,
        vpl.plot_species_3d_scatter,
    ],
)
def test_trait_table_plots_return_figures(traits_df, plot_fn):
    fig = plot_fn(traits_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0 # type: ignore


def test_species_3d_scatter_accepts_any_trait_axes(traits_df):
    keys = list(SPECIES_TRAIT_LABELS)
    fig = vpl.plot_species_3d_scatter(traits_df, x_key=keys[-1], y_key=keys[-2], z_key=keys[-3])
    assert fig.layout.scene.xaxis.title.text == SPECIES_TRAIT_LABELS[keys[-1]]
# endregion


# region Faceted lifespan charts
def test_growth_over_time_one_trace_per_species_in_correct_facet(tree_species_config, traits_df):
    fig = vpl.plot_growth_over_time(traits_df)
    assert len(fig.data) == len(tree_species_config) # type: ignore
    for trace in fig.data:
        dtype = tree_species_config[trace.name].dispersal_type # type: ignore
        expected_axis = "x" if vpl._facet_column(dtype) == 1 else f"x{vpl._facet_column(dtype)}"
        assert trace.xaxis == expected_axis # type: ignore


@pytest.mark.parametrize("wind_strength", [1, 7])
def test_dispersal_reach_matches_dispersal_model(tree_species_config, wind_strength: int):
    """The chart must plot exactly the r̄ the simulation uses, for every species and age."""
    fig = vpl.plot_dispersal_reach_by_age(tree_species_config, wind_strength=wind_strength)
    env = EnvironmentState(wind_strength=wind_strength)

    assert len(fig.data) == len(tree_species_config) # type: ignore
    for trace in fig.data:
        cfg = tree_species_config[trace.name] # type: ignore
        expected = [
            compute_mean_dispersal_distance(cfg, height_class=int(k), environment=env) for k in trace.customdata # type: ignore
        ]
        np.testing.assert_allclose(trace.y, expected) # type: ignore
        assert trace.line.color == vpl.DISPERSAL_STYLES[cfg.dispersal_type].color # type: ignore
        assert trace.line.dash == vpl.DISPERSAL_STYLES[cfg.dispersal_type].dash # type: ignore


def test_dispersal_reach_default_wind_strength_is_one(tree_species_config):
    assert "s = 1)" in vpl.plot_dispersal_reach_by_age(tree_species_config).layout.title.text
# endregion


# region Axis ranges adapt to data
def test_pioneer_plot_y_range_covers_late_maturing_species(tree_species_config, ahorn_species_config):
    late = dataclasses.replace(ahorn_species_config, species_name="Late", maturity_age=80, mean_life_span=300)
    traits = build_species_trait_table({**tree_species_config, "Late": late})

    fig = vpl.plot_pioneer_climax_tradeoff(traits)

    assert fig.layout.yaxis.range[1] > 80
# endregion
