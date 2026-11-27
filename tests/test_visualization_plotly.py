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
from tree_propagation.models import ForestSnapshot, TreeSnapshot


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

# region Forest simulation visualization

def test_forest_animation_frames():
    history = [
        ForestSnapshot(year=0, trees=()),
        ForestSnapshot(year=1, trees=()),
        ForestSnapshot(year=2, trees=()),
    ]

    fig = vpl.plot_forest_animation(
        history=history,
        rows=10,
        cols=10,
        cell_size=5.0,
        species_names=["Ahorn"],
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.frames) == 3
    assert [frame.name for frame in fig.frames] == ["0", "1", "2"]


def test_forest_animation_tree_coordinates():
    tree = TreeSnapshot(
        species="Ahorn",
        age=24,
        height_class=2,
        x=12.5,
        y=7.5,
    )

    history = [
        ForestSnapshot(year=0, trees=(tree,))
    ]

    fig = vpl.plot_forest_animation(
        history=history,
        rows=10,
        cols=10,
        cell_size=5.0,
        species_names=["Ahorn"],
    )

    np.testing.assert_allclose(fig.data[0].x, [12.5])
    np.testing.assert_allclose(fig.data[0].y, [7.5])


def test_forest_animation_preserves_species_traces():
    history = [
        ForestSnapshot(
            year=0,
            trees=(
                TreeSnapshot(
                    species="Ahorn",
                    age=24,
                    height_class=2,
                    x=2.5,
                    y=2.5,
                ),
            ),
        ),
        ForestSnapshot(year=1, trees=()),
    ]

    fig = vpl.plot_forest_animation(
        history=history,
        rows=10,
        cols=10,
        cell_size=5.0,
        species_names=["Ahorn", "Buche"],
    )

    for frame in fig.frames:
        assert len(frame.data) == 2
        assert [trace.name for trace in frame.data] == [
            "Ahorn", "Buche"
        ]


@pytest.mark.parametrize(
    ("rows", "cols", "cell_size"),
    [
        (10, 20, 5.0),
        (20, 10, 5.0),
        (10, 10, 2.0),
    ],
)
def test_forest_animation_boundaries(
    rows: int, cols: int, cell_size: float
):
    fig = vpl.plot_forest_animation(
        history=[ForestSnapshot(year=0, trees=())],
        rows=rows,
        cols=cols,
        cell_size=cell_size,
        species_names=["Ahorn"],
    )

    assert tuple(fig.layout.xaxis.range) == (
        0, cols * cell_size
    )
    assert tuple(fig.layout.yaxis.range) == (
        0, rows * cell_size
    )


def test_forest_population_plot():
    tree = TreeSnapshot(
        species="Ahorn",
        age=24,
        height_class=2,
        x=2.5,
        y=2.5,
    )

    history = [
        ForestSnapshot(year=0, trees=(tree,)),
        ForestSnapshot(year=1, trees=()),
    ]

    fig = vpl.plot_forest_population(
        history=history,
        species_names=["Ahorn"],
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2

    assert fig.data[0].name == "Ahorn"
    assert fig.data[-1].name == "Total population"

    np.testing.assert_array_equal(fig.data[0].x, [0, 1])
    np.testing.assert_array_equal(fig.data[0].y, [1, 0])
    np.testing.assert_array_equal(fig.data[-1].y, [1, 0])


def test_forest_animation_rejects_empty_history():
    with pytest.raises(ValueError, match="history"):
        vpl.plot_forest_animation(
            history=[],
            rows=10,
            cols=10,
            cell_size=5.0,
            species_names=["Ahorn"],
        )


def test_forest_animation_marker_size_reflects_height():
    history = [
        ForestSnapshot(
            year=0,
            trees=(
                TreeSnapshot("Ahorn", 0, 0, 2.5, 2.5),
                TreeSnapshot("Ahorn", 24, 2, 7.5, 7.5),
                TreeSnapshot("Ahorn", 60, 5, 12.5, 12.5),
            ),
        )
    ]

    fig = vpl.plot_forest_animation(
        history=history,
        rows=10,
        cols=10,
        cell_size=5.0,
        species_names=["Ahorn"],
    )

    sizes = fig.data[0].marker.size

    assert 0 < sizes[0] < sizes[1] < sizes[2]


# endregion