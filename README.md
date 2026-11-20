# Modeling and Simulation W2026: Tree Propagation in Wien

- **Course**: Modeling and Simulation
- **Semester**: W2026
- **Group**: 10
- **Project**: Tree Propagation in Wien

## Project overview

An agent-based model of how tree species spread across Vienna. Each tree is an agent that grows,
matures, produces and disperses seeds, and eventually dies. Seeds can establish new trees on a
grid, where they compete for space and light. Starting from the real Vienna tree inventory, the
model simulates how the species composition of the city's trees develops over time.

The full assignment is in [docs/tree_propagation.pdf](./docs/tree_propagation.pdf).

### Limitations 
The project will not handle city interference in the dispersal of seeds and establishement of trees, 
as it would require obtaining a fine-grained exclusion map, and height map of man-made constructions.

## Project planning

Project plan including responsibilities, status and dependencies:<br>
<https://docs.google.com/spreadsheets/d/1kVtIJiJGVDGRkX_ikg5S-Yd9biN36jSuWd2YMtzkjLM/edit?gid=0#gid=0>

## Current status

Implemented so far:

- Species parameters loaded and validated from [data/tree_species_config.csv](./data/tree_species_config.csv).
- Tree agents with maturity, seed production and wind, animal or gravity seed dispersal.
- A Monte Carlo simulation of a single parent tree's seed shadow, checked against the expected
  dispersal distribution.
- A Streamlit app to explore seed shadows, the wind rose and species traits.

Not yet implemented: growth and mortality, seedling establishment, the grid, multi-tree and
multi-year simulations, and loading the Vienna inventory ([data/trees_vienna.csv](./data/trees_vienna.csv)).

## Quickstart

Requires **Python ≥ 3.11**.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"   # [dev]: pytest, ruff, jupyter
```

## Interactive Dashboard
The interactive dashboard was creating using Streamlit. 

### Public dashboard
The public dashboard is hosted on Streamlit Community Cloud, and references the state of the Streamlit app in the `dev`.<br>
https://tuw-modsim-w2026-g10-f7w2jy8hjot5v3kxiwrp7o.streamlit.app/

### Running interactive dashboard locally
In order to run the dashboard locally to test modifications before modifying the public dashboard, the app can be run with the following command:
```sh
streamlit run app.py
```

## Spatial model conventions

The assignment defines the model on a grid of square cells (edge length Δ, default 5 m). Tree
agents keep exact positions; grid cells are derived from them:

- **Coordinates:** positions `(x, y)` are real-valued **meters** (`+x` = East, `+y` = North).
  Meters are the single spatial unit for positions, dispersal distances, cell size and plots.
- **Dispersal origin:** seeds are dispersed from the parent tree's **exact** `(x, y)`.
- **Seedling placement:** a tree established from a seed is created at the **exact** landing
  location.
- **Grid:** grid rules (one tree per cell, shade, seeds outside the grid are lost) use the cell
  *derived* from an exact position.

## Repository layout

```text
app.py                  Streamlit app
data/                   Species parameters and the Vienna tree inventory
docs/                   Assignment PDF and LaTeX report
src/tree_propagation/   Model package (agents, dispersal, simulation, metrics, plots)
tests/                  pytest suite
task*.ipynb             Notebooks for the assignment tasks
```

## Linting

```sh
ruff check .
```

# Testing

```sh
pytest
```

## Development notes

<details>
<summary>Taking full-page screenshots of the Streamlit app (Firefox)</summary>

1. Open the dev tools: `CTRL + SHIFT + I`
2. Open the console and run this to let Streamlit's containers overflow:
    ```javascript
    document.querySelectorAll('#js-plotly-tester').forEach(el => el.remove());
    let el = document.querySelector('[data-testid="stMainBlockContainer"]');
    while (el && el.style) {
    el.style.overflow = 'visible';
    el.style.overflowY = 'visible';
    el.style.height = 'auto';
    el.style.maxHeight = 'none';
    el.style.position = 'relative';
    el.style.inset = 'auto';
    el = el.parentElement;
    }
    ```
3. Click back on the page, open the Firefox screenshot tool (`CTRL + SHIFT + S`) and choose
   full page.

</details>
