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

The project will not handle city interference in the dispersal of seeds and establishment of trees,
as it would require obtaining a fine-grained exclusion map, and height map of man-made constructions.

## Project planning

Project plan including responsibilities, status and dependencies:<br>
<https://docs.google.com/spreadsheets/d/1kVtIJiJGVDGRkX_ikg5S-Yd9biN36jSuWd2YMtzkjLM/edit?gid=0#gid=0>

### Current status

Progress against the assignment tasks ([docs/tree_propagation.pdf](./docs/tree_propagation.pdf)):

## Agent model

Each tree is an agent with a species σ, an age a in years, a height class k and an exact position
(x, y) in meters. One time step is one year, and all trees are updated simultaneously from the state
at the beginning of the year. Symbols are listed in [Species parameters](#species-parameters).

1. **Height class.** A tree climbs one class every T years up to its species' maximum K:
   `k = min(⌊a / T⌋, K)`. If only the height class is known (e.g. from the tree register), the age
   is set to `a = k · T`.
2. **Seed production.** A tree is mature if `a ≥ A`. A mature tree with `k ≥ 1` produces
   `N ~ Poisson(φ · k)` seeds.
3. **Seed dispersal.** Every seed travels a distance `r ~ Exp` with mean r̄ from the parent's exact
   position:
   - *Wind:* `r̄ = s · f · k` (s = wind strength). The direction is drawn from the wind rose: pick
     one of 8 sectors by its frequency, draw an angle uniformly within the 45° sector, then turn it
     180°. The wind rose gives where the wind blows *from*; seeds fly downwind.
   - *Animal / gravity:* `r̄ = d` (fixed per species) and the direction is uniform.
4. **Establishment** *(not yet implemented).* A seed landing on an occupied cell is lost. On an
   empty cell it establishes with probability `p_est · (1 − (1 − τ) · S)`, where S is the fraction
   of the 8 neighbouring cells holding a taller tree. If several seeds establish in one cell, one is
   chosen at random.
5. **Growth** *(not yet implemented).* Every tree ages by one year.
6. **Mortality** *(not yet implemented).* A tree dies with probability `1 / L + β · (1 − τ) · S`.

Global defaults from the assignment: Δ = 5 m, s = 10, p_est = 0.5, β = 0.3.

### Spatial model conventions

The assignment defines the model on a grid of square cells (edge length Δ, default 5 m). Tree
agents keep exact positions; grid cells are derived from them:

- **Coordinates:** positions `(x, y)` are real-valued **meters** (`+x` = East, `+y` = North).
  Meters are the single spatial unit for positions, dispersal distances, cell size and plots.
- **Dispersal origin:** seeds are dispersed from the parent tree's **exact** `(x, y)`.
- **Seedling placement:** a tree established from a seed is created at the **exact** landing
  location.
- **Grid:** grid rules (one tree per cell, shade, seeds outside the grid are lost) use the cell
  *derived* from an exact position.
- **Bearings** are measured in degrees clockwise from North.

## Data

### Species parameters

[data/tree_species_config.csv](./data/tree_species_config.csv) holds one row per species and is the
single source of truth for the trait values (they may deviate from Table 1 of the assignment, _as modified by raising issues with the initial table with the project supervisor_). It is loaded and validated by `load_tree_configs_from_csv()` in
[tree_config.py](./src/tree_propagation/tree_config.py). The column-to-field mapping below is
defined once in `CSV_COLUMNS` there; keep this table in sync when changing it.

| CSV column  | Symbol | `TreeConfig` field        | Meaning                                                   |
|-------------|--------|---------------------------|-----------------------------------------------------------|
| `species`   | σ      | `species_name`            | Species name (German), unique key                         |
| `dispersal` | –      | `dispersal_type`          | `wind`, `animal` or `gravity`                             |
| `f`         | f      | `wind_spreading_factor`   | Spreading factor; required for `wind`, empty otherwise    |
| `d_m`       | d      | `mean_dispersal_distance` | Mean dispersal distance [m]; required for `animal`/`gravity`, empty otherwise |
| `phi`       | φ      | `fecundity`               | Fecundity factor (seeds per year per height class)        |
| `A_yr`      | A      | `maturity_age`            | Age at maturity [yr]                                      |
| `T_yr`      | T      | `years_per_height_class`  | Years per height class [yr]                               |
| `K`         | K      | `maximal_height_class`    | Maximal height class                                      |
| `L_yr`      | L      | `mean_life_span`          | Mean life span [yr]; must exceed A                        |
| `tau`       | τ      | `shade_tolerance`         | Shade tolerance in [0, 1]                                 |

### Vienna tree register

[data/trees_vienna.csv](./data/trees_vienna.csv) is the preprocessed Vienna tree register
(*Baumkataster*) provided with the assignment for Task 5. Each row is one tree:

| Column         | Meaning                                                          |
|----------------|------------------------------------------------------------------|
| `district`     | Vienna district number (1–23)                                    |
| `x_m`, `y_m`   | Position in meters east and north of a fixed reference point     |
| `species`      | One of the species in the species parameter CSV                 |
| `height_class` | Height class k                |

## Quickstart

Requires **Python ≥ 3.11**.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"   # [dev]: pytest, ruff, jupyter

pytest                    # run the test suite
streamlit run app.py      # launch the dashboard locally
```

## Interactive dashboard

The interactive dashboard was created using Streamlit.

### Public dashboard

The public dashboard is hosted on Streamlit Community Cloud and reflects the state of the Streamlit
app on the `dev` branch.<br>
https://tuw-modsim-w2026-g10-f7w2jy8hjot5v3kxiwrp7o.streamlit.app/

### Running the dashboard locally

To test modifications locally before they reach the public dashboard, run:

```sh
streamlit run app.py
```

### App overview

**Sidebar (global parameters):** random seed (with a *New seed* button), wind strength s, number of
Monte Carlo instances N, and whether the plot axes are shared across the comparison columns.

- **Seed Shadow Simulation** – Compares three species side by side (by default one per dispersal
  type), each with its own height class k. Every column shows the summary metrics (total seeds,
  theoretical r̄, simulated mean distance, KS-test p-value), the 2D seed shadow, the distance
  distribution against the Exp(r̄) density, directional transects along the compass axes, and the
  species traits.
- **Wind Rose** – Polar plot and table of the wind rose sector probabilities (direction the wind
  blows *from*).
- **Species Overview** – Compares the traits of all species at a chosen reference height class:
  pioneer–climax trade-off, life-cycle timelines, dispersal reach and height class over age, a trait
  heatmap, a 3D trait-space explorer and the full parameter table.

## Repository layout

```text
app.py                  Streamlit app
data/                   Species parameters and the Vienna tree inventory
docs/                   Assignment PDF and LaTeX report
src/tree_propagation/   Model package (agents, dispersal, simulation, metrics, plots)
tests/                  pytest suite
task*.ipynb             Notebooks for the assignment tasks
```

## Git workflow

- **`main`** – Stable state; updated from `dev` at milestones (e.g. a finished task).
- **`dev`** – Integration branch. The public Streamlit dashboard is deployed from it.
- **`dev-<initials>`** – Personal working branches (e.g. `dev-MRP`), created from `dev`.

Typical cycle:

1. Update your branch from `dev` before starting new work.
2. Keep commits small and focused; prefix messages with the area they touch
   (e.g. `tree agent: …`, `viz: …`, `task 1: …`).
3. Run `ruff check .` and `pytest` before pushing.
4. Open a pull request into `dev` and have it reviewed by another group member.
5. Check the public dashboard after the merge.

## Linting

```sh
ruff check .
```

## Testing

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
