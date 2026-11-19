# Modeling and Simulation W2026: Tree Propagation in Wien
- **Course**: Modeling and Simulation
- **Semester**: W2026
- **Group**: 10
- **Project**: Tree Propagation in Wien

## Project Instructions

The project instructions are contained in [tree_propagation.pdf](./docs/tree_propagation.pdf).

## Project planning

Project plan including responsabilities, status and dependencies.

https://docs.google.com/spreadsheets/d/1kVtIJiJGVDGRkX_ikg5S-Yd9biN36jSuWd2YMtzkjLM/edit?gid=0#gid=0

## Spatial model conventions

The project instructions define the model on a grid of square cells (edge length Δ, default 5 m). Tree agents keep exact positions grid cell coordinates are derived from them:

- **Coordinates:** a tree's position `(x, y)` is stored as real-valued **meters** (`+x` = East, `+y` = North). Meters are the single spatial unit for positions, dispersal distances, cell size and plots.
- **Dispersal origin:** seeds are always dispersed from the parent tree's **exact** `(x, y)`. Landing position = parent `(x, y)` + sampled offset `(dx, dy)`.
- **Seedling placement:** a tree established from a seed is created at the **exact** landing location.
- **Grid:** cells are derived from the meter coordinates using the cell size Δ. Grid rules (one tree per cell, occupancy, Moore-neighbourhood shade, seeds outside the grid are lost) use the cell *derived* from a exact metric position.

## Requirements

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[notebook,dev]"
```

## Streamlit

Launch the streamlit app:

```sh
streamlit run app.py
```

### Taking screenshots of Streamlit

#### Firefox
1. Open the dev tools: `CTRL + SHIFT + I`
2. Open the console and use the command to allow streamlit to overflow
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
3. Click back on the page and open the Firefox screenshot tool `CTRL+SHIFT+S`, then select the option for full page