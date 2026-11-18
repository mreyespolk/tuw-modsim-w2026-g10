# Modeling and Simulation W2026: Tree Propagation in Wien
- **Course**: Modeling and Simulation
- **Semester**: W2026
- **Group**: 10
- **Project**: Tree Propagation in Wien

## Project Instructions

The project instructions are contained in [tree_propagation.pdf](./docs/tree_propagation.pdf).

# Requirements

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[notebook,dev]"
```

# Streamlit

Launch the streamlit app:

```sh
streamlit run app.py
```

## Taking screenshots of Streamlit

### Firefox
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