import pandas as pd

from src.tree_propagation.wind_config import WindRose


def windrose_to_dataframe(wind_rose: WindRose) -> pd.DataFrame:
    return pd.DataFrame(
        [{direction.name: prob for direction, prob in wind_rose.probabilities.items()}],
        index=["Probability"],
    )