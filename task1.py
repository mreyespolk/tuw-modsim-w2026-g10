import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tree_propagation.agent import TreeAgent
from tree_propagation.tree_config import load_tree_configs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RNG = np.random.default_rng(seed=10)
SIMULATION_RUNS = 1_000_000
SPECIES_CONFIG_PATH = "./data/trees_vienna.csv"

tree_configs = load_tree_configs(SPECIES_CONFIG_PATH)

df_config =pd.DataFrame(tree_configs.values())

plt.figure(figsize=(12, 6))
sns.barplot(data=df_config.sort_values('maximal_height_class', ascending=False),
    x='species_name',
    y='maximal_height_class',
    palette='viridis')

plt.figure(figsize=(12, 6))
sns.barplot(data=df_config.sort_values('mean_life_span', ascending=False),
    x='species_name',
    y='mean_life_span',
    palette='viridis')

tree_agents = {}
for config in tree_configs.items():
    species_name, species_config = config
    species_agent = TreeAgent(
        species_config=species_config,
        age=species_config.maturity_age,
        height_class=4,
        x=0,
        y=0,
        rng=RNG
    )
    tree_agents[species_name] = species_agent
    logger.info(f"Created TreeAgent for species: {species_agent}")

    for _ in range(SIMULATION_RUNS):
        species_agent.produce_seeds()