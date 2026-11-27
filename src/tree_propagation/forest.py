import math
from collections.abc import Mapping

import numpy as np

import tree_propagation.type_validation as validation
from tree_propagation.agent import TreeAgent
from tree_propagation.species_config import SpeciesConfig
from tree_propagation.types import AgentStatus


Cell = tuple[int, int]

DEFAULT_CELL_SIZE = 5.0

MOORE_OFFSETS: tuple[Cell, ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


class ForestGrid:
    """Rectangular forest with at most one tree per cell."""

    def __init__(
        self,
        rows: int,
        cols: int,
        cell_size: float = DEFAULT_CELL_SIZE,
    ) -> None:

        for name, value in (("rows", rows), ("cols", cols)):
            if not validation.is_strict_int(value) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")

        if not validation.is_positive_finite(cell_size):
            raise ValueError("cell_size must be positive and finite.")

        self.rows = rows
        self.cols = cols
        self.cell_size = float(cell_size)

        self._trees: dict[Cell, TreeAgent] = {}

    def __len__(self) -> int:
        return len(self._trees)

    def position_to_cell(self, x: float, y: float) -> Cell:
        """Map metre coordinates to grid indices."""

        if not validation.is_finite_real(x) or not validation.is_finite_real(y):
            raise ValueError("Coordinates must be finite real numbers.")

        return (
            math.floor(y / self.cell_size),
            math.floor(x / self.cell_size),
        )

    def in_bounds(self, cell: Cell) -> bool:
        row, col = cell

        return (
            0 <= row < self.rows
            and 0 <= col < self.cols
        )

    def cell_center(self, cell: Cell) -> tuple[float, float]:
        """Return the centre of a cell in metres."""

        if not self.in_bounds(cell):
            raise ValueError("Cell is outside the forest.")

        row, col = cell

        return (
            (col + 0.5) * self.cell_size,
            (row + 0.5) * self.cell_size,
        )

    def get_tree(self, cell: Cell) -> TreeAgent | None:
        return self._trees.get(cell)

    def is_occupied(self, cell: Cell) -> bool:
        return cell in self._trees

    def add_tree(self, tree: TreeAgent) -> None:
        cell = self.position_to_cell(tree.x, tree.y)

        if not self.in_bounds(cell):
            raise ValueError("Tree is outside the forest.")

        if self.is_occupied(cell):
            raise ValueError("Cell is already occupied.")

        if tree.status != AgentStatus.ALIVE:
            raise ValueError("Cannot add a dead tree.")

        self._trees[cell] = tree

    def remove_tree(self, cell: Cell) -> TreeAgent | None:
        return self._trees.pop(cell, None)

    def trees(self) -> tuple[TreeAgent, ...]:
        """Return a snapshot of the population membership."""

        return tuple(self._trees.values())

    """ === Moore Neighbourhood and Shade Methods === """
    def neighbours(self, cell: Cell) -> tuple[Cell, ...]:
        """Return valid cells in the Moore neighbourhood."""

        if not self.in_bounds(cell):
            raise ValueError("Cell is outside the forest.")

        row, col = cell
        neighbours = []

        for dr, dc in MOORE_OFFSETS:
            neighbour = (row + dr, col + dc)

            if self.in_bounds(neighbour):
                neighbours.append(neighbour)

        return tuple(neighbours)

    def calculate_shade(self, cell: Cell) -> float:
        """Calculate shade from taller neighbouring trees."""

        tree = self.get_tree(cell)

        height_class = (
            tree.height_class if tree is not None else 0
        )

        taller_neighbours = 0

        for neighbour in self.neighbours(cell):
            neighbour_tree = self.get_tree(neighbour)

            if (
                neighbour_tree is not None
                and neighbour_tree.height_class > height_class
            ):
                taller_neighbours += 1

        return taller_neighbours / len(MOORE_OFFSETS)
    """ =================================== """

    @classmethod
    def random(
        cls,
        rows: int,
        cols: int,
        num_trees: int,
        configs: Mapping[str, SpeciesConfig],
        rng: np.random.Generator,
        cell_size: float = DEFAULT_CELL_SIZE,
    ) -> "ForestGrid":

        forest = cls(rows, cols, cell_size)

        if (
            not validation.is_strict_int(num_trees)
            or not 0 <= num_trees <= rows * cols
        ):
            raise ValueError(
                "num_trees must be between zero and grid capacity."
            )

        species = tuple(configs.values())

        if num_trees and not species:
            raise ValueError("At least one species is required.")

        selected_cells = rng.choice(
            rows * cols,
            size=num_trees,
            replace=False,
        )

        for index in selected_cells:
            cell = divmod(int(index), cols)

            config = species[
                int(rng.integers(len(species)))
            ]

            height_class = int(
                rng.integers(1, config.maximal_height_class + 1)
            )

            age = height_class * config.years_per_height_class

            x, y = forest.cell_center(cell)

            tree = TreeAgent.from_age(
                species_config=config,
                age=age,
                x=x,
                y=y,
            )

            forest.add_tree(tree)

        return forest