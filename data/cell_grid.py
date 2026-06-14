"""Grid math shared by the cell-aware contact head (Variant B).

A frame is split into a ``CELL_GRID x CELL_GRID`` grid of selection regions,
indexed in reading order (row-major): index ``0`` is the top-left region,
``CELL_GRID - 1`` the top-right, and ``CELL_GRID*CELL_GRID - 1`` the bottom-right.
This matches the canonical 3x3 region order used by the dataset converter
(``top_left, top, top_right, left, center, right, bottom_left, bottom,
bottom_right``), so a region NAME's position equals its grid index.

Two inverse operations:

* ``pixel_to_region_index`` — used by the dataset loader to derive the selection
  label from the normalized contact point (u, v).
* ``region_index_center`` — used by the Cellpose teacher to turn a labeled region
  into a point it can match detected cell centroids against.

Coordinates are normalized to ``[0, 1]``: ``u`` is horizontal (column / x),
``v`` is vertical (row / y), with the origin at the top-left of the frame.
"""

from __future__ import annotations

from typing import Tuple

from config import vla_config as C


def num_regions(grid: int = C.CELL_GRID) -> int:
    return int(grid) * int(grid)


def pixel_to_region_index(u: float, v: float, grid: int = C.CELL_GRID) -> int:
    """Normalized (u, v) in [0, 1] -> region index in [0, grid*grid)."""
    grid = int(grid)
    col = min(max(int(float(u) * grid), 0), grid - 1)
    row = min(max(int(float(v) * grid), 0), grid - 1)
    return row * grid + col


def region_index_center(index: int, grid: int = C.CELL_GRID) -> Tuple[float, float]:
    """Region index -> the normalized (u, v) center of that grid cell."""
    grid = int(grid)
    index = min(max(int(index), 0), grid * grid - 1)
    row, col = divmod(index, grid)
    return ((col + 0.5) / grid, (row + 0.5) / grid)
