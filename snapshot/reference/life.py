"""Exact Conway's Game of Life oracle and display-state helpers.

The default experiment is a 16x16 binary grid with fixed-dead boundaries.
Toroidal boundaries and other rectangular sizes are supported explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Iterable, Iterator, Sequence


DEFAULT_WIDTH = 16
DEFAULT_HEIGHT = 16


class Boundary(str, Enum):
    """How cells beyond the encoded grid are interpreted."""

    DEAD = "dead"
    TOROIDAL = "toroidal"


@dataclass(frozen=True, slots=True)
class Grid:
    """Immutable, row-major binary world state."""

    width: int
    height: int
    cells: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("grid dimensions must be positive")
        if len(self.cells) != self.width * self.height:
            raise ValueError(
                f"expected {self.width * self.height} cells, got {len(self.cells)}"
            )
        if any(cell not in (0, 1) for cell in self.cells):
            raise ValueError("cells must contain only integer bits 0 and 1")

    @classmethod
    def empty(
        cls, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT
    ) -> "Grid":
        return cls(width, height, (0,) * (width * height))

    @classmethod
    def from_live_cells(
        cls,
        live_cells: Iterable[tuple[int, int]],
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
    ) -> "Grid":
        cells = [0] * (width * height)
        for x, y in live_cells:
            if x < 0 or y < 0 or x >= width or y >= height:
                raise ValueError(f"live cell {(x, y)} is outside {width}x{height}")
            cells[y * width + x] = 1
        return cls(width, height, tuple(cells))

    @classmethod
    def from_rows(cls, rows: Sequence[Sequence[int]]) -> "Grid":
        if not rows or not rows[0]:
            raise ValueError("rows must not be empty")
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise ValueError("rows must be rectangular")
        cells = tuple(int(cell) for row in rows for cell in row)
        return cls(width, len(rows), cells)

    def at(self, x: int, y: int) -> int:
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            raise IndexError((x, y))
        return self.cells[y * self.width + x]

    def live_cells(self) -> frozenset[tuple[int, int]]:
        return frozenset(
            (index % self.width, index // self.width)
            for index, value in enumerate(self.cells)
            if value
        )


# Coordinates are relative to a pattern's upper-left corner.
PATTERNS: dict[str, tuple[tuple[int, int], ...]] = {
    "block": ((0, 0), (1, 0), (0, 1), (1, 1)),
    "blinker": ((0, 0), (1, 0), (2, 0)),
    "glider": ((1, 0), (2, 1), (0, 2), (1, 2), (2, 2)),
}


def place_pattern(
    name: str,
    x: int,
    y: int,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> Grid:
    """Create a blank grid with one named standard pattern."""

    try:
        pattern = PATTERNS[name.lower()]
    except KeyError as error:
        raise ValueError(f"unknown pattern: {name!r}") from error
    return Grid.from_live_cells(
        ((x + dx, y + dy) for dx, dy in pattern), width=width, height=height
    )


def local_rule(center: int, live_neighbors: int) -> int:
    """Return one Life output bit from the center bit and neighbor count."""

    if center not in (0, 1):
        raise ValueError("center must be 0 or 1")
    if not 0 <= live_neighbors <= 8:
        raise ValueError("live_neighbors must be between 0 and 8")
    return int(live_neighbors == 3 or (center == 1 and live_neighbors == 2))


def local_rule_from_neighborhood(neighborhood: Sequence[int]) -> int:
    """Apply Life to a row-major 3x3 neighborhood."""

    if len(neighborhood) != 9:
        raise ValueError("a local neighborhood must contain exactly 9 bits")
    if any(bit not in (0, 1) for bit in neighborhood):
        raise ValueError("a local neighborhood must contain only 0 and 1")
    center = neighborhood[4]
    return local_rule(center, sum(neighborhood) - center)


def all_neighborhoods() -> Iterator[tuple[int, ...]]:
    """Yield all 2^9 possible binary 3x3 neighborhoods."""

    yield from product((0, 1), repeat=9)


def _sample(grid: Grid, x: int, y: int, boundary: Boundary) -> int:
    if boundary is Boundary.TOROIDAL:
        return grid.cells[(y % grid.height) * grid.width + (x % grid.width)]
    if x < 0 or y < 0 or x >= grid.width or y >= grid.height:
        return 0
    return grid.cells[y * grid.width + x]


def neighbor_count(grid: Grid, x: int, y: int, boundary: Boundary) -> int:
    boundary = Boundary(boundary)
    return sum(
        _sample(grid, x + dx, y + dy, boundary)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if dx != 0 or dy != 0
    )


def step(grid: Grid, boundary: Boundary = Boundary.DEAD) -> Grid:
    """Advance the complete world by exactly one deterministic generation."""

    boundary = Boundary(boundary)
    next_cells = tuple(
        local_rule(
            grid.cells[y * grid.width + x],
            neighbor_count(grid, x, y, boundary),
        )
        for y in range(grid.height)
        for x in range(grid.width)
    )
    return Grid(grid.width, grid.height, next_cells)


def advance(
    grid: Grid, generations: int, boundary: Boundary = Boundary.DEAD
) -> Grid:
    if generations < 0:
        raise ValueError("generations must be non-negative")
    for _ in range(generations):
        grid = step(grid, boundary)
    return grid


def encode_state(grid: Grid) -> tuple[int, ...]:
    """Return the exact flat binary state consumed by the neural circuit."""

    return grid.cells


def decode_state(
    encoded: Sequence[int],
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> Grid:
    return Grid(width, height, tuple(encoded))


def step_encoded(
    encoded: Sequence[int],
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    boundary: Boundary = Boundary.DEAD,
) -> tuple[int, ...]:
    return encode_state(step(decode_state(encoded, width, height), boundary))


def render_ascii(grid: Grid, alive: str = "[]", dead: str = "  ") -> str:
    """Render a grid without introducing any game-state logic."""

    if not alive or not dead:
        raise ValueError("render glyphs must not be empty")
    return "\n".join(
        "".join(alive if grid.at(x, y) else dead for x in range(grid.width))
        for y in range(grid.height)
    )


def frame_u8(grid: Grid) -> tuple[tuple[int, ...], ...]:
    """Return a UI-ready grayscale frame: dead=0, alive=255."""

    return tuple(
        tuple(255 * grid.at(x, y) for x in range(grid.width))
        for y in range(grid.height)
    )


def rgba_bytes(
    grid: Grid,
    scale: int = 1,
    *,
    alive: tuple[int, int, int, int] = (255, 255, 255, 255),
    dead: tuple[int, int, int, int] = (0, 0, 0, 255),
) -> bytes:
    """Return tightly packed, nearest-neighbor-scaled RGBA display bytes."""

    if scale <= 0:
        raise ValueError("scale must be positive")
    if any(channel < 0 or channel > 255 for color in (alive, dead) for channel in color):
        raise ValueError("RGBA channels must be between 0 and 255")
    if len(alive) != 4 or len(dead) != 4:
        raise ValueError("RGBA colors must contain four channels")

    output = bytearray()
    for y in range(grid.height):
        row = bytearray()
        for x in range(grid.width):
            row.extend(alive if grid.at(x, y) else dead)
            if scale > 1:
                row.extend((alive if grid.at(x, y) else dead) * (scale - 1))
        for _ in range(scale):
            output.extend(row)
    return bytes(output)
