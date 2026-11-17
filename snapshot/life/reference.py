"""Small dependency-free oracle shared by the compiler tests and renderer.

The compiled circuit uses a 16 by 16 board with dead (not wrapping) boundaries. Rows are
unsigned 16-bit integers, bit 0 at the left edge. This module is deliberately ordinary Python:
it is the independently understandable specification that the neural circuit must equal.
"""

from __future__ import annotations

HEIGHT = 16
WIDTH = 16
MASK = (1 << WIDTH) - 1

# A glider, a blinker, and a still-life block make changes and invariants visible at once.
INITIAL_ROWS = (
    0b0000000000000000,
    0b0000000000000100,
    0b0000000000001000,
    0b0000000000001110,
    0b0000000000000000,
    0b0000000000000000,
    0b0000000001110000,
    0b0000000000000000,
    0b0000000000000000,
    0b0000000000000000,
    0b0000011000000000,
    0b0000011000000000,
    0b0000000000000000,
    0b0000000000000000,
    0b0000000000000000,
    0b0000000000000000,
)


def cell(rows: tuple[int, ...] | list[int], y: int, x: int) -> int:
    if not (0 <= y < HEIGHT and 0 <= x < WIDTH):
        return 0
    return (int(rows[y]) >> x) & 1


def step(rows: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """One exact Life generation with dead boundaries."""
    if len(rows) != HEIGHT:
        raise ValueError(f"expected {HEIGHT} rows, got {len(rows)}")
    out: list[int] = []
    for y in range(HEIGHT):
        row = 0
        for x in range(WIDTH):
            neighbours = sum(
                cell(rows, y + dy, x + dx)
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
                if dx or dy
            )
            alive = cell(rows, y, x)
            if neighbours == 3 or (alive and neighbours == 2):
                row |= 1 << x
        out.append(row & MASK)
    return tuple(out)


def render(rows: tuple[int, ...] | list[int], alive: str = "██", dead: str = "  ") -> str:
    return "\n".join(
        "".join(alive if (int(row) >> x) & 1 else dead for x in range(WIDTH))
        for row in rows
    )

