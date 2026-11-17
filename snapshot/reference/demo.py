"""Small terminal demonstration using the exact oracle."""

from __future__ import annotations

import argparse
import time

from life import Boundary, place_pattern, render_ascii, step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary", choices=[item.value for item in Boundary], default="dead")
    parser.add_argument("--generations", type=int, default=32)
    parser.add_argument("--delay", type=float, default=0.12)
    args = parser.parse_args()

    boundary = Boundary(args.boundary)
    grid = place_pattern("glider", 1, 1)
    for generation in range(args.generations + 1):
        print("\x1b[2J\x1b[H", end="")
        print(f"Conway's Game of Life — generation {generation} — {boundary.value} boundary")
        print(render_ascii(grid))
        if generation != args.generations:
            time.sleep(args.delay)
            grid = step(grid, boundary)


if __name__ == "__main__":
    main()
