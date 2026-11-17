from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROTOTYPE = Path(__file__).resolve().parent
PROJECT = PROTOTYPE.parent
for path in (PROJECT, PROJECT / "references" / "qwenputer", PROJECT / "references" / "qwenGPU"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from life.reference import INITIAL_ROWS, step
from prototype.circuit import DONE, LifeCircuit, TICK, build_trace


class PackedRowCircuitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.circuit = LifeCircuit()

    def assert_trace_exact(self, initial, generations: int) -> None:
        trace = build_trace(initial, generations)
        checked, bad = self.circuit.validate_trace(trace)
        self.assertEqual(checked, generations * 17)
        self.assertEqual(bad, [])

    def test_graph_fits_three_constructed_qwen_layers(self) -> None:
        self.assertEqual(self.circuit.n_layers(), 3)
        self.assertLessEqual(max(self.circuit.heads.values()), 16)
        self.assertLessEqual(max(self.circuit.units.values()), 3584)

    def test_known_patterns_for_100_generations(self) -> None:
        self.assert_trace_exact(INITIAL_ROWS, 100)

    def test_random_worlds_for_100_generations_each(self) -> None:
        rng = np.random.default_rng(713)
        for _ in range(4):
            rows = tuple(int(v) for v in rng.integers(0, 1 << 16, size=16))
            self.assert_trace_exact(rows, 100)

    def test_protocol_is_exactly_sixteen_rows_and_done(self) -> None:
        trace = build_trace(INITIAL_ROWS, 2)
        first_tick = trace.tokens.index(TICK)
        expected = step(INITIAL_ROWS)
        self.assertEqual(trace.tokens[first_tick + 1 : first_tick + 17], expected)
        self.assertEqual(trace.tokens[first_tick + 17], DONE)


if __name__ == "__main__":
    unittest.main()
