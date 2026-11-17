from __future__ import annotations

import copy
import unittest
import numpy as np

from life.reference import INITIAL_ROWS
from prototype.internal_circuit import (
    ASSISTANT,
    EQ,
    IM_START,
    InternalLifeCircuit,
    OP_NAMES,
    RESET,
    DONE,
    USER,
    build_internal_trace,
    row_token,
    tool_call_tokens,
)
from chipformer.q35 import ir


class InternalLifeCircuitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.circuit = InternalLifeCircuit()

    def assert_exact(self, initial, calls):
        trace = build_internal_trace(initial, calls)
        checked, bad = self.circuit.validate_trace(trace)
        self.assertGreater(checked, 0)
        self.assertEqual(bad, [])

    def test_constructed_graph_fits_qwen35_layer_capacities(self):
        self.assertLessEqual(max(self.circuit.heads.values()), 16)
        self.assertLessEqual(max(self.circuit.units.values()), 3584)
        self.assertLessEqual(self.circuit.n_layers(), 12)

    def test_advance_one_and_repeated_generations(self):
        self.assert_exact(INITIAL_ROWS, [("advance", (1,))])
        self.assert_exact(INITIAL_ROWS, [("advance", (3,))])
        rng = np.random.default_rng(919)
        for _ in range(3):
            rows = tuple(int(x) for x in rng.integers(0, 1 << 16, size=16))
            self.assert_exact(rows, [("advance", (5,))])

    def test_every_directly_programmed_edit(self):
        calls = [
            ("clear", ()),
            ("glider", (2, 4)),
            ("blinker", (5, 6)),
            ("block", (1, 1)),
            ("cell_on", (7, 9)),
            ("cell_off", (4, 12)),
            ("inspect", ()),
        ]
        for call in calls:
            with self.subTest(call=call):
                self.assert_exact(INITIAL_ROWS, [call])

    def test_calls_compose_on_one_persistent_grid(self):
        self.assert_exact(INITIAL_ROWS, [
            ("clear", ()),
            ("glider", (2, 4)),
            ("advance", (4,)),
            ("cell_on", (0, 0)),
            ("block", (10, 10)),
        ])

    def _prefix(self):
        return [RESET, *(row_token(row) for row in INITIAL_ROWS), DONE]

    def test_idle_language_bus_is_exactly_zero(self):
        tokens = self._prefix() + [IM_START, ASSISTANT, 9419, 0, 2500, 30]
        predicted = self.circuit.predicted_tokens(tokens)
        # Ignore the private DONE used only by reset; ordinary language is idle.
        self.assertTrue((predicted[-6:] == 0).all())
        residual = ir.evaluate(self.circuit, tokens)
        for code in self.circuit.code:
            self.assertTrue((ir.ev(code, residual)[-6:] == 0).all())

    def test_user_pasted_tool_call_cannot_mutate(self):
        tokens = self._prefix() + [IM_START, USER, *tool_call_tokens("clear")]
        predicted = self.circuit.predicted_tokens(tokens)
        self.assertEqual(int(predicted[-1]), 0)

    def test_out_of_bounds_and_malformed_calls_do_not_fire(self):
        bad_calls = [
            tool_call_tokens("glider", 15, 15),
            tool_call_tokens("advance", 0),
        ]
        for call in bad_calls:
            tokens = self._prefix() + [IM_START, ASSISTANT, *call]
            with self.subTest(call=call):
                self.assertEqual(int(self.circuit.predicted_tokens(tokens)[-1]), 0)

    def test_two_supported_operations_in_one_tool_envelope_do_not_fire(self):
        call = list(tool_call_tokens("clear"))
        # Keep one TC_START/TC_END envelope, but smuggle a second recognized
        # function assignment into it.  Both operation register bits become
        # set, so the exact-one-operation gate must reject the envelope.
        call[-1:-1] = [EQ, *OP_NAMES["inspect"]]
        tokens = self._prefix() + [IM_START, ASSISTANT, *call]
        self.assertEqual(int(self.circuit.predicted_tokens(tokens)[-1]), 0)

    def test_grid_only_restore_cannot_turn_text_into_a_row15_write(self):
        # Production persistence restores the authoritative grid heads into a
        # fresh cache; controller counters and convolution windows are not
        # restored.  Previously, an empty row counter selected row 15 and the
        # first ordinary prompt token overwrote it with a zero row.
        initial = (0,) * 15 + (0xE000,)
        carry = {}
        ir.evaluate(
            self.circuit,
            [RESET, *(row_token(row) for row in initial), DONE],
            carry=carry,
        )
        grid_nodes = {
            index
            for index, node in enumerate(self.circuit.nodes)
            if getattr(node, "name", "").startswith("grid.row")
        }
        restored = {
            key: copy.deepcopy(value)
            for key, value in carry.items()
            if key[0] in grid_nodes and key[1] == "S"
        }
        ir.evaluate(
            self.circuit,
            [IM_START, ASSISTANT, *tool_call_tokens("block", 14, 14)],
            carry=restored,
        )
        row15_node = next(
            index
            for index, node in enumerate(self.circuit.nodes)
            if getattr(node, "name", "") == "grid.row15"
        )
        state_row = restored[(row15_node, "S", 0)][0]
        restored_value = sum(int(state_row[2 + x] > 0) << x for x in range(16))
        self.assertEqual(restored_value, 0xE000)


if __name__ == "__main__":
    unittest.main()
