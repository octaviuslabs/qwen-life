import random
import unittest

import torch

from analytic_net import AnalyticLifeNet, grid_from_tensor, tensor_from_grid
from life import (
    Boundary,
    Grid,
    PATTERNS,
    advance,
    all_neighborhoods,
    decode_state,
    encode_state,
    frame_u8,
    local_rule_from_neighborhood,
    place_pattern,
    rgba_bytes,
    step,
    step_encoded,
)


class LifeOracleTests(unittest.TestCase):
    def test_all_512_local_neighborhoods(self) -> None:
        live_outputs = 0
        neighborhoods = list(all_neighborhoods())
        self.assertEqual(len(neighborhoods), 512)

        for bits in neighborhoods:
            center = bits[4]
            neighbors = sum(bits) - center
            definition = int(neighbors == 3 or (center == 1 and neighbors == 2))
            result = local_rule_from_neighborhood(bits)
            self.assertEqual(result, definition)
            live_outputs += result

        # C(8, 3) births + C(8, 2)+C(8, 3) survivals.
        self.assertEqual(live_outputs, 140)

    def test_standard_patterns(self) -> None:
        self.assertEqual(len(PATTERNS["block"]), 4)
        self.assertEqual(len(PATTERNS["blinker"]), 3)
        self.assertEqual(len(PATTERNS["glider"]), 5)

        block = place_pattern("block", 6, 6)
        self.assertEqual(step(block), block)

        blinker = place_pattern("blinker", 6, 7)
        self.assertNotEqual(step(blinker), blinker)
        self.assertEqual(advance(blinker, 2), blinker)

        glider = place_pattern("glider", 3, 3)
        expected = Grid.from_live_cells(
            ((x + 4, y + 4) for x, y in PATTERNS["glider"])
        )
        self.assertEqual(advance(glider, 4), expected)

    def test_boundary_modes_are_explicit(self) -> None:
        edge_blinker = Grid.from_live_cells(((7, 0), (8, 0), (9, 0)))
        dead_result = step(edge_blinker, Boundary.DEAD)
        torus_result = step(edge_blinker, Boundary.TOROIDAL)
        self.assertEqual(dead_result.live_cells(), {(8, 0), (8, 1)})
        self.assertEqual(torus_result.live_cells(), {(8, 15), (8, 0), (8, 1)})

    def test_state_encoding_round_trip_and_fixed_contract(self) -> None:
        grid = place_pattern("glider", 2, 4)
        encoded = encode_state(grid)
        self.assertEqual(len(encoded), 256)
        self.assertEqual(decode_state(encoded), grid)
        self.assertEqual(step_encoded(encoded), encode_state(step(grid)))

        with self.assertRaises(ValueError):
            decode_state(encoded[:-1])
        with self.assertRaises(ValueError):
            decode_state(encoded[:-1] + (2,))

    def test_renderer_api_contains_no_state_transition(self) -> None:
        grid = place_pattern("block", 0, 0, width=2, height=2)
        self.assertEqual(frame_u8(grid), ((255, 255), (255, 255)))
        frame = rgba_bytes(grid, scale=3)
        self.assertEqual(len(frame), 2 * 3 * 2 * 3 * 4)
        self.assertEqual(frame[:4], bytes((255, 255, 255, 255)))


class AnalyticNetworkTests(unittest.TestCase):
    def test_no_parameter_is_trainable(self) -> None:
        for boundary in Boundary:
            model = AnalyticLifeNet(boundary)
            self.assertTrue(tuple(model.parameters()))
            self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))

    def test_network_is_exact_on_all_512_neighborhoods(self) -> None:
        neighborhoods = list(all_neighborhoods())
        states = torch.tensor(neighborhoods, dtype=torch.float32).reshape(512, 1, 3, 3)
        expected = torch.tensor(
            [local_rule_from_neighborhood(bits) for bits in neighborhoods],
            dtype=torch.float32,
        )

        for boundary in Boundary:
            model = AnalyticLifeNet(boundary)
            actual = model(states)[:, 0, 1, 1]
            self.assertTrue(torch.equal(actual, expected))

    def test_randomized_multistep_worlds_match_oracle_exactly(self) -> None:
        rng = random.Random(0xC0FFEE)
        for boundary in Boundary:
            model = AnalyticLifeNet(boundary)
            for _case in range(12):
                initial = Grid(
                    16,
                    16,
                    tuple(int(rng.random() < 0.31) for _ in range(16 * 16)),
                )
                oracle = initial
                neural = tensor_from_grid(initial)
                for _generation in range(20):
                    oracle = step(oracle, boundary)
                    neural = model(neural)
                    expected = tensor_from_grid(oracle)
                    self.assertTrue(torch.equal(neural, expected))

                self.assertEqual(grid_from_tensor(neural), oracle)


if __name__ == "__main__":
    unittest.main()
