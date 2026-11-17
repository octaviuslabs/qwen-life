"""Exact 16x16 Conway Life controller compiled into Qwen3.5 primitives.

This is an isolated controller-only prototype.  It uses Qwenputer's exact q35
intermediate representation, but every logical weight below is assigned by
construction: there is no dataset, loss, optimizer, or gradient step.

Protocol
--------

``RESET row0 ... row15 DONE`` loads a grid (each row is one 16-bit token).

``TICK`` predicts ``next_row0 ... next_row15 DONE``.  Feeding those predicted
tokens back writes the new rows into the recurrent grid bank.  A later ``TICK``
therefore advances the *new* grid, and the loop can continue indefinitely.

The generated row tokens are the feedback bus required by an ordinary causal
Qwen checkpoint.  The host clocks generation and displays recurrent state; it
does not evaluate Conway's rule.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
for dependency in (PROJECT / "references" / "qwenputer", PROJECT / "references" / "qwenGPU"):
    value = str(dependency)
    if value not in sys.path:
        sys.path.insert(0, value)

from chipformer.q35 import ir  # noqa: E402
from chipformer.q35.ir import A_BIT, A_REF, K, Chan  # noqa: E402
from chipformer.q35.prims import Circ  # noqa: E402
from qgpu.circuit.gadgets import counter, thermo  # noqa: E402


WIDTH = 16
HEIGHT = 16
VAL_COUNT = 1 << WIDTH
RESET = VAL_COUNT
TICK = RESET + 1
DONE = TICK + 1
VOCAB_SIZE = DONE + 1
VOCAB = list(range(VOCAB_SIZE))
CODE_BITS = 17
CODE_AMP = 128


def is_value_token(token: int) -> bool:
    return 0 <= int(token) < VAL_COUNT


def token_name(token: int) -> str:
    if is_value_token(token):
        return f"ROW(0x{token:04x})"
    return {RESET: "RESET", TICK: "TICK", DONE: "DONE"}.get(token, f"?{token}")


@dataclass(frozen=True)
class Trace:
    tokens: tuple[int, ...]
    generated: tuple[bool, ...]
    generations: tuple[tuple[int, ...], ...]


class LifeCircuit(Circ):
    """Three-layer exact circuit and the expressions useful to tests/monitoring."""

    def __init__(self) -> None:
        super().__init__()
        one = self.one

        # Full-token features are required because special tokens live above 2^16;
        # ordinary q35 features intentionally see only a token's low 16 bits.
        full = True
        self.is_val = self.token_feature(
            "tok.is_val", lambda t: int(0 <= t < VAL_COUNT), full=full
        )
        self.is_reset = self.token_feature("tok.reset", lambda t: int(t == RESET), full=full)
        self.is_tick = self.token_feature("tok.tick", lambda t: int(t == TICK), full=full)
        self.is_done = self.token_feature("tok.done", lambda t: int(t == DONE), full=full)
        self.value_bits = [
            self.token_feature(
                f"tok.value.b{x}",
                lambda t, x=x: ((t >> x) & 1) if 0 <= t < VAL_COUNT else 0,
                full=full,
            )
            for x in range(WIDTH)
        ]
        self.code = [
            self.token_feature(
                f"tok.code.b{i}",
                lambda t, i=i: CODE_AMP * (2 * ((t >> i) & 1) - 1),
                full=full,
            )
            for i in range(CODE_BITS)
        ]

        # Layer 0: exact row counter and an active-generation register.
        # k is zero at RESET/TICK/DONE and increments on each 16-bit row token.
        reset_counter = self.is_reset + self.is_tick + self.is_done
        self.counter = counter(self, "row.counter", reset_counter, self.is_val)
        self.gt = thermo(self, self.counter, reset_counter, HEIGHT, "row.k")

        control = self.is_reset + self.is_tick + self.is_done
        (self.active,) = self.slot_head(
            "generation.active",
            {0: control},
            {0: one},
            {2: self.is_tick},
            control,
        )

        # The i-th value token after RESET/TICK writes row i.  DONE resets the
        # counter before its head pass, so no non-value token can repeat row 15.
        self.write_row = [
            self.gt[i] - (self.gt[i + 1] if i + 1 < HEIGHT else 0 * one)
            for i in range(HEIGHT)
        ]

        # At an output position, k=0 selects row 0, ..., k=15 selects row 15;
        # k=16 means all rows were emitted and DONE is next.
        self.output_row = [one - self.gt[0]] + [
            self.gt[i - 1] - self.gt[i] for i in range(1, HEIGHT)
        ]

        # Layer 1: one recurrent head per row.  TICK reads all 256 cells;
        # RESET clears the bank; row payloads overwrite their indexed row.
        self.current: list[list] = []
        for y in range(HEIGHT):
            row = self.slot_head(
                f"grid.row{y}",
                {0: self.write_row[y]},
                {0: self.is_tick},
                {2 + x: bit for x, bit in enumerate(self.value_bits)},
                self.write_row[y],
                a=238 * self.is_reset,
            )
            self.current.append(row)

        # The Life rule is a two-threshold exact MLP.  For a cell,
        # z = 2 * live-neighbour-count + current-cell is an integer 0..17;
        # the next cell is alive exactly for z in {5, 6, 7}.
        self.next_grid: list[list] = []
        for y in range(HEIGHT):
            out_row = []
            for x in range(WIDTH):
                neighbours = sum(
                    (
                        self.current[yy][xx]
                        for yy in range(max(0, y - 1), min(HEIGHT, y + 2))
                        for xx in range(max(0, x - 1), min(WIDTH, x + 2))
                        if (yy, xx) != (y, x)
                    ),
                    0 * one,
                )
                z = 2 * neighbours + self.current[y][x]
                out_row.append(self.node(self.t_in(z, 5, 8, 0, 17), f"life[{y},{x}]"))
            self.next_grid.append(out_row)

        # Layer 2: capture the complete next grid on TICK.  During the 16-row
        # trace every head is queried, but only the selected row's output gate
        # is open.  All heads add into the same 16 selected-row dimensions.
        self.selected: list | None = None
        for y in range(HEIGHT):
            v = {0: A_REF * self.is_tick, 1: A_REF * self.is_tick}
            v.update({2 + x: A_BIT * self.next_grid[y][x] for x in range(WIDTH)})
            z = {2 + x: K * self.output_row[y] for x in range(WIDTH)}
            outs = [(2 + x, 2.0**7) for x in range(WIDTH)]
            result = self.head(
                f"next.row{y}",
                {0: self._select(self.active)},
                {0: self._select(self.is_tick)},
                v,
                238 * self.is_reset,
                z,
                outs,
                into=self.selected,
            )
            if self.selected is None:
                self.selected = result
        assert self.selected is not None

        # k=16 while active is the position after the sixteenth emitted row.
        self.done_now = self.and_([self.active, self.gt[-1]], "emit.done")
        done_or_idle = one - self.active + self.done_now

        # Desired next-token bits.  A row token is simply the selected 16-bit
        # row.  Outside a row position, predict DONE (65536 + 2).
        self.output_bits = []
        for i in range(CODE_BITS):
            bit = self.selected[i] if i < WIDTH else 0 * one
            if (DONE >> i) & 1:
                bit = bit + done_or_idle
            self.output_bits.append(bit)

        # Clear the current token's code and install the desired next-token
        # code.  This is the exact output gadget used by Qwenputer's q35 tests.
        scale = 2**12
        for i in range(CODE_BITS):
            self.ffn(
                [
                    (2 * scale, one, self.output_bits[i]),
                    (-scale, one, one),
                    (-1, one, self.code[i]),
                    (1, one, -self.code[i]),
                ],
                f"exit.code{i}",
                into=self.code[i],
            )

        # Helpful invariants for compiler tests and the merger.
        assert self.n_layers() == 3, self.stats()

    def _select(self, value):
        """Exact key/query preactivation: +K for one and -K for zero."""
        if isinstance(value, Chan):
            return Chan(value.expr * (2 * K) - self.one * K, value.taps)
        return value * (2 * K) - self.one * K

    def predicted_tokens(self, tokens: Sequence[int], *, check: bool = True) -> np.ndarray:
        """Circuit next-token prediction at every input position."""
        residual = ir.evaluate(self, list(tokens), check=check)
        bits = np.stack([ir.ev(bit, residual) for bit in self.output_bits], axis=1)
        if not np.isin(bits, [0.0, 1.0]).all():
            bad = np.unique(bits[~np.isin(bits, [0.0, 1.0])])
            raise AssertionError(f"output bits are not binary: {bad[:8]}")
        powers = 1 << np.arange(CODE_BITS, dtype=np.int64)
        return bits.astype(np.int64) @ powers

    def validate_trace(self, trace: Trace, *, check: bool = True) -> tuple[int, list[tuple]]:
        predicted = self.predicted_tokens(trace.tokens, check=check)
        want = np.asarray(trace.tokens[1:] + (-1,), dtype=np.int64)
        generated = np.asarray(trace.generated[1:] + (False,), dtype=bool)
        positions = np.flatnonzero(generated)
        bad = positions[predicted[positions] != want[positions]]
        detail = [
            (
                int(pos),
                token_name(trace.tokens[pos]),
                token_name(int(want[pos])),
                token_name(int(predicted[pos])),
            )
            for pos in bad[:20]
        ]
        return int(positions.size), detail


def build_trace(initial_rows: Iterable[int], generations: int) -> Trace:
    """Reference trace whose generated positions are the circuit-owned rows/DONE."""
    from life.reference import step

    rows = tuple(int(row) for row in initial_rows)
    if len(rows) != HEIGHT or any(not 0 <= row < VAL_COUNT for row in rows):
        raise ValueError("expected sixteen unsigned 16-bit rows")
    if generations < 0:
        raise ValueError("generations must be non-negative")

    tokens = [RESET, *rows, DONE]
    generated = [False] * len(tokens)
    worlds = [rows]
    for _ in range(generations):
        tokens.append(TICK)
        generated.append(False)
        rows = step(rows)
        worlds.append(rows)
        tokens.extend(rows)
        generated.extend([True] * HEIGHT)
        tokens.append(DONE)
        generated.append(True)
    return Trace(tuple(tokens), tuple(generated), tuple(worlds))


def decode_code_rows(code_values: np.ndarray) -> np.ndarray:
    """Decode positive/negative output-code rails, useful for tiny-model hooks."""
    bits = (np.asarray(code_values) > 0).astype(np.int64)
    return bits @ (1 << np.arange(bits.shape[-1], dtype=np.int64))


__all__ = [
    "CODE_BITS",
    "DONE",
    "HEIGHT",
    "LifeCircuit",
    "RESET",
    "TICK",
    "Trace",
    "VAL_COUNT",
    "VOCAB",
    "VOCAB_SIZE",
    "WIDTH",
    "build_trace",
    "decode_code_rows",
    "token_name",
]
