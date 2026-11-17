"""Analytically programmed native-tool controller for a 16x16 Life world.

Qwen chooses and writes a native ``<tool_call>``. Constructed DeltaNet/MLP
weights decode the call and update the recurrent grid. Nothing here is learned.
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

from chipformer.dsl.graph import Expr  # noqa: E402
from chipformer.q35 import ir  # noqa: E402
from chipformer.q35.ir import A_BIT, A_REF, K, Chan  # noqa: E402
from chipformer.q35.prims import Circ  # noqa: E402
from qgpu.circuit.gadgets import counter, thermo  # noqa: E402

WIDTH = HEIGHT = 16
VAL_COUNT = 1 << WIDTH
BASE_VOCAB = 248_320
ROW_BASE = 1 << 18
RESET, TICK, DONE = ROW_BASE + VAL_COUNT, ROW_BASE + VAL_COUNT + 1, ROW_BASE + VAL_COUNT + 2
MODEL_VOCAB = -(-(DONE + 1) // 64) * 64
CODE_BITS, CODE_AMP = 19, 128

TC_START, TC_END, IM_START, IM_END = 248_058, 248_059, 248_045, 248_046
USER, ASSISTANT = 846, 74_455
NL, LT, GT, EQ, LTS, FUNCTION, PARAMETER = 198, 27, 29, 28, 510, 1628, 15704
DIGIT_IDS = tuple(range(15, 25))
OP_NAMES = {
    "advance": (57_316,), "clear": (7_187,), "glider": (5_882, 1_713),
    "blinker": (1_956, 39_755), "block": (4_425,),
    "cell_on": (5_691, 4_322), "cell_off": (5_691, 13_258), "inspect": (79_221,),
}
OPS = tuple(OP_NAMES)
PATTERNS = {
    "glider": ((0, 1), (1, 2), (2, 0), (2, 1), (2, 2)),
    "blinker": ((0, 0), (0, 1), (0, 2)),
    "block": ((0, 0), (0, 1), (1, 0), (1, 1)),
}

def row_token(value: int) -> int:
    if not 0 <= int(value) < VAL_COUNT:
        raise ValueError("row must be unsigned 16-bit")
    return ROW_BASE + int(value)

def row_value(token: int) -> int:
    if not ROW_BASE <= int(token) < RESET:
        raise ValueError("not a row token")
    return int(token) - ROW_BASE

def is_row_token(token: int) -> bool:
    return ROW_BASE <= int(token) < RESET

def token_name(token: int) -> str:
    if is_row_token(token):
        return f"ROW(0x{row_value(token):04x})"
    return {RESET: "RESET", TICK: "TICK", DONE: "DONE", TC_START: "<tool_call>",
            TC_END: "</tool_call>", IM_END: "<|im_end|>"}.get(int(token), str(int(token)))

def tool_call_tokens(op: str, *args: int) -> tuple[int, ...]:
    if op not in OP_NAMES:
        raise ValueError(op)
    names = ("generations",) if op == "advance" else ("row", "column")
    if op in {"clear", "inspect"}:
        names = ()
    if len(args) != len(names):
        raise ValueError(f"{op} takes {len(names)} args")
    params = {"generations": (7_255, 778), "row": (63_319,), "column": (EQ, 6_033)}
    out = [TC_START, NL, LT, FUNCTION, EQ, *OP_NAMES[op], GT, NL]
    for name, value in zip(names, args):
        digits = [DIGIT_IDS[int(ch)] for ch in str(int(value))]
        out += [LT, PARAMETER, *params[name], GT, NL, *digits, NL, LTS, PARAMETER, GT, NL]
    out += [LTS, FUNCTION, GT, NL, TC_END]
    return tuple(out)

@dataclass(frozen=True)
class InternalTrace:
    tokens: tuple[int, ...]
    generated: tuple[bool, ...]
    worlds: tuple[tuple[int, ...], ...]


class InternalLifeCircuit(Circ):
    """Exact native-call parser, state editor, and Life transition circuit."""

    def __init__(self) -> None:
        super().__init__()
        one, f, full = self.one, self.token_feature, True
        self.is_row = f("tok.row", lambda t: int(ROW_BASE <= t < RESET), full=full)
        self.is_reset = f("tok.reset", lambda t: int(t == RESET), full=full)
        self.is_tick = f("tok.tick", lambda t: int(t == TICK), full=full)
        self.is_done = f("tok.done", lambda t: int(t == DONE), full=full)
        self.tc_start = f("tok.tc_start", lambda t: int(t == TC_START), full=full)
        self.tc_end = f("tok.tc_end", lambda t: int(t == TC_END), full=full)
        self.im_start = f("tok.im_start", lambda t: int(t == IM_START), full=full)
        self.im_end = f("tok.im_end", lambda t: int(t == IM_END), full=full)
        self.is_user = f("tok.user", lambda t: int(t == USER), full=full)
        self.is_assistant = f("tok.assistant", lambda t: int(t == ASSISTANT), full=full)
        self.is_eq = f("tok.eq", lambda t: int(t == EQ), full=full)
        self.is_digit = f("tok.digit", lambda t: int(t in DIGIT_IDS), full=full)
        self.digit_bits = [f(
            f"tok.digit.b{i}",
            lambda t, i=i: ((t - DIGIT_IDS[0]) >> i) & 1 if t in DIGIT_IDS else 0,
            full=full) for i in range(4)]
        self.value_bits = [f(
            f"tok.row.b{i}",
            lambda t, i=i: ((t - ROW_BASE) >> i) & 1 if ROW_BASE <= t < RESET else 0,
            full=full) for i in range(WIDTH)]
        interesting = {tok for seq in OP_NAMES.values() for tok in seq}
        self.op_token = {tok: f(f"tok.id{tok}", lambda t, tok=tok: int(t == tok), full=full)
                         for tok in sorted(interesting)}
        # Absolute-id output rails for every base/private vocabulary row.
        self.code = [f(f"tok.code.b{i}",
                       lambda t, i=i: CODE_AMP * (2 * ((int(t) >> i) & 1) - 1),
                       full=full) for i in range(CODE_BITS)]

        # Calls are accepted only in an assistant turn. Tool examples in the
        # system prompt and pasted calls in a user turn remain inert.
        (prev_imstart,) = self.slot_head(
            "role.lag", {0: one}, {0: one}, {2: Chan(self.im_start, (0, 1, 0, 0))}, one)
        assistant_start = self.and_([prev_imstart, self.is_assistant], "role.assistant.start")
        user_start = self.and_([prev_imstart, self.is_user], "role.user.start")
        role_control = self.im_start + assistant_start + user_start + self.im_end
        (assistant_phase,) = self.slot_head(
            "role.assistant", {0: role_control}, {0: one}, {2: assistant_start}, role_control)
        accepted_start = self.and_([self.tc_start, assistant_phase], "call.accept.start")
        accepted_end = self.and_([self.tc_end, assistant_phase], "call.accept.end")
        (self.in_call,) = self.slot_head(
            "call.active", {0: accepted_start + self.tc_end + self.im_end}, {0: one},
            {2: accepted_start}, accepted_start + self.tc_end + self.im_end)

        # Previous digit / equality / first half of two-token function names.
        lag_data = {2: Chan(self.is_digit, (0, 1, 0, 0)),
                    3: Chan(self.is_digit, (0, 0, 1, 0)),
                    4: Chan(self.is_eq, (0, 1, 0, 0))}
        lag_tokens = sorted({seq[0] for seq in OP_NAMES.values() if len(seq) == 2})
        for col, tok in enumerate(lag_tokens, 5):
            lag_data[col] = Chan(self.op_token[tok], (0, 1, 0, 0))
        lag = self.slot_head("call.lag", {0: one}, {0: one}, lag_data, one)
        self.prev_digit, self.prev2_digit, self.prev_eq = lag[:3]
        prev_name = dict(zip(lag_tokens, lag[3:]))

        op_write = {}
        for op, seq in OP_NAMES.items():
            if len(seq) == 1:
                op_write[op] = self.and_([self.in_call, self.op_token[seq[0]], self.prev_eq],
                                         f"call.op.{op}")
            else:
                op_write[op] = self.and_([self.in_call, self.op_token[seq[1]], prev_name[seq[0]]],
                                         f"call.op.{op}")

        self.run_start = self.and_([self.in_call, self.is_digit, one - self.prev_digit],
                                   "call.arg.start")
        self.second_digit = self.and_([self.in_call, self.is_digit, self.prev_digit,
                                       one - self.prev2_digit], "call.arg.second")
        self.extra_digit = self.and_([self.in_call, self.is_digit, self.prev_digit,
                                      self.prev2_digit], "call.arg.extra")
        arg_counter = counter(self, "call.arg.counter", accepted_start, self.run_start)
        self.arg_gt = thermo(self, arg_counter, accepted_start, 3, "call.arg.n")
        arg_at = [self.arg_gt[0] - self.arg_gt[1], self.arg_gt[1] - self.arg_gt[2]]

        # One register head with disjoint columns holds the operation and four
        # decimal digit slots. Row 127 is a permanent normalization ballast.
        writes, data, op_cols, col = {}, {}, {}, 2
        for slot, op in enumerate(OPS):
            writes[slot], op_cols[op], data[col] = op_write[op], col, op_write[op]
            col += 1
        digit_cols: list[list[tuple[list[int], int]]] = [[], []]
        for arg in range(2):
            for pos, event in enumerate((self.run_start, self.second_digit)):
                w = self.and_([arg_at[arg], event], f"call.a{arg}.d{pos}.write")
                writes[16 + 2 * arg + pos] = w
                bits = []
                for b, bit in enumerate(self.digit_bits):
                    data[col + b] = self.and_([w, bit], f"call.a{arg}.d{pos}.b{b}.write")
                    bits.append(col + b)
                data[col + 4] = w
                digit_cols[arg].append((bits, col + 4))
                col += 5
        overflow_write = self.and_([self.extra_digit, one - self.arg_gt[2]],
                                   "call.overflow.write")
        writes[24], overflow_col, data[col] = overflow_write, col, overflow_write
        writes[127] = accepted_start
        reg = self.slot_head("call.registers", writes, {slot: one for slot in writes}, data,
                             accepted_start, a=238 * accepted_start)
        by_col = dict(zip(sorted(data), reg))
        self.op = {op: by_col[c] for op, c in op_cols.items()}
        self.overflow = by_col[overflow_col]
        # A malformed envelope can contain more than one otherwise valid
        # function name.  The operation register deliberately remembers every
        # name it sees.  Each operation-specific fire already requires its own
        # bit; this additional linear gate is 1 only when every *other* bit is
        # zero.  Keeping it linear avoids adding a layer to the compiled graph.
        op_count = sum(self.op.values(), Expr())
        self.no_other_op = {
            op: one - (op_count - self.op[op]) for op in OPS
        }

        # Decode canonical decimal 0..15 only; malformed or extra digits cannot fire.
        self.arg_oh = []
        for arg in range(2):
            digits = [([by_col[c] for c in bits], by_col[p]) for bits, p in digit_cols[arg]]
            (d0, p0), (d1, p1) = digits
            values = []
            for value in range(16):
                if value < 10:
                    ds, lits = [(d0, value)], [p0, one - p1]
                else:
                    ds, lits = [(d0, 1), (d1, value - 10)], [p0, p1]
                lits.append(one - self.overflow)
                for bits, digit in ds:
                    lits += [bit if digit >> b & 1 else one - bit
                             for b, bit in enumerate(bits)]
                values.append(self.and_(lits, f"call.a{arg}=={value}"))
            self.arg_oh.append(values)
        arg_valid = [sum(values, Expr()) for values in self.arg_oh]
        no_args, one_arg = one - self.arg_gt[0], self.arg_gt[0] - self.arg_gt[1]
        two_args = self.arg_gt[1] - self.arg_gt[2]

        self.call = {
            "advance": self.and_([accepted_end, self.op["advance"],
                                  self.no_other_op["advance"], one_arg, arg_valid[0],
                                  one - self.arg_oh[0][0]], "call.fire.advance"),
            "clear": self.and_([accepted_end, self.op["clear"], self.no_other_op["clear"],
                                no_args],
                               "call.fire.clear"),
            "inspect": self.and_([accepted_end, self.op["inspect"],
                                  self.no_other_op["inspect"], no_args],
                                 "call.fire.inspect"),
        }
        for op, (rmax, cmax) in {"glider": (13, 13), "blinker": (15, 13),
                                 "block": (14, 14), "cell_on": (15, 15),
                                 "cell_off": (15, 15)}.items():
            row_ok = sum(self.arg_oh[0][:rmax + 1], Expr())
            col_ok = sum(self.arg_oh[1][:cmax + 1], Expr())
            self.call[op] = self.and_([accepted_end, self.op[op], self.no_other_op[op], two_args,
                                      arg_valid[0], arg_valid[1], row_ok, col_ok],
                                     f"call.fire.{op}")
        self.call_trigger = sum(self.call.values(), Expr())
        # Advance first emits a private TICK; its input triggers the first row
        # pass. Consequently an advance(n) trace contains exactly n TICKs,
        # which is useful for a generic renderer without parsing the call.
        self.advance_start = self.call["advance"]
        self.edit_trigger = self.call_trigger - self.advance_start
        self.exec_trigger = self.edit_trigger + self.is_tick

        tick_counter = counter(self, "advance.ticks", self.advance_start, self.is_tick)
        tick_gt = thermo(self, tick_counter, self.advance_start, 15, "advance.tick.n")
        repeat_terms = []
        for count in range(2, 16):
            repeat_terms += self.t_and([self.op["advance"], self.arg_oh[0][count],
                                        one - tick_gt[count - 1]])
        self.repeat = self.node(repeat_terms, "advance.repeat")

        # Private row feedback protocol and persistent grid bank.
        reset_counter = self.is_reset + self.exec_trigger + self.is_done
        self.row_counter = counter(self, "row.counter", reset_counter, self.is_row)
        self.row_gt = thermo(self, self.row_counter, reset_counter, HEIGHT, "row.k")
        control = self.is_reset + self.is_done + self.exec_trigger
        (self.active,) = self.slot_head(
            "generation.active", {0: control}, {0: one}, {2: self.exec_trigger}, control)
        self.write_row = [
            self.row_gt[i] - (self.row_gt[i + 1] if i + 1 < HEIGHT else 0 * one)
            for i in range(HEIGHT)
        ]
        # A persisted world restores only the authoritative grid heads, not
        # this controller's row counter.  An empty counter makes every
        # thermometer bit high, so ``write_row[15]`` is spuriously 1 until the
        # counter is primed.  Require an actual private row token in every
        # grid-write key.  The linear form is 1 only when both conditions are
        # true and <= 0 otherwise, which slot_head maps to an exact absent key.
        self.grid_write_key = [
            self.is_row + selected - one for selected in self.write_row
        ]
        self.output_row = [one - self.row_gt[0]] + [
            self.row_gt[i - 1] - self.row_gt[i] for i in range(1, HEIGHT)
        ]

        self.current = []
        for y in range(HEIGHT):
            self.current.append(self.slot_head(
                f"grid.row{y}", {0: self.grid_write_key[y]}, {0: self.exec_trigger},
                {2 + x: bit for x, bit in enumerate(self.value_bits)}, self.write_row[y],
                a=238 * self.is_reset))

        # Exact Conway successor.
        life = []
        for y in range(HEIGHT):
            out = []
            for x in range(WIDTH):
                neighbours = sum((self.current[yy][xx]
                    for yy in range(max(0, y - 1), min(HEIGHT, y + 2))
                    for xx in range(max(0, x - 1), min(WIDTH, x + 2))
                    if (yy, xx) != (y, x)), 0 * one)
                z = 2 * neighbours + self.current[y][x]
                out.append(self.node(self.t_in(z, 5, 8, 0, 17), f"life[{y},{x}]"))
            life.append(out)

        # Pattern masks selected by Qwen's concrete tool call.
        masks = {}
        for op, offsets in PATTERNS.items():
            grid = []
            for y in range(HEIGHT):
                row = []
                for x in range(WIDTH):
                    terms = []
                    for dy, dx in offsets:
                        y0, x0 = y - dy, x - dx
                        if 0 <= y0 < HEIGHT and 0 <= x0 < WIDTH:
                            terms += self.t_and([self.call[op], self.arg_oh[0][y0],
                                                 self.arg_oh[1][x0]])
                    row.append(self.node(terms, f"{op}.mask[{y},{x}]"))
                grid.append(row)
            masks[op] = grid
        cell_selected = [[self.and_([
            self.call["cell_on"] + self.call["cell_off"], self.arg_oh[0][y],
            self.arg_oh[1][x]], f"cell[{y},{x}]") for x in range(WIDTH)]
            for y in range(HEIGHT)]

        # Select one deterministic replacement bit per operation. Clear has no
        # live-bit term and therefore naturally emits sixteen zero rows.
        self.target_grid = []
        for y in range(HEIGHT):
            row = []
            for x in range(WIDTH):
                terms = self.t_and([self.is_tick, life[y][x]])
                terms += self.t_and([self.call["inspect"], self.current[y][x]])
                for op in PATTERNS:
                    placed = self.node(self.t_ge(
                        self.current[y][x] + masks[op][y][x], 1, 0, 2),
                        f"{op}.out[{y},{x}]")
                    terms += self.t_and([self.call[op], placed])
                on = self.node(self.t_ge(
                    self.current[y][x] + cell_selected[y][x], 1, 0, 2),
                    f"cell.on.out[{y},{x}]")
                off = self.and_([self.current[y][x], one - cell_selected[y][x]],
                                f"cell.off.out[{y},{x}]")
                terms += self.t_and([self.call["cell_on"], on])
                terms += self.t_and([self.call["cell_off"], off])
                row.append(self.node(terms, f"target[{y},{x}]"))
            self.target_grid.append(row)

        # Capture all rows on a TICK/edit trigger, then expose the selected row
        # on each feedback step while the earlier bank commits it.
        self.selected = None
        for y in range(HEIGHT):
            v = {0: A_REF * self.exec_trigger, 1: A_REF * self.exec_trigger}
            v.update({2 + x: A_BIT * self.target_grid[y][x] for x in range(WIDTH)})
            result = self.head(
                f"next.row{y}", {0: self._select(self.active)},
                {0: self._select(self.exec_trigger)}, v, 238 * self.is_reset,
                {2 + x: K * self.output_row[y] for x in range(WIDTH)},
                [(2 + x, 2.0**7) for x in range(WIDTH)], into=self.selected)
            if self.selected is None:
                self.selected = result
        assert self.selected is not None

        self.done_now = self.and_([self.active, self.row_gt[-1]], "emit.done")
        self.repeat_done = self.and_([self.done_now, self.repeat], "emit.repeat")
        self.final_done = self.done_now - self.repeat_done
        self.row_phase = self.active - self.done_now
        self.owned = self.advance_start + self.active + self.is_done

        # Absolute next-token bits. At idle every bit is exactly zero, so the
        # added output rails contribute no logits and Qwen retains control.
        self.output_bits = []
        for i in range(CODE_BITS):
            bit = self.selected[i] if i < WIDTH else 0 * one
            if ROW_BASE >> i & 1:
                bit = bit + self.row_phase
            if TICK >> i & 1:
                bit = bit + self.advance_start + self.repeat_done
            if DONE >> i & 1:
                bit = bit + self.final_done
            if IM_END >> i & 1:
                bit = bit + self.is_done
            self.output_bits.append(bit)

        scale = 2**12
        for i in range(CODE_BITS):
            self.ffn(
                [(2 * scale, one, self.output_bits[i]), (-scale, one, self.owned),
                 (-1, one, self.code[i]), (1, one, -self.code[i])],
                f"exit.code{i}", into=self.code[i])

    def _select(self, value):
        if isinstance(value, Chan):
            return Chan(value.expr * (2 * K) - self.one * K, value.taps)
        return value * (2 * K) - self.one * K

    def predicted_tokens(self, tokens: Sequence[int], *, check: bool = True) -> np.ndarray:
        residual = ir.evaluate(self, list(tokens), check=check)
        bits = np.stack([ir.ev(bit, residual) for bit in self.output_bits], axis=1)
        if not np.isin(bits, [0.0, 1.0]).all():
            raise AssertionError(f"output bits are not binary: {np.unique(bits)[:8]}")
        return bits.astype(np.int64) @ (1 << np.arange(CODE_BITS, dtype=np.int64))

    def validate_trace(self, trace: InternalTrace, *, check: bool = True) -> tuple[int, list]:
        predicted = self.predicted_tokens(trace.tokens, check=check)
        want = np.asarray(trace.tokens[1:] + (-1,), dtype=np.int64)
        generated = np.asarray(trace.generated[1:] + (False,), dtype=bool)
        positions = np.flatnonzero(generated)
        bad = positions[predicted[positions] != want[positions]]
        detail = [(int(p), token_name(trace.tokens[p]), token_name(int(want[p])),
                   token_name(int(predicted[p]))) for p in bad[:20]]
        return int(positions.size), detail


def _apply(rows: tuple[int, ...], op: str, args: tuple[int, ...]) -> tuple[int, ...]:
    if op == "clear":
        return (0,) * HEIGHT
    if op == "inspect":
        return rows
    current = list(rows)
    if op in PATTERNS:
        y0, x0 = args
        for dy, dx in PATTERNS[op]:
            current[y0 + dy] |= 1 << (x0 + dx)
    elif op == "cell_on":
        current[args[0]] |= 1 << args[1]
    elif op == "cell_off":
        current[args[0]] &= ~(1 << args[1])
    else:
        raise ValueError(op)
    return tuple(current)


def build_internal_trace(initial_rows: Iterable[int],
                         calls: Sequence[tuple[str, tuple[int, ...]]]) -> InternalTrace:
    """Script exact Qwen calls and the private tokens the circuit must own."""
    from life.reference import step
    rows = tuple(int(row) for row in initial_rows)
    if len(rows) != HEIGHT or any(not 0 <= row < VAL_COUNT for row in rows):
        raise ValueError("expected sixteen unsigned 16-bit rows")
    tokens = [RESET, *(row_token(row) for row in rows), DONE]
    generated = [False] * len(tokens)
    worlds = [rows]
    for op, args in calls:
        prefix = [IM_START, ASSISTANT]
        call = tool_call_tokens(op, *args)
        tokens.extend(prefix + list(call))
        generated.extend([False] * (len(prefix) + len(call)))
        count = args[0] if op == "advance" else 1
        for n in range(count):
            if op == "advance":
                tokens.append(TICK)
                generated.append(True)
            rows = step(rows) if op == "advance" else (_apply(rows, op, args) if n == 0 else rows)
            tokens.extend(row_token(row) for row in rows)
            generated.extend([True] * HEIGHT)
        worlds.append(rows)
        tokens.extend([DONE, IM_END])
        generated.extend([True, True])
    return InternalTrace(tuple(tokens), tuple(generated), tuple(worlds))


__all__ = [
    "ASSISTANT", "BASE_VOCAB", "CODE_AMP", "CODE_BITS", "DONE", "HEIGHT", "IM_END",
    "IM_START", "InternalLifeCircuit", "InternalTrace", "MODEL_VOCAB", "OPS", "RESET",
    "ROW_BASE", "TC_END", "TC_START", "TICK", "USER", "VAL_COUNT", "WIDTH",
    "build_internal_trace", "is_row_token", "row_token", "row_value", "token_name",
    "tool_call_tokens",
]
