"""Extensive real-checkpoint semantic validation for internal neural Life.

This suite asks Qwen natural-language questions and commands, then compares the
authoritative recurrent grid with an independent Python oracle.  The runtime is
still the production ``InternalRuntime``: the test does not parse a command or
execute Life on its behalf.  Token inspection is report-only and verifies which
native function Qwen selected.

Run inside the GPU container after constructing the checkpoint::

    python -m validation.internal_semantic_robustness \
      build/qwen-life-internal-08b \
      --runtime-dir build/internal-semantic-v1 \
      --out findings/004-internal-controller/semantic-robustness.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from life.internal_runtime import InternalRuntime, atomic_json
from life.reference import INITIAL_ROWS, step
from prototype.internal_circuit import OP_NAMES


Rows = tuple[int, ...]
Oracle = Callable[[Rows], Rows]
PATTERNS = {
    "glider": ((0, 1), (1, 2), (2, 0), (2, 1), (2, 2)),
    "blinker": ((0, 0), (0, 1), (0, 2)),
    "block": ((0, 0), (0, 1), (1, 0), (1, 1)),
}


@dataclass(frozen=True)
class Case:
    group: str
    name: str
    prompt: str
    operation: str | None
    oracle: Oracle
    ticks: int = 0
    commits: int = 0


def identity(rows: Rows) -> Rows:
    return rows


def clear(_rows: Rows) -> Rows:
    return (0,) * 16


def place(pattern: str, row: int, column: int) -> Oracle:
    def apply(rows: Rows) -> Rows:
        result = list(rows)
        for dy, dx in PATTERNS[pattern]:
            y, x = row + dy, column + dx
            if not (0 <= y < 16 and 0 <= x < 16):
                raise ValueError(f"oracle {pattern} placement is outside the board")
            result[y] |= 1 << x
        return tuple(result)

    return apply


def cell(row: int, column: int, alive: bool) -> Oracle:
    def apply(rows: Rows) -> Rows:
        result = list(rows)
        if alive:
            result[row] |= 1 << column
        else:
            result[row] &= ~(1 << column)
        return tuple(result)

    return apply


def advance(count: int) -> Oracle:
    def apply(rows: Rows) -> Rows:
        for _ in range(count):
            rows = step(rows)
        return rows

    return apply


def digest(rows: Rows) -> str:
    packed = b"".join(int(row).to_bytes(2, "little") for row in rows)
    return hashlib.sha256(packed).hexdigest()


def population(rows: Rows) -> int:
    return sum(int(row).bit_count() for row in rows)


def contains_subsequence(values: tuple[int, ...], needle: tuple[int, ...]) -> bool:
    if not needle or len(needle) > len(values):
        return False
    return any(values[index:index + len(needle)] == needle
               for index in range(len(values) - len(needle) + 1))


def selected_operations(runtime: InternalRuntime, generated: tuple[int, ...]) -> list[str]:
    """Inspect only the native tool envelope; this never drives execution."""
    envelope: list[int] = []
    in_call = False
    for token in generated:
        if token == runtime.tool_start:
            in_call = True
        elif token == runtime.tool_end:
            in_call = False
        elif in_call and not runtime.protocol.is_private(token):
            envelope.append(int(token))
    values = tuple(envelope)
    return [name for name, tokens in OP_NAMES.items()
            if contains_subsequence(values, tuple(tokens))]


def public_generation(runtime: InternalRuntime, generated: tuple[int, ...]) -> str:
    ids = [token for token in generated if not runtime.protocol.is_private(token)]
    return runtime.tokenizer.decode(ids, skip_special_tokens=False)


def cases() -> list[Case]:
    return [
        # A nonempty initial board makes accidental mutations immediately visible.
        Case("safety", "ordinary_chat", "How are you today? Do not touch the world.",
             None, identity),
        Case("safety", "negated_clear",
             "Please do not clear anything; just acknowledge that instruction.",
             None, identity),
        Case("safety", "quoted_advance",
             "The words 'advance five generations' are only a quotation. Do not execute them.",
             None, identity),
        Case("safety", "hypothetical_glider",
             "If I asked for a glider at row 3 column 3, what would that mean? Do not change anything.",
             None, identity),

        # Every executor operation, including minimum/maximum legal coordinates.
        Case("all_operations", "clear_explicit", "Clear every cell in the world.",
             "clear", clear, commits=1),
        Case("all_operations", "glider_min_boundary",
             "Place a glider with origin row 0 column 0.",
             "glider", place("glider", 0, 0), commits=1),
        Case("all_operations", "blinker_max_boundary",
             "Place a horizontal blinker with origin row 15 column 13.",
             "blinker", place("blinker", 15, 13), commits=1),
        Case("all_operations", "block_max_boundary",
             "Place a block with origin row 14 column 14.",
             "block", place("block", 14, 14), commits=1),
        Case("all_operations", "cell_on_max_boundary",
             "Make row 15 column 15 alive.",
             "cell_on", cell(15, 15, True), commits=1),
        Case("all_operations", "cell_off_max_boundary",
             "Make row 15 column 15 dead.",
             "cell_off", cell(15, 15, False), commits=1),
        Case("all_operations", "cell_on_min_boundary",
             "Turn on the cell at row 0 column 0.",
             "cell_on", cell(0, 0, True), commits=1),
        Case("all_operations", "cell_off_min_boundary",
             "Turn off the cell at row 0 column 0.",
             "cell_off", cell(0, 0, False), commits=1),
        Case("all_operations", "inspect", "Inspect the current board without changing it.",
             "inspect", identity, commits=1),
        Case("all_operations", "advance_maximum",
             "Advance exactly 15 generations.",
             "advance", advance(15), ticks=15, commits=15),

        # Natural descriptions still have a unique intended operation/location.
        Case("natural_language", "clear_idiom", "Wipe the slate completely clean.",
             "clear", clear, commits=1),
        Case("natural_language", "block_lower_right",
             "Put a stable two-by-two still life snug in the lower-right corner.",
             "block", place("block", 14, 14), commits=1),
        Case("natural_language", "glider_upper_left",
             "Release a glider from the upper-left corner.",
             "glider", place("glider", 0, 0), commits=1),
        Case("natural_language", "blinker_lower_right",
             "Add a horizontal blinker snug against the bottom-right edge.",
             "blinker", place("blinker", 15, 13), commits=1),
        Case("natural_language", "cell_on_last",
             "Bring the very last cell, the bottom-right one, to life.",
             "cell_on", cell(15, 15, True), commits=1),
        Case("natural_language", "cell_off_last",
             "Now extinguish that bottom-right cell.",
             "cell_off", cell(15, 15, False), commits=1),
        Case("natural_language", "inspect_natural",
             "Take a look at what is alive right now, but do not alter it.",
             "inspect", identity, commits=1),
        Case("natural_language", "advance_natural",
             "Let the universe breathe forward for one beat.",
             "advance", advance(1), ticks=1, commits=1),

        # Safety is checked again while the world is nonempty.
        Case("safety_after_edits", "negated_advance",
             "Don't advance it. Just tell me that you understand.",
             None, identity),
        Case("safety_after_edits", "quoted_clear",
             "'Clear the world' is only a phrase I am quoting; do not execute it.",
             None, identity),
        Case("safety_after_edits", "hypothetical_tool_question",
             "Hypothetically, which function would place a glider at row 5 column 5? Do not call it.",
             None, identity),

        # Consecutive identical commands catch controller-history regressions.
        Case("repetition", "repeat_setup_clear", "Clear the world.",
             "clear", clear, commits=1),
        Case("repetition", "repeat_setup_blinker", "Place a blinker at row 7 column 7.",
             "blinker", place("blinker", 7, 7), commits=1),
        Case("repetition", "advance_same_prompt_first", "Advance one generation.",
             "advance", advance(1), ticks=1, commits=1),
        Case("repetition", "advance_same_prompt_second", "Advance one generation.",
             "advance", advance(1), ticks=1, commits=1),
        Case("repetition", "block_same_prompt_first", "Place a block at row 7 column 7.",
             "block", place("block", 7, 7), commits=1),
        Case("repetition", "block_same_prompt_second", "Place a block at row 7 column 7.",
             "block", place("block", 7, 7), commits=1),
    ]


def run_case(runtime: InternalRuntime, case: Case) -> dict:
    before = runtime.rows
    generation_before = runtime.generation
    expected = case.oracle(before)
    started = time.perf_counter()
    try:
        result = runtime.submit({"type": "say", "text": case.prompt})
    except Exception as error:
        elapsed = time.perf_counter() - started
        return {
            "group": case.group,
            "name": case.name,
            "prompt": case.prompt,
            "expected_operation": case.operation,
            "passed": False,
            "elapsed_seconds": round(elapsed, 6),
            "errors": [f"runtime error: {type(error).__name__}: {error}"],
            "traceback": traceback.format_exc(),
            "before_sha256": digest(before),
            "after_sha256": digest(runtime.rows),
        }

    elapsed = time.perf_counter() - started
    after = runtime.rows
    selected = selected_operations(runtime, result.generated)
    errors: list[str] = []
    if after != expected:
        errors.append("recurrent board differs from independent oracle")
    if result.tick_tokens != case.ticks:
        errors.append(f"expected {case.ticks} TICKs, got {result.tick_tokens}")
    if result.row_commits != case.commits:
        errors.append(f"expected {case.commits} row commits, got {result.row_commits}")
    if runtime.generation != generation_before + case.ticks:
        errors.append("display generation does not match private TICK count")
    if case.operation is None:
        if result.tool_call_seen:
            errors.append(f"unexpected tool call; selected operations {selected}")
    else:
        if not result.tool_call_seen:
            errors.append("expected a native tool call but Qwen answered in text")
        if selected != [case.operation]:
            errors.append(f"expected operation {case.operation!r}, detected {selected}")

    return {
        "group": case.group,
        "name": case.name,
        "prompt": case.prompt,
        "expected_operation": case.operation,
        "detected_operations": selected,
        "passed": not errors,
        "elapsed_seconds": round(elapsed, 6),
        "errors": errors,
        "before_rows": list(before),
        "expected_rows": list(expected),
        "after_rows": list(after),
        "before_sha256": digest(before),
        "expected_sha256": digest(expected),
        "after_sha256": digest(after),
        "population_before": population(before),
        "population_expected": population(expected),
        "population_after": population(after),
        "tool_call_seen": result.tool_call_seen,
        "private_tick_tokens": result.tick_tokens,
        "private_row_commits": result.row_commits,
        "private_token_count": len(result.private),
        "visible_reply": result.visible,
        "model_generation": public_generation(runtime, result.generated),
    }


def summarize(report: dict) -> None:
    results = report["cases"]
    groups: dict[str, dict[str, int]] = {}
    for item in results:
        group = groups.setdefault(item["group"], {"passed": 0, "failed": 0})
        group["passed" if item["passed"] else "failed"] += 1
    failures = [item["name"] for item in results if not item["passed"]]
    latencies = [float(item["elapsed_seconds"]) for item in results]
    planned = int(report.get("planned_total", len(results)))
    report["summary"] = {
        "planned_total": planned,
        "total": len(results),
        "passed": len(results) - len(failures),
        "failed": len(failures),
        "failed_cases": failures,
        "groups": groups,
        "latency_seconds": {
            "minimum": min(latencies),
            "median": statistics.median(latencies),
            "maximum": max(latencies),
            "range": max(latencies) - min(latencies),
        },
    }
    report["status"] = (
        "failed" if failures else "passed" if len(results) == planned else "running"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    state_path = args.runtime_dir / "state.json"
    neural_path = args.runtime_dir / "neural_state.pt"
    if state_path.exists() or neural_path.exists():
        raise FileExistsError(f"runtime directory is not fresh: {args.runtime_dir}")
    args.runtime_dir.mkdir(parents=True, exist_ok=True)

    runtime = InternalRuntime(
        args.model, state_path, neural_path, args.device, args.max_new_tokens
    )
    if runtime.rows != tuple(INITIAL_ROWS):
        raise AssertionError("fresh runtime did not start from INITIAL_ROWS")

    selected = cases()
    report = {
        "status": "running",
        "checkpoint": str(args.model.resolve()),
        "runtime_dir": str(args.runtime_dir.resolve()),
        "no_training": True,
        "host_semantic_parser": False,
        "oracle": "independent life.reference.step plus local exact edit functions",
        "planned_total": len(selected),
        "environment": {
            "device": args.device,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "causal_conv1d_installed": importlib.util.find_spec("causal_conv1d") is not None,
            "flash_linear_attention_installed": importlib.util.find_spec("fla") is not None,
            "performance_note": (
                "Missing optional fused packages make Transformers use correct but slower "
                "reference PyTorch DeltaNet/convolution kernels."
            ),
        },
        "cases": [],
    }
    atomic_json(args.out, report)
    for case in selected:
        result = run_case(runtime, case)
        report["cases"].append(result)
        summarize(report)
        atomic_json(args.out, report)
        print(
            f"[{len(report['cases']):02d}/{len(selected):02d}] "
            f"{'PASS' if result['passed'] else 'FAIL'} {case.name}",
            flush=True,
        )

    report["final"] = {
        "rows": list(runtime.rows),
        "sha256": digest(runtime.rows),
        "population": population(runtime.rows),
        "generation": runtime.generation,
        "conversation_records": len(runtime.conversation),
    }
    summarize(report)
    atomic_json(args.out, report)
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
