"""Small fresh-context probe for fixes found by the broad semantic suite.

Unlike the 31-case discovery sweep, this script is intentionally short.  It
checks valid overlapping edits, rollback of a rejected native call, a positive
command immediately after negation, and consecutive repeated commands.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch

from life.internal_runtime import InternalRuntime, atomic_json
from life.reference import INITIAL_ROWS
from validation.internal_semantic_robustness import (
    Case,
    advance,
    cell,
    clear,
    digest,
    identity,
    place,
    population,
    run_case,
    summarize,
)


def targeted_cases() -> list[Case]:
    return [
        Case(
            "negation_then_positive",
            "negated_clear",
            "Do not clear the world. Just acknowledge this sentence.",
            None,
            identity,
        ),
        Case(
            "negation_then_positive",
            "positive_clear_immediately_after",
            "Clear the world now.",
            "clear",
            clear,
            commits=1,
        ),
        Case(
            "overlapping_rows",
            "blinker_bottom_right",
            "Place a blinker at row 15 column 13.",
            "blinker",
            place("blinker", 15, 13),
            commits=1,
        ),
        Case(
            "overlapping_rows",
            "block_overlapping_blinker",
            "Place a block at row 14 column 14.",
            "block",
            place("block", 14, 14),
            commits=1,
        ),
        Case(
            "overlapping_rows",
            "cell_on_already_live_with_live_neighbor",
            "Make row 15 column 15 alive.",
            "cell_on",
            cell(15, 15, True),
            commits=1,
        ),
        Case(
            "transaction_guard",
            "rejected_out_of_range_glider",
            "Place a glider with origin row 15 column 15.",
            "glider",
            identity,
            commits=0,
        ),
        Case(
            "repetition",
            "repeat_setup_clear",
            "Clear the world.",
            "clear",
            clear,
            commits=1,
        ),
        Case(
            "repetition",
            "blinker_same_prompt_first",
            "Place a blinker at row 7 column 7.",
            "blinker",
            place("blinker", 7, 7),
            commits=1,
        ),
        Case(
            "repetition",
            "blinker_same_prompt_second",
            "Place a blinker at row 7 column 7.",
            "blinker",
            place("blinker", 7, 7),
            commits=1,
        ),
        Case(
            "repetition",
            "advance_same_prompt_first",
            "Advance one generation.",
            "advance",
            advance(1),
            ticks=1,
            commits=1,
        ),
        Case(
            "repetition",
            "advance_same_prompt_second",
            "Advance one generation.",
            "advance",
            advance(1),
            ticks=1,
            commits=1,
        ),
        Case(
            "repetition",
            "block_same_prompt_first",
            "Place a block at row 7 column 7.",
            "block",
            place("block", 7, 7),
            commits=1,
        ),
        Case(
            "repetition",
            "block_same_prompt_second",
            "Place a block at row 7 column 7.",
            "block",
            place("block", 7, 7),
            commits=1,
        ),
    ]


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

    selected = targeted_cases()
    report = {
        "status": "running",
        "checkpoint": str(args.model.resolve()),
        "runtime_dir": str(args.runtime_dir.resolve()),
        "purpose": "fresh-context regressions discovered by the broad semantic suite",
        "no_training": True,
        "host_semantic_parser": False,
        "planned_total": len(selected),
        "environment": {
            "device": args.device,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "causal_conv1d_installed": importlib.util.find_spec("causal_conv1d") is not None,
            "flash_linear_attention_installed": importlib.util.find_spec("fla") is not None,
        },
        "context_resets": [],
        "cases": [],
    }
    atomic_json(args.out, report)
    for index, case in enumerate(selected, 1):
        if case.name == "repeat_setup_clear":
            # Repetition gets an explicitly fresh semantic conversation while
            # retaining the same authoritative recurrent world.  This
            # separates repetition from the preceding negation/rejection probes.
            runtime.conversation.clear()
            report["context_resets"].append({
                "before_case": case.name,
                "reason": "isolate repeated-command routing from prior safety prompts",
            })
        result = run_case(runtime, case)
        report["cases"].append(result)
        summarize(report)
        atomic_json(args.out, report)
        print(
            f"[{index:02d}/{len(selected):02d}] "
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
