"""Real-checkpoint validation for Qwen's directly programmed Life controller.

The language cases exercise Qwen's native semantic tool selection.  Expected
boards are computed independently and compared with recurrent neural state.
The scripted case bypasses language choice and feeds a native tool call into
the same checkpoint, proving the model-level private-token protocol exactly.

Example (after building the internal checkpoint)::

    python -m validation.internal_e2e build/qwen-life-internal-08b \
      --runtime-dir build/internal-e2e-run \
      --out findings/004-internal-controller/e2e.json

The runtime directory must be new unless ``--resume`` is explicitly supplied;
the script never deletes an existing world.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import traceback
from pathlib import Path
from typing import Callable

import torch

from life.internal_runtime import (
    InternalRuntime,
    Protocol,
    read_rows,
    restore_grid,
    snapshot_grid,
)
from life.reference import INITIAL_ROWS, step
from prototype.internal_circuit import (
    ASSISTANT,
    DONE,
    IM_END,
    IM_START,
    TICK,
    row_token,
    tool_call_tokens,
)


PATTERNS = {
    "glider": ((0, 1), (1, 2), (2, 0), (2, 1), (2, 2)),
    "block": ((0, 0), (0, 1), (1, 0), (1, 1)),
}


def digest(rows: tuple[int, ...]) -> str:
    packed = b"".join(int(row).to_bytes(2, "little") for row in rows)
    return hashlib.sha256(packed).hexdigest()


def population(rows: tuple[int, ...]) -> int:
    return sum(int(row).bit_count() for row in rows)


def place(
    rows: tuple[int, ...], pattern: str, row: int, column: int
) -> tuple[int, ...]:
    """Independent input oracle used only by this validation runner."""
    result = list(rows)
    for dy, dx in PATTERNS[pattern]:
        y, x = row + dy, column + dx
        if not (0 <= y < 16 and 0 <= x < 16):
            raise ValueError("validation pattern is outside the board")
        result[y] |= 1 << x
    return tuple(result)


def set_cell(rows: tuple[int, ...], row: int, column: int, alive: int) -> tuple[int, ...]:
    result = list(rows)
    result[row] = (
        result[row] | 1 << column if alive else result[row] & ~(1 << column)
    )
    return tuple(result)


def public_generation_text(runtime: InternalRuntime, token_ids: tuple[int, ...]) -> str:
    public = [token for token in token_ids if not runtime.protocol.is_private(token)]
    return runtime.tokenizer.decode(public, skip_special_tokens=False)


def natural_case(
    runtime: InternalRuntime,
    name: str,
    prompt: str,
    expected: Callable[[tuple[int, ...]], tuple[int, ...]],
    *,
    expected_ticks: int,
    expected_commits: int | None,
) -> dict:
    before = runtime.rows
    before_generation = runtime.generation
    result = runtime.submit({"type": "say", "text": prompt})
    after = runtime.rows
    want = expected(before)
    if after != want:
        raise AssertionError(f"{name}: recurrent neural state disagreed with oracle")
    if result.tick_tokens != expected_ticks:
        raise AssertionError(
            f"{name}: expected {expected_ticks} private TICKs, got {result.tick_tokens}"
        )
    if expected_commits is not None and result.row_commits != expected_commits:
        raise AssertionError(
            f"{name}: expected {expected_commits} row commits, got {result.row_commits}"
        )
    if runtime.generation != before_generation + expected_ticks:
        raise AssertionError(f"{name}: display generation did not follow private TICKs")
    return {
        "name": name,
        "prompt": prompt,
        "passed": True,
        "before_sha256": digest(before),
        "after_sha256": digest(after),
        "population_before": population(before),
        "population_after": population(after),
        "tool_call_seen": result.tool_call_seen,
        "private_tick_tokens": result.tick_tokens,
        "private_row_commits": result.row_commits,
        "private_token_count": len(result.private),
        "visible_reply": result.visible,
        "model_generation": public_generation_text(runtime, result.generated),
    }


@torch.no_grad()
def scripted_advance_check(runtime: InternalRuntime, count: int = 2) -> dict:
    """Feed a native call and compare every circuit-owned model prediction."""
    before = runtime.rows
    saved = snapshot_grid(runtime.cache, runtime.protocol)
    cache = restore_grid(runtime.model, saved, runtime.protocol)

    prefix = (IM_START, ASSISTANT, *tool_call_tokens("advance", count))
    output = None
    for token in prefix:
        ids = torch.tensor([[token]], device=runtime.model.device)
        output = runtime.model(input_ids=ids, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
    if output is None:
        raise AssertionError("scripted call prefix was empty")

    generated: list[int] = []
    for _ in range(2 + count * 18):
        token = int(output.logits[0, -1].float().argmax())
        generated.append(token)
        if token == IM_END:
            break
        ids = torch.tensor([[token]], device=runtime.model.device)
        output = runtime.model(input_ids=ids, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
    else:
        raise AssertionError("scripted internal protocol did not terminate")

    worlds, current = [], before
    expected_tokens: list[int] = []
    for _ in range(count):
        current = step(current)
        worlds.append(current)
        expected_tokens.extend([TICK, *(row_token(row) for row in current)])
    expected_tokens.extend([DONE, IM_END])
    if generated != expected_tokens:
        mismatch = next(
            (index for index, pair in enumerate(zip(generated, expected_tokens))
             if pair[0] != pair[1]),
            min(len(generated), len(expected_tokens)),
        )
        raise AssertionError(
            f"scripted private trace differs at token {mismatch}: "
            f"got {generated[mismatch:mismatch + 3]}, "
            f"expected {expected_tokens[mismatch:mismatch + 3]}"
        )
    after = read_rows(cache, runtime.protocol)
    if after != current:
        raise AssertionError("scripted call's recurrent state disagreed with oracle")
    if runtime.rows != before:
        raise AssertionError("isolated scripted check mutated the live runtime cache")
    return {
        "passed": True,
        "operation": f"advance({count})",
        "input_was_native_tool_call": True,
        "host_executed_operation": False,
        "private_predictions_exact": len(expected_tokens),
        "private_tick_tokens": generated.count(TICK),
        "private_row_tokens": sum(runtime.protocol.is_row(token) for token in generated),
        "before_sha256": digest(before),
        "after_sha256": digest(after),
        "world_sha256_after_each_generation": [digest(world) for world in worlds],
    }


def run(args) -> dict:
    runtime_dir = args.runtime_dir
    state_path = runtime_dir / "state.json"
    neural_path = runtime_dir / "neural_state.pt"
    if not args.resume and (state_path.exists() or neural_path.exists()):
        raise FileExistsError(
            f"{runtime_dir} already contains a world; choose a new directory or pass --resume"
        )
    runtime_dir.mkdir(parents=True, exist_ok=True)

    runtime = InternalRuntime(
        args.model,
        state_path,
        neural_path,
        args.device,
        args.max_new_tokens,
    )
    protocol = runtime.protocol
    report: dict = {
        "status": "running",
        "checkpoint": str(args.model.resolve()),
        "runtime_dir": str(runtime_dir.resolve()),
        "no_training": True,
        "host_semantic_parser": False,
        "protocol": {
            "row_base": protocol.row_base,
            "reset_token": protocol.reset_token,
            "tick_token": protocol.tick_token,
            "done_token": protocol.done_token,
            "grid_layer": protocol.grid_layer,
            "grid_heads": list(protocol.grid_heads),
            "model_vocab": protocol.model_vocab,
        },
        "cases": [],
    }
    if not args.resume and runtime.rows != tuple(INITIAL_ROWS):
        raise AssertionError("fresh internal runtime did not start from INITIAL_ROWS")

    # Conversation and negation must leave a nonempty starting world intact.
    report["cases"].append(natural_case(
        runtime,
        "ordinary_chat",
        "Hello. Just greet me without changing or inspecting the world.",
        lambda rows: rows,
        expected_ticks=0,
        expected_commits=0,
    ))
    report["cases"].append(natural_case(
        runtime,
        "negation_no_mutation",
        "Do not clear, advance, or otherwise change the world. Just acknowledge this request.",
        lambda rows: rows,
        expected_ticks=0,
        expected_commits=None,
    ))
    report["cases"].append(natural_case(
        runtime,
        "clear",
        "Clear the entire world now.",
        lambda _rows: (0,) * 16,
        expected_ticks=0,
        expected_commits=1,
    ))
    report["cases"].append(natural_case(
        runtime,
        "place_glider",
        "Place a glider with its origin at row 2 column 4.",
        lambda rows: place(rows, "glider", 2, 4),
        expected_ticks=0,
        expected_commits=1,
    ))
    report["cases"].append(natural_case(
        runtime,
        "advance_three",
        "Advance exactly three generations.",
        lambda rows: step(step(step(rows))),
        expected_ticks=3,
        expected_commits=3,
    ))
    report["cases"].append(natural_case(
        runtime,
        "cell_on",
        "Make the cell at row 10 column 11 alive.",
        lambda rows: set_cell(rows, 10, 11, 1),
        expected_ticks=0,
        expected_commits=1,
    ))

    before_restart = {
        "rows": runtime.rows,
        "generation": runtime.generation,
        "conversation": list(runtime.conversation),
    }
    del runtime
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    restored = InternalRuntime(
        args.model,
        state_path,
        neural_path,
        args.device,
        args.max_new_tokens,
    )
    persistence = {
        "rows_preserved": restored.rows == before_restart["rows"],
        "generation_preserved": restored.generation == before_restart["generation"],
        "conversation_preserved": restored.conversation == before_restart["conversation"],
    }
    if not all(persistence.values()):
        raise AssertionError(f"cold runtime restart did not preserve state: {persistence}")
    persistence["passed"] = True
    persistence["row_sha256"] = digest(restored.rows)
    persistence["conversation_records"] = len(restored.conversation)
    report["persistence"] = persistence

    report["cases"].append(natural_case(
        restored,
        "post_restart_advance",
        "Advance one generation.",
        lambda rows: step(rows),
        expected_ticks=1,
        expected_commits=1,
    ))
    report["scripted_model_protocol"] = scripted_advance_check(restored, 2)

    saved = torch.load(neural_path, weights_only=True)
    report["snapshot"] = {
        "dtype": str(saved["states"].dtype),
        "shape": list(saved["states"].shape),
        "layer": int(saved["layer"]),
        "heads": list(map(int, saved["heads"])),
    }
    if report["snapshot"]["dtype"] != "torch.float32":
        raise AssertionError("saved recurrent state was narrowed below FP32")
    report["final"] = {
        "generation": restored.generation,
        "population": population(restored.rows),
        "row_sha256": digest(restored.rows),
        "conversation_records": len(restored.conversation),
    }
    report["status"] = "passed"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    report: dict
    try:
        report = run(args)
    except Exception as error:
        report = {
            "status": "failed",
            "checkpoint": str(args.model.resolve()),
            "runtime_dir": str(args.runtime_dir.resolve()),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        raise
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
