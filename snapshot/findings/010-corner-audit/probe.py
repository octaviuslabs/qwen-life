"""Separate-world tests of the real v3 checkpoint; never opens the live cache."""
import json
from pathlib import Path
import random
import time

import torch
from life.internal_runtime import InternalRuntime, atomic_json, read_rows, restore_grid, snapshot_grid
from life.reference import step
from prototype.internal_circuit import ASSISTANT, IM_END, IM_START, tool_call_tokens

HERE = Path(__file__).resolve().parent
RUN = HERE / "isolated-runtime"
REPORT = HERE / "report.json"


def points(rows):
    return [(r, c) for r in range(16) for c in range(16) if rows[r] >> c & 1]


def board(cells):
    rows = [0] * 16
    for r, c in cells:
        rows[r] |= 1 << c
    return tuple(rows)


@torch.no_grad()
def seed(runtime, rows):
    cache = None
    p = runtime.protocol
    for token in [p.reset_token, *(p.row_base + r for r in rows), p.done_token]:
        cache, _ = runtime._feed([token], cache)
    runtime.cache = cache
    runtime.conversation = []
    runtime.generation = 0
    assert runtime.rows == rows


@torch.no_grad()
def advance_batch(runtime, expected, count, case):
    p = runtime.protocol
    cache = restore_grid(runtime.model, snapshot_grid(runtime.cache, p), p)
    for token in (IM_START, ASSISTANT, *tool_call_tokens("advance", count)):
        cache, logits = runtime._feed([token], cache)
    expected_tokens = []
    worlds = []
    for _ in range(count):
        expected = step(expected)
        worlds.append(expected)
        expected_tokens += [p.tick_token, *(p.row_base + row for row in expected)]
    expected_tokens += [p.done_token, IM_END]
    for i, want in enumerate(expected_tokens):
        actual = int(logits.float().argmax())
        if actual != want:
            raise AssertionError(f"{case['name']}: private token {i}: {actual} != {want}")
        case["checked_tokens"] += 1
        if actual == IM_END:
            break
        cache, logits = runtime._feed([actual], cache)
        if i < count * 17 and i % 17 == 16:
            row_state = read_rows(cache, p)
            expected_state = worlds[i // 17]
            if row_state != expected_state:
                raise AssertionError(f"{case['name']}: state differs at generation {case['checked_generations'] + 1}")
            case["checked_generations"] += 1
    runtime.cache = cache
    assert runtime.rows == expected
    return expected


def main():
    if RUN.exists():
        raise FileExistsError("Use a fresh isolated runtime directory")
    RUN.mkdir()
    report = {"status": "running", "checkpoint": "build/qwen-life-internal-08b-v3",
              "isolated_from_live_world": True, "cases": [], "language_probes": []}
    atomic_json(REPORT, report)
    runtime = InternalRuntime(Path(report["checkpoint"]), RUN / "state.json", RUN / "neural_state.pt", "cuda", 512)
    rng = random.Random(20261106)
    selected = [
        ("center_block_and_blinker", board([(4,4),(4,5),(5,4),(5,5),(10,6),(10,7),(10,8)]), 64),
        ("off_diagonal_glider", board([(2,9),(3,10),(4,8),(4,9),(4,10)]), 64),
        ("random_board", board([(r,c) for r in range(16) for c in range(16) if rng.random()<.28]), 32),
    ]
    try:
        for name, initial, generations in selected:
            seed(runtime, initial)
            case = {"name": name, "initial_points": points(initial), "planned_generations": generations,
                    "checked_generations": 0, "checked_tokens": 0, "passed": False}
            report["cases"].append(case)
            started = time.monotonic()
            expected = initial
            while case["checked_generations"] < generations:
                count = min(15, generations - case["checked_generations"])
                expected = advance_batch(runtime, expected, count, case)
                case["final_points"] = points(runtime.rows)
                atomic_json(REPORT, report)
                print(f"{name}: {case['checked_generations']}/{generations} exact", flush=True)
            case.update(passed=True, seconds=round(time.monotonic()-started,3))
            atomic_json(REPORT, report)

        contexts = [
            ("interesting_items", "place an interesting set of items", []),
            ("unique_fractal", "idk something unique and fractal", [
                {"role":"user", "text":"put something cool in", "kind":"user-chat"},
                {"role":"assistant", "kind":"chat", "text":"I'd love to help you create something cool! What kind of pattern or design would you like to see?\n\nFor example:\n- A glider\n- A blinker\n- A block\n- A specific pattern\n- Something more abstract\n\nJust let me know what you're thinking, and I'll create it for you!"},
            ]),
        ]
        for name, prompt, history in contexts:
            seed(runtime, (0,)*16)
            runtime.conversation = history
            result = runtime.submit({"type":"say", "text":prompt})
            public = [t for t in result.generated if not runtime.protocol.is_private(t)]
            report["language_probes"].append({"name":name, "prompt":prompt,
                "generated_public_text":runtime.tokenizer.decode(public, skip_special_tokens=False),
                "row_commits":result.row_commits, "result_points":points(runtime.rows)})
            atomic_json(REPORT, report)
            print(f"language probe: {name}", flush=True)
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        atomic_json(REPORT, report)


if __name__ == "__main__":
    main()
