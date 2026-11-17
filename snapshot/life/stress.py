"""Stress the saved merged checkpoint against the independent Life oracle.

This is deliberately an end-to-end model test.  It loads the checkpoint from
disk, clocks its native causal cache one token at a time, reads the recurrent
grid heads directly, and treats :mod:`life.reference` only as an external
oracle.  It never substitutes the oracle for model execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .host import Feeder, load_model
from .monitor import load as load_snapshot
from .monitor import read, restore, save as save_snapshot, snapshot
from .reference import HEIGHT, WIDTH, step


@dataclass(frozen=True)
class Case:
    name: str
    rows: tuple[int, ...]
    recycle_every: int


def _points(*points: tuple[int, int]) -> tuple[int, ...]:
    rows = [0] * HEIGHT
    for y, x in points:
        rows[y] |= 1 << x
    return tuple(rows)


def _random_rows(rng: random.Random, density: float) -> tuple[int, ...]:
    return tuple(
        sum((rng.random() < density) << x for x in range(WIDTH))
        for _ in range(HEIGHT)
    )


def cases(seed: int, random_boards: int) -> list[Case]:
    fixed = [
        Case("empty", (0,) * HEIGHT, 1),
        Case("full", ((1 << WIDTH) - 1,) * HEIGHT, 2),
        Case("single-corner", _points((0, 0)), 5),
        Case("single-center", _points((8, 8)), 7),
        Case("block-corner", _points((0, 0), (0, 1), (1, 0), (1, 1)), 11),
        Case("blinker-top", _points((0, 6), (0, 7), (0, 8)), 0),
        Case(
            "glider-bottom-right",
            _points((13, 14), (14, 15), (15, 13), (15, 14), (15, 15)),
            17,
        ),
        Case(
            "checkerboard",
            tuple(sum(((x + y) & 1) << x for x in range(WIDTH)) for y in range(HEIGHT)),
            3,
        ),
    ]
    rng = random.Random(seed)
    densities = (0.01, 0.05, 0.15, 0.30, 0.50, 0.70, 0.90, 0.99)
    cadences = (0, 1, 2, 4, 9, 16)
    for index in range(random_boards):
        density = densities[index % len(densities)]
        fixed.append(
            Case(
                f"random-{index:02d}-density-{density:.2f}",
                _random_rows(rng, density),
                cadences[index % len(cadences)],
            )
        )
    return fixed


def _cache_length(cache, full_attention_layers: list[int]) -> int:
    lengths = []
    for layer in full_attention_layers:
        try:
            lengths.append(int(cache.get_seq_length(layer)))
        except (AttributeError, IndexError, TypeError):
            continue
    return max(lengths, default=0)


def _rows_bytes(rows: tuple[int, ...]) -> bytes:
    return b"".join(int(row).to_bytes(2, "little") for row in rows)


def _parameter_sample_digest(model, samples_per_tensor: int = 17) -> str:
    """Hash deterministic samples from every parameter tensor.

    PyTorch version counters below cover every in-memory parameter.  This
    digest adds a value-level check over every tensor without copying the full
    2.3 GB checkpoint back to the CPU twice.
    """
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        flat = parameter.detach().reshape(-1)
        count = min(samples_per_tensor, flat.numel())
        # Integer construction avoids float32 linspace rounding the final index
        # up to ``numel`` for very large embedding matrices.
        if count == 1:
            positions = [0]
        else:
            positions = [i * (flat.numel() - 1) // (count - 1) for i in range(count)]
        indices = torch.tensor(positions, device=flat.device, dtype=torch.long)
        values = flat.index_select(0, indices).float().cpu().numpy().tobytes()
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(str(parameter.dtype).encode())
        digest.update(values)
    return digest.hexdigest()


def run_case(
    model,
    offset: int,
    case: Case,
    generations: int,
    directory: Path,
    full_attention_layers: list[int],
    persist_midpoint: bool = True,
) -> dict:
    feeder = Feeder(model, offset)
    feeder.reset(case.rows)
    if read(feeder.cache) != case.rows:
        raise AssertionError(f"{case.name}: RESET did not reach recurrent state")
    initial_state_dtype = str(snapshot(feeder.cache)["states"].dtype)

    expected = case.rows
    digest = hashlib.sha256(_rows_bytes(expected))
    calls = feeder.tokens
    owned = 0
    recycle_events = 0
    disk_roundtrips = 0
    max_context = _cache_length(feeder.cache, full_attention_layers)
    snapshot_bytes = 0
    disk_at = max(1, generations // 2)

    for generation in range(1, generations + 1):
        before = feeder.tokens
        emitted = feeder.generation()
        calls += feeder.tokens - before
        owned += 17
        expected = step(expected)
        state = read(feeder.cache)
        if emitted != expected or state != expected:
            raise AssertionError(
                f"{case.name} generation {generation}: "
                f"emitted={emitted == expected}, state={state == expected}"
            )
        digest.update(_rows_bytes(state))
        max_context = max(max_context, _cache_length(feeder.cache, full_attention_layers))

        should_recycle = bool(case.recycle_every and generation % case.recycle_every == 0)
        should_persist = persist_midpoint and generation == disk_at
        if should_recycle or should_persist:
            saved = snapshot(feeder.cache)
            if should_persist:
                path = directory / f"{case.name}.pt"
                save_snapshot(saved, path)
                snapshot_bytes = max(snapshot_bytes, path.stat().st_size)
                saved = load_snapshot(path)
                disk_roundtrips += 1
            feeder = Feeder(model, offset, cache=restore(model, saved))
            if read(feeder.cache) != expected:
                raise AssertionError(f"{case.name}: snapshot restore changed the grid")
            restored_dtype = str(snapshot(feeder.cache)["states"].dtype)
            if restored_dtype != str(saved["states"].dtype):
                raise AssertionError(
                    f"{case.name}: restore narrowed {saved['states'].dtype} to {restored_dtype}"
                )
            recycle_events += 1
            if _cache_length(feeder.cache, full_attention_layers) != 0:
                raise AssertionError(f"{case.name}: full-attention history survived recycle")

    return {
        "name": case.name,
        "generations": generations,
        "recycle_every": case.recycle_every,
        "recycle_events": recycle_events,
        "disk_roundtrips": disk_roundtrips,
        "midpoint_persistence_enabled": persist_midpoint,
        "snapshot_bytes": snapshot_bytes,
        "model_calls": calls,
        "owned_outputs": owned,
        "max_full_attention_context": max_context,
        "recurrent_state_dtype": initial_state_dtype,
        "final_population": sum(row.bit_count() for row in expected),
        "trace_sha256": digest.hexdigest(),
        "final_rows": expected,
    }


def run(model_path: str, generations: int, random_boards: int, seed: int, device: str) -> dict:
    if generations < 2:
        raise ValueError("generations must be at least two for persistence checks")
    load_started = time.perf_counter()
    model, offset = load_model(model_path, device)
    load_seconds = time.perf_counter() - load_started
    full_attention_layers = [
        index for index, kind in enumerate(model.config.layer_types) if kind == "full_attention"
    ]
    versions = tuple(parameter._version for parameter in model.parameters())
    parameter_digest_before = _parameter_sample_digest(model)
    if torch.cuda.is_available() and model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
        torch.cuda.synchronize(model.device)

    suite_started = time.perf_counter()
    results = []
    with tempfile.TemporaryDirectory(prefix="qwen-life-stress-") as temporary:
        directory = Path(temporary)
        for case in cases(seed, random_boards):
            results.append(
                run_case(model, offset, case, generations, directory, full_attention_layers)
            )

        # The same world must have the same complete trace whether history is
        # retained or the neural grid is restored after every generation.
        rng = random.Random(seed ^ 0x51F15EED)
        rows = _random_rows(rng, 0.37)
        plain = run_case(
            model,
            offset,
            Case("determinism-no-recycle", rows, 0),
            generations,
            directory,
            full_attention_layers,
            persist_midpoint=False,
        )
        recycled = run_case(
            model,
            offset,
            Case("determinism-recycle-every-step", rows, 1),
            generations,
            directory,
            full_attention_layers,
        )
        if plain["trace_sha256"] != recycled["trace_sha256"]:
            raise AssertionError("cache recycling changed the deterministic trace")

        # Two contradictory worlds share one immutable model but carry fully
        # independent recurrent caches.  Interleave their execution so any
        # accidental cache aliasing is exposed immediately.
        board_a = (0,) * HEIGHT
        board_b = _points((7, 7), (7, 8), (8, 7), (8, 8))
        expected_a, expected_b = board_a, board_b
        feeder_a, feeder_b = Feeder(model, offset), Feeder(model, offset)
        feeder_a.reset(board_a)
        feeder_b.reset(board_b)
        if feeder_a.cache is feeder_b.cache:
            raise AssertionError("independent worlds unexpectedly share a cache object")
        for generation in range(1, generations + 1):
            expected_a, expected_b = step(expected_a), step(expected_b)
            emitted_a, emitted_b = feeder_a.generation(), feeder_b.generation()
            if emitted_a != expected_a or read(feeder_a.cache) != expected_a:
                raise AssertionError(f"independent world A failed at {generation}")
            if emitted_b != expected_b or read(feeder_b.cache) != expected_b:
                raise AssertionError(f"independent world B failed at {generation}")
            if read(feeder_a.cache) == read(feeder_b.cache):
                raise AssertionError("contradictory worlds collapsed to the same state")
        independent_calls = feeder_a.tokens + feeder_b.tokens

        # RESET must overwrite an existing live cache, not merge with it.
        feeder = Feeder(model, offset)
        feeder.reset(rows)
        reset_expected = step(rows)
        reset_emitted = feeder.generation()
        if reset_emitted != reset_expected or read(feeder.cache) != reset_expected:
            raise AssertionError("pre-overwrite generation failed")
        replacement = _random_rows(rng, 0.12)
        feeder.reset(replacement)
        if read(feeder.cache) != replacement:
            raise AssertionError("RESET did not overwrite an existing world")
        reset_overwrite_calls = feeder.tokens

        # A TICK without a loaded world is deliberately outside the low-level
        # protocol; the guarded high-level generation API must fail closed.
        fresh = Feeder(model, offset)
        try:
            fresh.generation()
        except RuntimeError:
            uninitialized_tick_rejected = True
        else:
            uninitialized_tick_rejected = False
            raise AssertionError("generation before RESET was not rejected")

    # Normal evolution must not write the checkpoint.  Check every parameter's
    # version counter and deterministic value samples before deliberate
    # ablation below.
    versions_after_evolution = tuple(parameter._version for parameter in model.parameters())
    parameter_digest_after_evolution = _parameter_sample_digest(model)
    if versions_after_evolution != versions:
        raise AssertionError("a parameter version changed during ordinary inference")
    if parameter_digest_after_evolution != parameter_digest_before:
        raise AssertionError("sampled parameter values changed during ordinary inference")

    # Discriminating ablation: erase the MLP output weights in the constructed
    # rule layer.  A known oscillator must stop matching the oracle.  Restore
    # the tensor byte-for-byte and prove exact execution returns.
    ablation_name = "model.layers.1.mlp.down_proj.weight"
    target = dict(model.named_parameters())[ablation_name]
    backup = target.detach().clone()
    oscillator = _points((8, 7), (8, 8), (8, 9))
    expected_oscillator = step(oscillator)
    broken_feeder = Feeder(model, offset)
    try:
        with torch.no_grad():
            target.zero_()
        try:
            broken_emitted = broken_feeder.reset(oscillator) or broken_feeder.generation()
            ablation_broke_exactness = (
                broken_emitted != expected_oscillator
                or read(broken_feeder.cache) != expected_oscillator
            )
        except AssertionError:
            ablation_broke_exactness = True
    finally:
        with torch.no_grad():
            target.copy_(backup)
    if not ablation_broke_exactness:
        raise AssertionError("zeroing the constructed rule tensor did not break exactness")
    restored_feeder = Feeder(model, offset)
    restored_feeder.reset(oscillator)
    restored_emitted = restored_feeder.generation()
    if restored_emitted != expected_oscillator or read(restored_feeder.cache) != expected_oscillator:
        raise AssertionError("exactness did not return after restoring the ablated tensor")
    parameter_digest_after_restore = _parameter_sample_digest(model)
    if parameter_digest_after_restore != parameter_digest_before:
        raise AssertionError("ablation tensor was not restored to its original values")

    if torch.cuda.is_available() and model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    elapsed = time.perf_counter() - suite_started
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("inference unexpectedly populated parameter gradients")

    all_results = results + [plain, recycled]
    total_generations = (
        sum(result["generations"] for result in all_results)
        + 2 * generations  # the two interleaved independent worlds
        + 1  # the checked generation before RESET overwrite
        + 1  # exact generation after restoring the ablated tensor
    )
    ablation_calls = broken_feeder.tokens + restored_feeder.tokens
    total_calls = (
        sum(result["model_calls"] for result in all_results)
        + independent_calls
        + reset_overwrite_calls
        + fresh.tokens
        + ablation_calls
    )
    total_owned = (
        sum(result["owned_outputs"] for result in all_results)
        + 2 * generations * 17
        + 17
        + 17
    )
    max_snapshot = max(result["snapshot_bytes"] for result in all_results)
    peak_gpu = (
        int(torch.cuda.max_memory_allocated(model.device))
        if torch.cuda.is_available() and model.device.type == "cuda"
        else None
    )
    return {
        "status": "success",
        "checkpoint": str(model_path),
        "device": str(model.device),
        "load_seconds": load_seconds,
        "cases": len(all_results),
        "random_seed": seed,
        "generations_per_case": generations,
        "total_generations": total_generations,
        "total_model_calls": total_calls,
        "total_owned_outputs_correct": total_owned,
        "oracle_state_emission_agreement": True,
        "determinism_across_recycling": True,
        "uninterrupted_vs_recycle_every_step": {
            "passed": True,
            "generations": generations,
            "uninterrupted_disk_roundtrips": plain["disk_roundtrips"],
            "uninterrupted_max_full_attention_context": plain["max_full_attention_context"],
            "recycled_max_full_attention_context": recycled["max_full_attention_context"],
            "trace_sha256": plain["trace_sha256"],
        },
        "independent_contradictory_caches": {
            "passed": True,
            "generations_each": generations,
            "world_a_final_population": sum(row.bit_count() for row in expected_a),
            "world_b_final_population": sum(row.bit_count() for row in expected_b),
        },
        "disk_snapshot_roundtrips": sum(r["disk_roundtrips"] for r in all_results),
        "max_snapshot_bytes": max_snapshot,
        "uninitialized_tick_rejected": uninitialized_tick_rejected,
        "reset_overwrite_verified": True,
        "parameter_versions_unchanged_during_evolution": True,
        "parameter_sample_sha256_before": parameter_digest_before,
        "parameter_sample_sha256_after_evolution": parameter_digest_after_evolution,
        "parameter_sample_sha256_after_ablation_restore": parameter_digest_after_restore,
        "parameter_gradients_absent": True,
        "constructed_weight_ablation": {
            "tensor": ablation_name,
            "zeroing_broke_oracle_exactness": ablation_broke_exactness,
            "restoring_tensor_restored_exactness": True,
        },
        "full_attention_layers": full_attention_layers,
        "suite_seconds": elapsed,
        "model_calls_per_second": total_calls / elapsed,
        "generations_per_second": total_generations / elapsed,
        "peak_gpu_memory_bytes_after_load": peak_gpu,
        "case_results": [
            {key: value for key, value in result.items() if key != "final_rows"}
            for result in all_results
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--random-boards", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out")
    args = parser.parse_args()
    result = run(args.model, args.generations, args.random_boards, args.seed, args.device)
    output = json.dumps(result, indent=2)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
