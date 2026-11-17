"""Emit LifeCircuit into real Qwen3.5 modules and check greedy execution exactly."""

from __future__ import annotations

import argparse
import time

import torch

from chipformer.q35 import tiny
from life.host import Feeder
from life.monitor import read, restore, snapshot
from life.reference import INITIAL_ROWS, step
from prototype.circuit import LifeCircuit, VOCAB


def build_model(device: str):
    circuit = LifeCircuit()
    code = [next(iter(expr.terms)) for expr in circuit.code]
    model = tiny.build(circuit, VOCAB, code, dtype=torch.bfloat16, share=True)
    return circuit, model.to(device).eval()


def run(generations: int = 10, device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    circuit, model = build_model(device)
    feeder = Feeder(model)
    feeder.reset(INITIAL_ROWS)
    if read(feeder.cache) != INITIAL_ROWS:
        raise AssertionError("RESET did not reach recurrent state")
    expected = tuple(INITIAL_ROWS)
    start = time.perf_counter()
    for generation in range(1, generations + 1):
        expected = step(expected)
        generated = feeder.generation()
        state = read(feeder.cache)
        if generated != expected or state != expected:
            raise AssertionError(
                f"generation {generation}: generated={generated == expected}, state={state == expected}"
            )

    saved = snapshot(feeder.cache)
    restored_cache = restore(model, saved)
    restored = snapshot(restored_cache)
    if restored["states"].dtype != saved["states"].dtype:
        raise AssertionError(
            f"snapshot restore narrowed {saved['states'].dtype} "
            f"to {restored['states'].dtype}"
        )
    fresh = Feeder(model, cache=restored_cache)
    expected = step(expected)
    generated = fresh.generation()
    if generated != expected or read(fresh.cache) != expected:
        raise AssertionError("fresh-cache snapshot continuation failed")
    elapsed = time.perf_counter() - start
    return {
        "status": "exact",
        "generations": generations + 1,
        "forced_rows_and_done": (generations + 1) * 17,
        "ticks": feeder.tokens + fresh.tokens,
        "seconds": elapsed,
        "ticks_per_second": (feeder.tokens + fresh.tokens) / elapsed,
        "layers": circuit.n_layers(),
        "hidden": model.config.hidden_size,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "recurrent_state_dtype": str(saved["states"].dtype),
        "restore_preserved_recurrent_dtype": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--device")
    args = parser.parse_args()
    import json

    print(json.dumps(run(args.generations, args.device), indent=2))


if __name__ == "__main__":
    main()

