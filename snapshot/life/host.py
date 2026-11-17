"""Clock a tiny or merged packed-row Life checkpoint and render its neural state."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from prototype.circuit import DONE, RESET, TICK, VAL_COUNT, VOCAB_SIZE, token_name

from .monitor import read
from .reference import render


class Feeder:
    def __init__(self, model, offset: int = 0, cache=None):
        self.model, self.offset, self.cache = model, offset, cache
        self.tokens = 0
        self.initialized = cache is not None

    @torch.no_grad()
    def feed(self, token: int) -> int:
        token = int(token)
        if not 0 <= token < VOCAB_SIZE:
            raise ValueError(f"Life token is outside 0..{VOCAB_SIZE - 1}: {token}")
        x = torch.tensor([[self.offset + token]], device=self.model.device)
        out = self.model(input_ids=x, past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values
        self.tokens += 1
        predicted = int(out.logits[0, -1].float().argmax()) - self.offset
        if not 0 <= predicted < VOCAB_SIZE:
            raise AssertionError(f"circuit lost ownership: model token {predicted + self.offset}")
        return predicted

    def reset(self, rows) -> None:
        rows = tuple(int(row) for row in rows)
        if len(rows) != 16:
            raise ValueError(f"expected sixteen rows, got {len(rows)}")
        if any(not 0 <= row < VAL_COUNT for row in rows):
            raise ValueError("every row must be an unsigned 16-bit value")
        self.feed(RESET)
        for row in rows:
            self.feed(int(row))
        self.feed(DONE)
        self.initialized = True

    def generation(self) -> tuple[int, ...]:
        if not self.initialized:
            raise RuntimeError("RESET or a restored neural snapshot is required before TICK")
        token, rows = TICK, []
        for _ in range(16):
            token = self.feed(token)
            if not 0 <= token < VAL_COUNT:
                raise AssertionError(f"expected row, got {token_name(token)}")
            rows.append(token)
        token = self.feed(token)  # feed row 15 so it commits; prediction must be DONE
        if token != DONE:
            raise AssertionError(f"expected DONE, got {token_name(token)}")
        self.feed(DONE)  # close the phase and reset its row counter
        return tuple(rows)


def load_model(path: str, device: str):
    model = Qwen3_5ForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()
    meta_path = Path(path) / "life.json"
    offset = json.loads(meta_path.read_text()).get("vocab_offset", 0) if meta_path.exists() else 0
    return model, offset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--delay", type=float, default=0.15)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    model, offset = load_model(args.model, args.device)
    from .reference import INITIAL_ROWS, step

    feeder = Feeder(model, offset)
    feeder.reset(INITIAL_ROWS)
    expected = tuple(INITIAL_ROWS)
    start = time.perf_counter()
    print("generation 0\n" + render(read(feeder.cache)))
    for generation in range(1, args.generations + 1):
        expected = step(expected)
        predicted = feeder.generation()
        state = read(feeder.cache)
        if predicted != expected or state != expected:
            raise AssertionError(
                f"generation {generation}: generated={predicted == expected}, state={state == expected}"
            )
        print(f"\ngeneration {generation}\n{render(state)}")
        if args.delay:
            time.sleep(args.delay)
    elapsed = time.perf_counter() - start
    print(f"\n{feeder.tokens} ticks in {elapsed:.2f}s ({feeder.tokens / elapsed:.1f}/s)")


if __name__ == "__main__":
    main()

