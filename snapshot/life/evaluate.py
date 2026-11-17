"""End-to-end exactness, persistence, and ordinary-text checks for a saved merge."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from .host import Feeder, load_model
from .merge import empty_model, qwen_tensors, text_config
from .monitor import read, restore, snapshot
from .reference import INITIAL_ROWS, step


def life_check(model_path: str, generations: int, device: str) -> dict:
    model, offset = load_model(model_path, device)
    feeder = Feeder(model, offset)
    feeder.reset(INITIAL_ROWS)
    expected = tuple(INITIAL_ROWS)
    owned = 0
    initialization_ticks = feeder.tokens
    start = time.perf_counter()
    for generation in range(1, generations + 1):
        expected = step(expected)
        generated = feeder.generation()
        owned += 17
        if generated != expected or read(feeder.cache) != expected:
            raise AssertionError(f"saved merge failed at generation {generation}")

    saved = snapshot(feeder.cache)
    fresh = Feeder(model, offset, restore(model, saved))
    expected = step(expected)
    generated = fresh.generation()
    owned += 17
    if generated != expected or read(fresh.cache) != expected:
        raise AssertionError("fresh-cache continuation failed")
    elapsed = time.perf_counter() - start
    timed_ticks = feeder.tokens - initialization_ticks + fresh.tokens
    return {
        "generations": generations + 1,
        "owned_outputs_correct": owned,
        "owned_outputs_total": owned,
        "snapshot_continuation": True,
        "initialization_ticks": initialization_ticks,
        "ticks": timed_ticks,
        "total_ticks": feeder.tokens + fresh.tokens,
        "seconds": elapsed,
        "ticks_per_second": timed_ticks / elapsed,
    }


@torch.no_grad()
def text_check(base_path: str, merged_path: str, device: str) -> dict:
    base = empty_model(Qwen3_5TextConfig(**text_config(base_path)))
    base.load_state_dict(qwen_tensors(base_path), strict=False)
    base.tie_weights()
    base = base.to(device).eval()
    merged, _ = load_model(merged_path, device)
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Explain why the sky appears blue in two sentences.",
        "def fibonacci(n):\n    if n < 2:\n        return n\n",
        "A careful scientist records each observation before drawing a conclusion.",
    ]
    kls, agree, positions = [], 0, 0
    for text in texts:
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        a = base(input_ids=ids).logits[0].float()
        b = merged(input_ids=ids).logits[0, :, : a.shape[-1]].float()
        loga, logb = a.log_softmax(-1), b.log_softmax(-1)
        kls.extend((loga.exp() * (loga - logb)).sum(-1).cpu().tolist())
        agree += int((a.argmax(-1) == b.argmax(-1)).sum())
        positions += a.shape[0]
    return {
        "positions": positions,
        "top1_agreement": agree,
        "mean_kl_base_to_merged": sum(kls) / len(kls),
        "max_kl_base_to_merged": max(kls),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("merged")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-text", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args()
    result = {"status": "success", "life": life_check(args.merged, args.generations, args.device)}
    if not args.skip_text:
        # This deliberately compares only the original Qwen vocabulary.  The
        # merged checkpoint has appended Life tokens, so an unmasked argmax is
        # a different question and would make the metric easy to misread.
        result["base_vocab_masked"] = text_check(args.base, args.merged, args.device)
    output = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
