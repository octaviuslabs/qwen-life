"""Probe Qwen's native Life tool selection before installing the executor.

This deliberately masks the old appended Life vocabulary: the existing
checkpoint's circuit owns those rows while idle.  The replacement checkpoint
will instead have an analytically programmed, zero-when-idle output bus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from life.host import load_model
from life.internal_tools import CONTROLLER_SYSTEM, TOOLS


DEFAULT_PROMPTS = [
    "hello",
    "place a glider",
    "put a glider somewhere near the top",
    "place something interesting",
    "advance three generations",
    "make row 7 column 9 alive",
    "turn off cell 4, 12",
    "clear the world",
    "do not clear the world",
    "what is happening in this world?",
]


@torch.no_grad()
def generate(model, tokenizer, prompt: str, base_vocab: int, limit: int) -> tuple[str, list[int]]:
    encoded = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": CONTROLLER_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        tools=TOOLS,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )
    ids = encoded["input_ids"].to(model.device)
    out = model(input_ids=ids, use_cache=True)
    cache = out.past_key_values
    generated: list[int] = []
    eos = tokenizer.eos_token_id
    eos_ids = {int(eos)} if isinstance(eos, int) else {int(x) for x in eos or []}
    for _ in range(limit):
        token = int(out.logits[0, -1, :base_vocab].float().argmax())
        if token in eos_ids:
            break
        generated.append(token)
        current = torch.tensor([[token]], device=model.device)
        out = model(input_ids=current, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
    return tokenizer.decode(generated, skip_special_tokens=False).strip(), generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("prompts", nargs="*")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--base-vocab", type=int, default=248_320)
    parser.add_argument("--limit", type=int, default=96)
    args = parser.parse_args()

    model, _ = load_model(str(args.model), args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    results = []
    for prompt in args.prompts or DEFAULT_PROMPTS:
        text, ids = generate(model, tokenizer, prompt, args.base_vocab, args.limit)
        result = {"prompt": prompt, "output": text, "token_ids": ids}
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
