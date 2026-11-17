"""Audit that native Qwen tensors survived the analytic merge."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import torch
from safetensors import safe_open

PREFIX = "model.language_model."
SHIFT = 3
NORM = re.compile(
    r"(input_layernorm|post_attention_layernorm)\.weight$|^model\.norm\.weight$"
)


def index(path: Path) -> dict:
    return json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]


def tensor(path: Path, mapping: dict, key: str) -> torch.Tensor:
    with safe_open(str(path / mapping[key]), framework="pt") as sf:
        return sf.get_tensor(key)


def target_key(source: str) -> str:
    key = "model." + source[len(PREFIX):]
    return re.sub(
        r"^model\.layers\.(\d+)\.",
        lambda match: f"model.layers.{int(match[1]) + SHIFT}.",
        key,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(base: str | Path, merged: str | Path) -> dict:
    base, merged = Path(base), Path(merged)
    bi, mi = index(base), index(merged)
    exact = compensated = 0
    scale = math.sqrt(1024 / 1408)
    for source in sorted(key for key in bi if key.startswith(PREFIX)):
        target = target_key(source)
        if target not in mi:
            raise AssertionError(f"missing {target}")
        a, b = tensor(base, bi, source), tensor(merged, mi, target)
        overlap = b[tuple(slice(0, size) for size in a.shape)]
        if NORM.search(target):
            expected = ((1 + a.float()) * scale - 1).to(overlap.dtype)
            if not torch.equal(overlap, expected):
                raise AssertionError(f"norm compensation mismatch: {target}")
            compensated += 1
        else:
            if not torch.equal(overlap, a.to(overlap.dtype)):
                raise AssertionError(f"native tensor changed: {target}")
            exact += 1
    files = sorted(merged.glob("*.safetensors")) + [merged / "model.safetensors.index.json", merged / "config.json"]
    return {
        "native_tensors_exact": exact,
        "native_norms_analytically_compensated": compensated,
        "native_audit_passed": True,
        "checkpoint_sha256": {file.name: sha256(file) for file in files},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("merged")
    parser.add_argument("--out")
    args = parser.parse_args()
    result = audit(args.base, args.merged)
    output = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
