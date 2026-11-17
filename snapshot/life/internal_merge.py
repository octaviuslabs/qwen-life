"""Install the native-tool Life controller into pristine Qwen3.5-0.8B.

Unlike :mod:`life.merge`, this variant gives every Qwen token circuit features
in the widened residual stream.  The added front layers can therefore read
Qwen's own ``<tool_call>`` tokens, execute the requested operation, and force
the private row-token trace.  Their exit clears every added dimension except
an output-code bus; when the bus is zero, Qwen's ordinary logits are untouched.

All new tensors are assigned by construction.  There is no dataset, loss,
optimizer, gradient, LoRA, or fine-tuning step.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

PROJECT = Path(__file__).resolve().parents[1]
for dependency in (
    PROJECT / "references" / "qwenputer",
    PROJECT / "references" / "qwenGPU",
):
    value = str(dependency)
    if value not in sys.path:
        sys.path.insert(0, value)

from chipformer.q35.emit import Layout, allocate, embedding_rows, emit_layers
from chipformer.q35.merge import pick_hidden
from chipformer.q35.tiny import code_gain
from prototype.internal_circuit import (
    BASE_VOCAB,
    DONE,
    MODEL_VOCAB,
    RESET,
    ROW_BASE,
    TICK,
    VAL_COUNT,
    InternalLifeCircuit,
)


PREFIX = "model.language_model."
VOCAB_ALIGNMENT = 64


def aligned_vocab(last_token: int = DONE) -> int:
    """Model vocabulary size, including a small unreachable alignment tail."""
    return -(-(int(last_token) + 1) // VOCAB_ALIGNMENT) * VOCAB_ALIGNMENT


def text_config(src: str | Path) -> dict:
    cfg = json.loads((Path(src) / "config.json").read_text(encoding="utf-8"))["text_config"]
    cfg.pop("model_type", None)
    cfg["tie_word_embeddings"] = True
    return cfg


def qwen_tensors(src: str | Path) -> dict[str, torch.Tensor]:
    """Load only Qwen's text model, keyed as ``Qwen3_5ForCausalLM`` expects."""
    src = Path(src)
    index = json.loads(
        (src / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    out: dict[str, torch.Tensor] = {}
    for filename in sorted(set(index.values())):
        with safe_open(str(src / filename), framework="pt") as handle:
            for key in handle.keys():
                if key.startswith(PREFIX):
                    out["model." + key[len(PREFIX) :]] = handle.get_tensor(key)
    return out


def empty_model(cfg: Qwen3_5TextConfig) -> Qwen3_5ForCausalLM:
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(cfg)
    model = model.to_empty(device="cpu").to(torch.bfloat16)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model


def _code_dims(circuit: InternalLifeCircuit) -> list[int]:
    code = [next(iter(expr.terms)) for expr in circuit.code]
    if len(set(code)) != len(code):
        raise AssertionError("output-code expressions must occupy distinct dimensions")
    return code


def grid_layer(circuit: InternalLifeCircuit) -> int:
    """Return the physical added-layer index containing the authoritative rows."""
    heads = [
        node
        for node in circuit.nodes
        if getattr(node, "name", "").startswith("grid.row0")
    ]
    if len(heads) != 1:
        raise AssertionError(f"expected one grid.row0 head, found {len(heads)}")
    level = int(heads[0].level)
    if level % 2 != 1:
        raise AssertionError(f"grid row is not in an attention sublayer: level {level}")
    return (level - 1) // 2


def compile_plan(intermediate: int = 3584):
    circuit = InternalLifeCircuit()
    code = _code_dims(circuit)
    keep = set(code)
    allocation = allocate(circuit, None, keep, intermediate)
    hidden, ballast = pick_hidden(1024, 1 + allocation.n_slots)
    layout = Layout(hidden=hidden, off=1024, m=ballast, slots=allocation.slots)
    # Force the exact-power-of-two RMS invariant to be checked while planning.
    layout.scale
    return circuit, code, keep, allocation, layout


@torch.no_grad()
def write_circuit_embeddings(
    weight: torch.Tensor,
    circuit: InternalLifeCircuit,
    layout: Layout,
    base_hidden: int,
    *,
    chunk_size: int = 4096,
) -> None:
    """Program added coordinates for every token without touching Qwen coordinates.

    Chunking avoids materializing the roughly 1.8 GB float32 full embedding table.
    ``embedding_rows`` evaluates token-feature functions on *absolute model token
    ids*, which is what lets the front layers recognize ordinary Qwen tokens.
    """
    if layout.off != base_hidden:
        raise ValueError(f"layout offset {layout.off} must equal base hidden {base_hidden}")
    if weight.shape[1] != layout.hidden:
        raise ValueError(f"embedding hidden {weight.shape[1]} != layout {layout.hidden}")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    for start in range(0, weight.shape[0], chunk_size):
        stop = min(start + chunk_size, weight.shape[0])
        rows = embedding_rows(circuit, layout, list(range(start, stop)), layout.hidden)
        weight[start:stop, base_hidden:].copy_(rows[:, base_hidden:].to(weight.dtype))


def merge(src: str | Path) -> tuple[Qwen3_5ForCausalLM, dict]:
    tc = text_config(src)
    n0, d0, v0 = tc["num_hidden_layers"], tc["hidden_size"], tc["vocab_size"]
    if (d0, v0) != (1024, BASE_VOCAB):
        raise ValueError(
            f"expected Qwen3.5-0.8B geometry, got hidden={d0}, vocab={v0}"
        )
    if ROW_BASE < v0 or ROW_BASE % VAL_COUNT:
        raise ValueError(
            f"row base must be >= {v0} and aligned to {VAL_COUNT}: {ROW_BASE}"
        )

    circuit, code, keep, allocation, layout = compile_plan(tc["intermediate_size"])
    hidden, n_internal = layout.hidden, circuit.n_layers()
    vocab = aligned_vocab()
    if vocab != MODEL_VOCAB:
        raise AssertionError(f"merger vocabulary {vocab} != circuit vocabulary {MODEL_VOCAB}")
    native_scale = math.sqrt(d0 / hidden)
    tc.update(
        hidden_size=hidden,
        num_hidden_layers=n_internal + n0,
        vocab_size=vocab,
        layer_types=["linear_attention"] * n_internal + tc["layer_types"],
    )
    tc.pop("full_attention_interval", None)
    model = empty_model(Qwen3_5TextConfig(**tc))
    params = dict(model.named_parameters())
    shift = re.compile(r"^model\.layers\.(\d+)\.")
    norm_re = re.compile(
        r"(input_layernorm|post_attention_layernorm)\.weight$|^model\.norm\.weight$"
    )

    with torch.no_grad():
        for key, tensor in qwen_tensors(src).items():
            shifted = shift.sub(
                lambda match: f"model.layers.{int(match[1]) + n_internal}.", key
            )
            target = params[shifted]
            if norm_re.search(shifted):
                target.fill_(-1)
                target[:d0] = (
                    (1 + tensor.float()) * native_scale - 1
                ).to(target.dtype)
            elif target.shape == tensor.shape:
                target.copy_(tensor)
            else:
                target[tuple(slice(0, size) for size in tensor.shape)] = tensor

        emitted = emit_layers(circuit, layout, tc, 0, "model.", allocation, keep)
        for key, tensor in emitted.items():
            params[key].copy_(tensor.to(torch.bfloat16))

        # Qwen's source rows/base coordinates were copied above. Only the added
        # coordinates are filled here, including tool-token detectors, ballast,
        # and the absolute-id output code.
        write_circuit_embeddings(
            model.model.embed_tokens.weight,
            circuit,
            layout,
            d0,
        )
        final = model.model.norm.weight
        final[d0:] = -1.0
        for dim in code:
            final[layout.dim(dim)] = code_gain(layout)

    model.tie_weights()
    meta = {
        "format": "qwen-life-internal-v1",
        "base": Path(src).name,
        "base_vocab": v0,
        "model_vocab": vocab,
        "row_base": ROW_BASE,
        "reset_token": RESET,
        "tick_token": TICK,
        "done_token": DONE,
        "base_hidden": d0,
        "hidden": hidden,
        "ballast_dims": layout.m,
        "internal_layers": n_internal,
        "grid_layer": grid_layer(circuit),
        "grid_heads": list(range(16)),
        "circuit_dims": len(circuit.dims),
        "residual_slots": allocation.n_slots,
        "output_code_bits": len(code),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "board": [16, 16],
        "max_advance": 15,
        "gradients": False,
        "semantic_host_parser": False,
    }
    return model.eval(), meta


def special_names() -> list[str]:
    names = [f"<|life_row_{value:04x}|>" for value in range(VAL_COUNT)]
    names += ["<|life_reset|>", "<|life_tick|>", "<|life_done|>"]
    if (RESET, TICK, DONE) != (
        ROW_BASE + VAL_COUNT,
        ROW_BASE + VAL_COUNT + 1,
        ROW_BASE + VAL_COUNT + 2,
    ):
        raise AssertionError("internal Life tokens are not contiguous after the row bank")
    return names


def extend_tokenizer(src: str | Path, dst: str | Path) -> None:
    from tokenizers import AddedToken, Tokenizer

    src, dst = Path(src), Path(dst)
    tokenizer = Tokenizer.from_file(str(src / "tokenizer.json"))
    current = tokenizer.get_vocab_size(with_added_tokens=True)
    if current > ROW_BASE:
        raise ValueError(f"source tokenizer has {current} rows, above row base {ROW_BASE}")
    padding = [
        AddedToken(f"<|life_internal_reserved_{index}|>", special=True)
        for index in range(ROW_BASE - current)
    ]
    added = [AddedToken(name, special=True) for name in special_names()]
    tokenizer.add_special_tokens(padding + added)
    if tokenizer.token_to_id(special_names()[0]) != ROW_BASE:
        raise AssertionError("row tokens did not begin at the aligned row base")

    dst.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(dst / "tokenizer.json"))
    for filename in ("chat_template.jinja", "generation_config.json", "LICENSE"):
        if (src / filename).exists():
            shutil.copy(src / filename, dst / filename)
    cfg = json.loads((src / "tokenizer_config.json").read_text(encoding="utf-8"))
    decoder = cfg.setdefault("added_tokens_decoder", {})
    for token_id, token in enumerate(padding + added, current):
        decoder[str(token_id)] = {
            "content": token.content,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        }
    (dst / "tokenizer_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    args = parser.parse_args()
    model, meta = merge(args.src)
    args.dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.dst, max_shard_size="2GB")
    extend_tokenizer(args.src, args.dst)
    (args.dst / "life_internal.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
