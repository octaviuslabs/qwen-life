"""Merge the exact-tested packed-row Life circuit into pristine Qwen3.5-0.8B.

Every added tensor is written analytically. Native tensors are copied into wider zero-padded
matrices; there is no optimizer, dataset, fitting, or gradient.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from chipformer.q35.emit import Layout, allocate, embedding_rows, emit_layers
from chipformer.q35.merge import pick_hidden
from chipformer.q35.tiny import code_gain
from prototype.circuit import DONE, RESET, TICK, VAL_COUNT, LifeCircuit, VOCAB, VOCAB_SIZE

PREFIX = "model.language_model."
V_QWEN = 248_320


def text_config(src: str | Path) -> dict:
    cfg = json.loads((Path(src) / "config.json").read_text())["text_config"]
    cfg.pop("model_type", None)
    cfg["tie_word_embeddings"] = True
    return cfg


def qwen_tensors(src: str | Path) -> dict[str, torch.Tensor]:
    src = Path(src)
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for filename in sorted(set(index.values())):
        with safe_open(str(src / filename), framework="pt") as sf:
            for key in sf.keys():
                if key.startswith(PREFIX):
                    out["model." + key[len(PREFIX):]] = sf.get_tensor(key)
    return out


def empty_model(cfg: Qwen3_5TextConfig) -> Qwen3_5ForCausalLM:
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(cfg)
    model = model.to_empty(device="cpu").to(torch.bfloat16)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model


def compile_plan(intermediate: int = 3584):
    circuit = LifeCircuit()
    code = [next(iter(expr.terms)) for expr in circuit.code]
    keep = set(code)
    allocation = allocate(circuit, None, keep, intermediate)
    hidden, ballast = pick_hidden(1024, 1 + allocation.n_slots)
    layout = Layout(hidden=hidden, off=1024, m=ballast, slots=allocation.slots)
    layout.scale
    return circuit, code, keep, allocation, layout


def merge(src: str | Path) -> tuple[Qwen3_5ForCausalLM, dict]:
    tc = text_config(src)
    n0, d0, v0 = tc["num_hidden_layers"], tc["hidden_size"], tc["vocab_size"]
    if (d0, v0) != (1024, V_QWEN):
        raise ValueError(f"expected Qwen3.5-0.8B geometry, got hidden={d0}, vocab={v0}")
    circuit, code, keep, allocation, layout = compile_plan(tc["intermediate_size"])
    hidden, n_life = layout.hidden, circuit.n_layers()
    vocab = -(-(v0 + VOCAB_SIZE) // 64) * 64
    native_scale = math.sqrt(d0 / hidden)
    tc.update(
        hidden_size=hidden,
        num_hidden_layers=n_life + n0,
        vocab_size=vocab,
        layer_types=["linear_attention"] * n_life + tc["layer_types"],
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
            shifted = shift.sub(lambda m: f"model.layers.{int(m[1]) + n_life}.", key)
            target = params[shifted]
            if norm_re.search(shifted):
                target.fill_(-1)
                target[:d0] = ((1 + tensor.float()) * native_scale - 1).to(target.dtype)
            elif target.shape == tensor.shape:
                target.copy_(tensor)
            else:
                target[tuple(slice(0, size) for size in tensor.shape)] = tensor

        emitted = emit_layers(circuit, layout, tc, 0, "model.", allocation, keep)
        for key, tensor in emitted.items():
            params[key].copy_(tensor.to(torch.bfloat16))

        rows = embedding_rows(circuit, layout, VOCAB, hidden).to(torch.bfloat16)
        model.model.embed_tokens.weight[v0 : v0 + VOCAB_SIZE] = rows
        final = model.model.norm.weight
        final[d0:] = -1.0
        for dim in code:
            final[layout.dim(dim)] = code_gain(layout)

    model.tie_weights()
    meta = {
        "base": Path(src).name,
        "base_hidden": d0,
        "hidden": hidden,
        "ballast_dims": layout.m,
        "life_layers": n_life,
        "circuit_dims": len(circuit.dims),
        "residual_slots": allocation.n_slots,
        "vocab_offset": v0,
        "life_vocab": VOCAB_SIZE,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "board": [16, 16],
        "generated_tokens_per_generation": 17,
        "gradients": False,
    }
    return model.eval(), meta


def special_names() -> list[str]:
    names = [f"<|life_row_{token:04x}|>" for token in range(VAL_COUNT)]
    names += ["<|life_reset|>", "<|life_tick|>", "<|life_done|>"]
    assert (RESET, TICK, DONE) == (VAL_COUNT, VAL_COUNT + 1, VAL_COUNT + 2)
    assert len(names) == VOCAB_SIZE
    return names


def extend_tokenizer(src: str | Path, dst: str | Path):
    from tokenizers import AddedToken, Tokenizer

    src, dst = Path(src), Path(dst)
    tokenizer = Tokenizer.from_file(str(src / "tokenizer.json"))
    current = tokenizer.get_vocab_size(with_added_tokens=True)
    if current > V_QWEN:
        raise ValueError(f"source tokenizer has {current} rows, above configured {V_QWEN}")
    padding = [AddedToken(f"<|life_reserved_{i}|>", special=True) for i in range(V_QWEN - current)]
    added = [AddedToken(name, special=True) for name in special_names()]
    tokenizer.add_special_tokens(padding + added)
    if tokenizer.token_to_id(special_names()[0]) != V_QWEN:
        raise AssertionError("Life tokens are not contiguous")
    tokenizer.save(str(dst / "tokenizer.json"))
    for filename in ("chat_template.jinja", "generation_config.json"):
        if (src / filename).exists():
            shutil.copy(src / filename, dst / filename)
    cfg = json.loads((src / "tokenizer_config.json").read_text())
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
    (dst / "tokenizer_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    args = parser.parse_args()
    model, meta = merge(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dst, max_shard_size="2GB")
    extend_tokenizer(args.src, dst)
    (dst / "life.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
