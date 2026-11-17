"""Decode the authoritative 16x16 grid directly from DeltaNet state."""

from __future__ import annotations

import torch


def _grid_state(cache):
    layer = cache.layers[1]  # packed-row circuit: layer 1, one value head per row
    states = getattr(layer, "recurrent_states", None)
    if not states or states[0] is None:
        return None
    return states[0][0]


def read(cache) -> tuple[int, ...]:
    """Return sixteen unsigned rows. No token log or Life rule is consulted."""
    state = _grid_state(cache)
    if state is None:
        return (0,) * 16
    rows = []
    for y in range(16):
        row = state[y, 0]  # head y, constant key row zero, value columns
        rows.append(sum(int(row[2 + x] > 0) << x for x in range(16)))
    return tuple(rows)


def snapshot(cache) -> dict:
    state = _grid_state(cache)
    if state is None:
        raise ValueError("cache has no initialized Life state")
    return {"layer": 1, "heads": list(range(16)), "states": state[:16].detach().clone().cpu()}


def restore(model, saved: dict, batch: int = 1):
    """Restore only the grid heads into a fresh stock Transformers cache."""
    from transformers import DynamicCache

    cfg = model.config.get_text_config(decoder=True)
    cache = DynamicCache(config=model.config)
    device = next(model.parameters()).device
    layer = cache.layers[int(saved["layer"])]
    # Qwen's reference Gated DeltaNet kernels accumulate recurrent state in
    # FP32 even when model weights and convolution state are BF16. Never narrow
    # that state to ``model.dtype``. The promotion also repairs legacy BF16
    # snapshots created by the pre-fix runtime while preserving FP32/FP64.
    recurrent_dtype = torch.promote_types(saved["states"].dtype, torch.float32)
    rec = torch.zeros(
        batch,
        cfg.linear_num_value_heads,
        cfg.linear_key_head_dim,
        cfg.linear_value_head_dim,
        dtype=recurrent_dtype,
        device=device,
    )
    for head, state in zip(saved["heads"], saved["states"]):
        rec[:, head] = state.to(device)
    conv_dim = (
        2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim
        + cfg.linear_num_value_heads * cfg.linear_value_head_dim
    )
    layer.recurrent_states[0] = rec
    layer.conv_states[0] = torch.zeros(
        batch,
        conv_dim,
        cfg.linear_conv_kernel_dim,
        dtype=model.dtype,
        device=device,
    )
    layer.conv_kernel_size[0] = cfg.linear_conv_kernel_dim
    layer.is_conv_states_initialized[0] = True
    layer.is_recurrent_states_initialized[0] = True
    layer.has_previous_state[0] = True
    layer.dtype, layer.device = model.dtype, device
    return cache


def save(saved: dict, path):
    torch.save(saved, path)


def load(path) -> dict:
    return torch.load(path, weights_only=True)

