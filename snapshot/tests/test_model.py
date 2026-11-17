import os

import pytest
import torch

from life.host import Feeder
from life.monitor import read, restore, snapshot
from life.reference import INITIAL_ROWS, step
from prototype.tiny_check import build_model, run


def _require_model_runtime():
    if not torch.cuda.is_available() and not os.environ.get("LIFE_ALLOW_CPU_MODEL"):
        pytest.skip("run in the CUDA dev container or set LIFE_ALLOW_CPU_MODEL=1")


@pytest.mark.model
def test_real_qwen_modules_are_exact_and_recurrent():
    _require_model_runtime()
    result = run(generations=3)
    assert result["status"] == "exact"
    assert result["forced_rows_and_done"] == 68


@pytest.mark.model
def test_legacy_bfloat16_snapshot_is_promoted_and_continues_exactly():
    """A pre-fix BF16 save should self-heal when loaded by the new runtime."""
    _require_model_runtime()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, model = build_model(device)
    feeder = Feeder(model)
    feeder.reset(INITIAL_ROWS)

    legacy = snapshot(feeder.cache)
    legacy["states"] = legacy["states"].to(torch.bfloat16)
    restored_cache = restore(model, legacy)
    assert snapshot(restored_cache)["states"].dtype == torch.float32

    fresh = Feeder(model, cache=restored_cache)
    expected = step(tuple(INITIAL_ROWS))
    assert fresh.generation() == expected
    assert read(fresh.cache) == expected

