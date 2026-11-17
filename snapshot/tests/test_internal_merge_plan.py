import torch

from life.internal_merge import (
    BASE_VOCAB,
    aligned_vocab,
    compile_plan,
    grid_layer,
    special_names,
    write_circuit_embeddings,
)
from prototype.internal_circuit import (
    DONE,
    MODEL_VOCAB,
    RESET,
    ROW_BASE,
    TICK,
    VAL_COUNT,
    InternalLifeCircuit,
)


def test_internal_token_layout_is_aligned_and_contiguous():
    assert ROW_BASE >= BASE_VOCAB
    assert ROW_BASE % VAL_COUNT == 0
    assert (RESET, TICK, DONE) == (
        ROW_BASE + VAL_COUNT,
        ROW_BASE + VAL_COUNT + 1,
        ROW_BASE + VAL_COUNT + 2,
    )
    assert aligned_vocab() % 64 == 0
    assert aligned_vocab() > DONE
    assert aligned_vocab() == MODEL_VOCAB
    names = special_names()
    assert len(names) == VAL_COUNT + 3
    assert names[0] == "<|life_row_0000|>"
    assert names[VAL_COUNT - 1] == "<|life_row_ffff|>"


def test_internal_merge_plan_fits_qwen35_08b():
    circuit, code, keep, allocation, layout = compile_plan()
    assert isinstance(circuit, InternalLifeCircuit)
    assert circuit.n_layers() >= 3
    assert len(code) >= 19
    assert keep == set(code)
    assert allocation.n_slots > 0
    assert layout.off == 1024
    assert layout.hidden > layout.off
    assert 0 <= grid_layer(circuit) < circuit.n_layers()


def test_embedding_programmer_never_touches_base_coordinates():
    circuit, _, _, _, layout = compile_plan()
    # A tiny prefix is sufficient to prove the write boundary; the function is
    # identical for later chunks and token features see absolute ids.
    weight = torch.randn(8, layout.hidden, dtype=torch.bfloat16)
    before = weight[:, :1024].clone()
    write_circuit_embeddings(weight, circuit, layout, 1024, chunk_size=3)
    assert torch.equal(weight[:, :1024], before)
    assert torch.count_nonzero(weight[:, 1024:]) > 0
