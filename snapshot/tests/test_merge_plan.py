from life.merge import V_QWEN, compile_plan
from prototype.circuit import LifeCircuit, VOCAB_SIZE


def test_merge_plan_fits_qwen35_08b():
    circuit, code, keep, allocation, layout = compile_plan()
    assert isinstance(circuit, LifeCircuit)
    assert circuit.n_layers() == 3
    assert len(code) == 17
    assert keep == set(code)
    assert allocation.n_slots == 310
    assert (layout.hidden, layout.m, layout.off) == (1408, 22, 1024)
    assert V_QWEN + VOCAB_SIZE < 314_000

