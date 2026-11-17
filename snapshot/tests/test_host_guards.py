import pytest

from life.host import Feeder
from prototype.circuit import VOCAB_SIZE


class _NeverCalledModel:
    device = "cpu"

    def __call__(self, *args, **kwargs):  # pragma: no cover - a failed guard would call it
        raise AssertionError("invalid input reached the model")


def test_feed_rejects_tokens_outside_the_life_vocabulary():
    feeder = Feeder(_NeverCalledModel())
    with pytest.raises(ValueError, match="outside"):
        feeder.feed(-1)
    with pytest.raises(ValueError, match="outside"):
        feeder.feed(VOCAB_SIZE)


def test_reset_requires_exactly_sixteen_unsigned_rows():
    feeder = Feeder(_NeverCalledModel())
    with pytest.raises(ValueError, match="sixteen rows"):
        feeder.reset([0] * 15)
    with pytest.raises(ValueError, match="16-bit"):
        feeder.reset([0] * 15 + [1 << 16])


def test_generation_requires_reset_or_a_restored_cache():
    feeder = Feeder(_NeverCalledModel())
    with pytest.raises(RuntimeError, match="RESET"):
        feeder.generation()
