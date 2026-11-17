"""Optional integration test for Qwen3.5's actual recurrent implementation.

The symbolic suite is the fast default. Set RUN_QWEN_MODEL_TESTS=1 to allocate
the 94M-parameter tiny controller and exercise Qwen's real cache one token at a
time. CI or a local GPU can select QWEN_LIFE_DEVICE=cuda.
"""

from __future__ import annotations

import os
import unittest


@unittest.skipUnless(
    os.environ.get("RUN_QWEN_MODEL_TESTS") == "1",
    "set RUN_QWEN_MODEL_TESTS=1 for the real Qwen3.5 recurrent check",
)
class TinyQwenRecurrentTests(unittest.TestCase):
    def test_native_cache_feedback_is_exact(self) -> None:
        from prototype.tiny_check import run

        generations = int(os.environ.get("QWEN_LIFE_GENERATIONS", "2"))
        device = os.environ.get("QWEN_LIFE_DEVICE", "auto")
        result = run(generations, None if device == "auto" else device)
        # run() also checks one generation after restoring the recurrent grid
        # into a fresh cache.
        self.assertEqual(result["forced_rows_and_done"], (generations + 1) * 17)
        self.assertEqual(result["status"], "exact")


if __name__ == "__main__":
    unittest.main()
