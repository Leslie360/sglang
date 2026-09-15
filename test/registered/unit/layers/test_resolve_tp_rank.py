"""Unit tests for the tp_size==1 shard-index clamp in parallel linear layers.

When a layer is explicitly built with tp_size == 1 (a full replica on every
rank), the shard index must clamp to 0 regardless of the global TP rank --
otherwise a rank >= 1 narrows past the end of the replicated weight during
loading. The helper is pure and testable without any accelerator.
"""

from sglang.srt.layers.linear import _resolve_tp_rank
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch


class TestResolveTpRank(unittest.TestCase):
    def test_clamps_to_zero_when_tp_size_is_one(self):
        for global_rank in (0, 1, 3, 7):
            with patch(
                "sglang.srt.layers.linear.get_parallel"
            ) as gp:
                gp.return_value.tp_rank = global_rank
                gp.return_value.tp_size = 8  # global engine is TP>1
                rank, size = _resolve_tp_rank(None, 1)
            self.assertEqual((rank, size), (0, 1))

    def test_passthrough_when_both_given_and_size_gt_one(self):
        rank, size = _resolve_tp_rank(3, 8)
        self.assertEqual((rank, size), (3, 8))

    def test_fills_from_parallel_state_when_missing(self):
        with patch("sglang.srt.layers.linear.get_parallel") as gp:
            gp.return_value.tp_rank = 2
            gp.return_value.tp_size = 8
            rank, size = _resolve_tp_rank(None, None)
        self.assertEqual((rank, size), (2, 8))

    def test_explicit_rank_kept_when_size_gt_one(self):
        rank, size = _resolve_tp_rank(5, 8)
        self.assertEqual((rank, size), (5, 8))


if __name__ == "__main__":
    unittest.main()
