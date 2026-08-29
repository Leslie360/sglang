"""Unit tests for the ReplaySSM ring memory-budget charge.

The per-slot ring buffers are allocated for BOTH --enable-linear-replayssm
(buffered decode) and --enable-linear-replayssm-spec (fold-every-commit), but
the memory budget used to charge only the spec flag -- a non-spec
--enable-linear-replayssm run allocated the ring on top of the budgeted pools,
silently over-provisioning GPU memory (~2.9GB/GPU on Qwen3.8-27B TP2). The
helper must charge the ring for either flag.

Note: test_replayssm_ring_accounting.py already pins the low-level byte
arithmetic of replayssm_ring_bytes_per_req(). This file guards the higher-level
budget decision in KVCacheConfigurator._replayssm_ring_bytes_per_req(), i.e.
which flag combinations trigger a charge and how the record length is chosen.
"""

import unittest

from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestReplayssmRingBudget(CustomTestCase):
    # Identity ring_bytes_per_req: the returned value equals the selected
    # record_len, so assertions also pin the ring sizing.
    _charge = staticmethod(
        lambda **kw: KVCacheConfigurator._replayssm_ring_bytes_per_req(
            **kw, ring_bytes_per_req=lambda record_len: record_len
        )
    )

    def _base(self, **kw):
        base = dict(
            enable_linear_replayssm=False,
            enable_linear_replayssm_spec=False,
            is_gdn=False,
            is_kda=False,
            linear_replayssm_cache_len=16,
            max_draft_tokens=None,
        )
        base.update(kw)
        return base

    def test_non_spec_replayssm_charges_ring(self):
        # Regression: the non-spec flag used to return 0 (ring uncharged).
        value = self._charge(**self._base(enable_linear_replayssm=True, is_gdn=True))
        self.assertGreater(value, 0)
        self.assertEqual(value, 16)  # non-spec ring is L-sized

    def test_spec_gdn_charges_draft_sized_ring(self):
        value = self._charge(
            **self._base(
                enable_linear_replayssm_spec=True, is_gdn=True, max_draft_tokens=8
            )
        )
        self.assertEqual(value, 8)

    def test_spec_kda_charges_l_sized_ring(self):
        value = self._charge(
            **self._base(
                enable_linear_replayssm_spec=True, is_kda=True, max_draft_tokens=8
            )
        )
        self.assertEqual(value, 16)

    def test_off_returns_zero(self):
        self.assertEqual(self._charge(**self._base()), 0)

    def test_non_replayssm_model_returns_zero(self):
        value = self._charge(**self._base(enable_linear_replayssm_spec=True))
        self.assertEqual(value, 0)


if __name__ == "__main__":
    unittest.main()
