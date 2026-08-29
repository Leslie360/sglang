"""DFLASH speculative decoding under prefill/decode (PD) disaggregation.

This test exercises the DFLASH disagg wiring in
`SpeculativeAlgorithm.build_disagg_draft_input` (the `is_dflash()` branch that
routes to `build_dflash_family_disagg_draft_input`). It launches a 1P1D
prefill/decode pair with DFLASH enabled on both sides, then checks:

1.  GSM8K accuracy is preserved (the draft does not corrupt outputs).
2.  Speculation is actually active: the decode server's `spec_accept_length`
    metric is > 1 (a silent fallback to non-speculative AR would pin it at 1).

Uses the small Llama-3.1-8B target + its DFlash draft so CI can run it on
2 GPUs (one prefill, one decode).
"""

import time
import unittest
from types import SimpleNamespace

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_DFLASH,
    DEFAULT_TARGET_MODEL_DFLASH,
)

register_cuda_ci(est_time=500, stage="base-b", runner_config="2-gpu-large")


class TestDisaggregationDFlash(PDDisaggregationServerBase):
    # The DFlash draft's max_position_embeddings (40960) is shorter than the
    # Llama-3.1-8B target's (131072); allow the shorter draft context.
    extra_prefill_env = {"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1"}
    extra_decode_env = {"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1"}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_TARGET_MODEL_DFLASH
        spec_args = [
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-draft-model-path",
            DEFAULT_DRAFT_MODEL_DFLASH,
            "--speculative-dflash-block-size",
            "8",
            "--attention-backend",
            "triton",
            "--speculative-draft-attention-backend",
            "triton",
            "--cuda-graph-max-bs-decode",
            "8",
            "--mem-fraction-static",
            "0.7",
            "--enable-metrics",
        ]
        cls.extra_prefill_args = spec_args
        cls.extra_decode_args = spec_args
        cls.launch_all()

    def _decode_accept_length(self) -> float:
        """Query the decode server's spec_accept_length gauge."""
        resp = requests.get(self.decode_url + "/metrics", timeout=10)
        resp.raise_for_status()
        for line in resp.text.splitlines():
            if line.startswith("sglang:spec_accept_length{"):
                return float(line.split()[-1])
        return 0.0

    def _generate(self, prompt: str, max_tokens: int = 64) -> str:
        resp = requests.post(
            self.lb_url + "/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=f"http://{self.base_host}:{self.lb_port}",
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"Evaluation metrics: {metrics}")

        self.assertGreater(metrics["score"], 0.74)

    def test_speculation_active(self):
        """DFLASH must actually speculate under PD (accept_length > 1), not
        silently fall back to autoregressive decoding."""
        # Drive enough decode traffic for the accept-length gauge to accumulate.
        for i in range(8):
            out = self._generate(f"Write a short sentence about topic {i}.")
            self.assertGreater(len(out), 0)
            time.sleep(0.5)
        accept_len = self._decode_accept_length()
        print(f"Decode spec_accept_length: {accept_len}")

        self.assertGreater(
            accept_len,
            1.0,
            "DFLASH should accept > 1 token per verify step under PD; "
            "a value of 1.0 indicates speculation silently fell back to AR.",
        )


if __name__ == "__main__":
    unittest.main()
