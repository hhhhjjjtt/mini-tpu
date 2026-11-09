"""Decode with the KV cache: a prompt through prefill, then one token at a time through the decode
programs, must give the reference forward pass's logits for every position, bit for bit."""
import unittest

import numpy as np

from mini_tpu import codegen, model
from mini_tpu.codegen import Target
from mini_tpu.session import Session

TINY = model.GPTConfig(n_layer=2, d_model=64, n_head=2, d_ff=256, vocab=250, seq=32)
CTX = 64                                    # buckets 16, 32, 64


def build(cfg=TINY, seed=0, ctx=CTX, target=None):
    rng = np.random.default_rng(seed)
    w = model.random_gpt(cfg, seed, ctx)
    calib = [rng.integers(0, cfg.vocab, cfg.seq) for _ in range(8)]
    q = model.quantize(cfg, w, model.calibrate_max(cfg, w, calib))
    ids = rng.integers(0, cfg.vocab, ctx)
    return q, ids, codegen.compile_model(q, target)


def generate(s, ids, n, last):
    """Prompt ids[:n], then ids[n], ... ids[last] one at a time. Returns logits by position."""
    out = {n - 1: s.prefill(ids[:n])}
    for p in range(n, last + 1):
        out[p] = s.step(ids[p], p)
    return out


class Decode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.q, cls.ids, cls.c = build()
        cls.ref = model.forward(cls.q, cls.q.embed(cls.ids))

    def check(self, s, out):
        for p, row in out.items():
            np.testing.assert_array_equal(row, self.ref[p], err_msg=f"position {p}")
        for name, descs, races, rules, uninit in s.runs:
            self.assertEqual((races, rules, uninit), (0, 0, 0), f"{name}: {[str(f) for f in s.m.checker.findings[:5]]}")

    def test_buckets(self):
        self.assertEqual(codegen.buckets(64), [16, 32, 64])
        self.assertEqual(codegen.buckets(48), [16, 32, 48])
        self.assertEqual(codegen.buckets(1024), [16, 32, 64, 128, 256, 512, 1024])
        self.assertEqual(list(self.c.decode), [16, 32, 64])

    def test_short_prompt_to_full_context(self):
        """Prompt of 5, padded to S = 32: the cache holds padding until decode overwrites it."""
        s = Session.of(self.q, self.c)
        self.check(s, generate(s, self.ids, 5, CTX - 1))
        print(f"tiny decode: {len(s.runs)} runs, descriptors per bucket "
              f"{ {L: c.descriptors for L, c in self.c.decode.items()} }, vmem {max(c.vmem_used for c in self.c.decode.values())} B")

    def test_full_prompt(self):
        s = Session.of(self.q, self.c)
        self.check(s, generate(s, self.ids, TINY.seq, TINY.seq + 3))

    def test_any_execution_order(self):
        for seed in range(2):
            s = Session.of(self.q, self.c, order_seed=seed)
            self.check(s, generate(s, self.ids, 14, 17))

    def test_long_context_k_split(self):
        """max_k 32: P.V of a 64-position bucket accumulates over two weight halves."""
        q, ids, c = build(seed=2, target=Target(max_k=32))
        s = Session.of(q, c)
        out = generate(s, ids, 30, CTX - 1)
        ref = model.forward(q, q.embed(ids))
        for p, row in out.items():
            np.testing.assert_array_equal(row, ref[p], err_msg=f"position {p}")
        self.assertEqual(sum(r[2] + r[3] for r in s.runs), 0)

    def test_shares_weights(self):
        """Decode adds no weights: every tile is already in prefill's."""
        alone = codegen.compile_gpt(self.q)
        self.assertEqual(len(self.c.prefill.weights), len(alone.weights))


if __name__ == "__main__":
    unittest.main()
