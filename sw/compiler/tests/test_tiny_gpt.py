"""A small random GPT, compiled and run: the program must match the reference forward pass bit for bit."""
import unittest

import numpy as np

from mini_tpu import codegen, model
from mini_tpu.isa import C
from mini_tpu.sim import Machine

TINY = model.GPTConfig(n_layer=2, d_model=64, n_head=2, d_ff=256, vocab=250, seq=32)
BASES = (0x1000, 0x40000, 0x80000, 0xB0000, 0xC0000, 0xE0000)   # program, regions 0-3, KV cache: below 0xF0000, where tb_blob's DDR model returns AXI errors


def build(cfg=TINY, seed=0):
    rng = np.random.default_rng(seed)
    w = model.random_gpt(cfg, seed)
    calib = [rng.integers(0, cfg.vocab, cfg.seq) for _ in range(8)]
    q = model.quantize(cfg, w, model.calibrate_max(cfg, w, calib))
    ids = rng.integers(0, cfg.vocab, cfg.seq)
    return w, q, ids, codegen.compile_gpt(q)


def simulate(q, ids, c, **kw):
    m = Machine(ddr_bytes=1 << 20, **kw)
    status = codegen.run(m, c, q.embed(ids), BASES)
    out = np.frombuffer(m.read_ddr(BASES[4], c.output_bytes), dtype="<i4").reshape(q.cfg.seq, q.cfg.vocab_pad)
    return m, status, out


class TinyGPT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w, cls.q, cls.ids, cls.c = build()
        cls.m, cls.status, cls.out = simulate(cls.q, cls.ids, cls.c)
        cls.ref = model.forward(cls.q, cls.q.embed(cls.ids))

    def test_runs(self):
        self.assertEqual(self.status, C.ST_DONE, self.m.message)

    def test_fits_the_rtl_ddr(self):
        c = self.c
        for start, size, end in ((BASES[0], len(c.program), BASES[1]), (BASES[1], len(c.weights), BASES[2]),
                                 (BASES[2], len(c.params), BASES[3]), (BASES[3], c.input_bytes, BASES[4]),
                                 (BASES[4], c.output_bytes, BASES[5]), (BASES[5], c.kv_bytes, 0xF0000)):
            self.assertLessEqual(start + size, end)

    def test_no_races(self):
        k = self.m.checker
        self.assertEqual(k.races, 0, [str(f) for f in k.findings[:5]])
        self.assertEqual(k.rule_violations, 0, [str(f) for f in k.findings[:5]])
        self.assertEqual(k.count.get("UNINIT", 0), 0, [str(f) for f in k.findings[:5]])

    def test_matches_reference_bit_for_bit(self):
        np.testing.assert_array_equal(self.out, self.ref)

    def test_any_execution_order(self):
        for seed in range(3):
            np.testing.assert_array_equal(simulate(self.q, self.ids, self.c, order_seed=seed)[2], self.ref)

    def test_tracks_the_float_model(self):
        fl = model.float_forward(self.q.cfg, self.w, self.ids)
        qt = self.q.logits(self.ref.astype(np.float64))
        corr = np.corrcoef(fl.ravel(), qt.ravel())[0, 1]
        top1 = (fl.argmax(-1) == qt.argmax(-1)).mean()
        print(f"tiny GPT: {self.c.descriptors} descriptors, {self.c.vmem_used} B vmem, logit correlation {corr:.4f}, top-1 match {top1:.2f}")
        self.assertGreater(corr, 0.9)

    def test_k_split_shapes(self):
        """d_ff > 512 and a head dimension of 64, as in GPT-2."""
        cfg = model.GPTConfig(n_layer=1, d_model=128, n_head=2, d_ff=640, vocab=64, seq=16)
        _, q, ids, c = build(cfg, seed=1)
        m, status, out = simulate(q, ids, c)
        self.assertEqual(status, C.ST_DONE, m.message)
        self.assertEqual(m.checker.races + m.checker.rule_violations, 0)
        np.testing.assert_array_equal(out, model.forward(q, q.embed(ids)))


if __name__ == "__main__":
    unittest.main()
