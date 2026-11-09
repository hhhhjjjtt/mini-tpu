"""Simulator behaviour the RTL comparison cannot show: the race checker must find real races."""
import unittest

import numpy as np

from mini_tpu import isa
from mini_tpu.isa import C
from mini_tpu.sim import Machine

R0, R1, R2, R3 = 0x10000, 0x20000, 0x30000, 0x40000
PROG = 0x1000
LOAD, MATRIX, VECTOR, STORE = range(4)


def machine(descs, **kw):
    m = Machine(**kw)
    m.region[:4] = [R0, R1, R2, R3]
    rng = np.random.default_rng(0)
    m.write_ddr(R0, rng.integers(0, 256, 0x10000, dtype=np.uint8).tobytes())
    m.write_ddr(R2, rng.integers(0, 256, 0x10000, dtype=np.uint8).tobytes())
    m.write_ddr(PROG, isa.build_program(descs))
    return m


def gemm_program(wait_act=True, wait_store=True):
    """LOAD A, LOAD_WGT, GEMM, STORE_I32 epilogue, STORE."""
    g = isa.gemm(0, 64, 1, 64, 0)
    if wait_act:
        isa.wait(g, LOAD, 0)
    s = isa.store(3, 0, 0x40000, 16, 64, 64, 64)
    if wait_store:
        isa.wait(s, VECTOR, 0)
    return [isa.load(2, 0, 0, 16, 64, 64, 64), isa.load_wgt(0, 0, 0, 64), g,
            isa.epilogue(C.EP_STORE_I32, 0, 0, 0, 0, 16, 64, 0x40000, 0), s, isa.halt()]


class Races(unittest.TestCase):
    def run_ok(self, descs):
        m = machine(descs)
        self.assertEqual(m.run(PROG), C.ST_DONE, m.message)
        return m.checker

    def test_clean_program(self):
        c = self.run_ok(gemm_program())
        self.assertEqual((c.races, c.rule_violations), (0, 0), [str(f) for f in c.findings])

    def test_raw_on_activations_and_weights(self):
        c = self.run_ok(gemm_program(wait_act=False))
        self.assertEqual({f.space for f in c.findings if f.kind == "RAW"}, {"vmem", "weights"})

    def test_raw_vector_to_store(self):
        c = self.run_ok(gemm_program(wait_store=False))
        self.assertEqual([(f.kind, f.later) for f in c.findings], [("RAW", (STORE, 1))])

    def test_fence_orders_everything(self):
        d = gemm_program(wait_act=False, wait_store=False)
        d.insert(2, isa.fence())
        d.insert(5, isa.fence())
        self.assertEqual(self.run_ok(d).races, 0)

    def test_war_on_weight_half(self):
        d = gemm_program()[:4] + [isa.load_wgt(0, 0, 0, 64), isa.halt()]
        self.assertEqual([f.kind for f in self.run_ok(d).findings], ["WAR"])
        d[4] = isa.wait(isa.load_wgt(0, 0, 0, 64), MATRIX, 0)
        self.assertEqual(self.run_ok(d).races, 0)

    def test_waw_two_units(self):
        d = [isa.load(2, 0, 0, 1, 16, 16, 16), isa.load(2, 0, 0x100, 1, 64, 64, 64), isa.fence(),
             isa.eltwise(C.EW_PASS_A, 0, 0, 0, 0, 1, 16, 0, 0, 0x200, 16, 16, 16, 0x100),
             isa.load(2, 0, 0x200, 1, 16, 16, 16), isa.halt()]
        self.assertEqual([f.kind for f in self.run_ok(d).findings], ["WAW"])

    def test_raw_through_ddr(self):
        d = [isa.load(2, 0, 0, 1, 16, 16, 16), isa.wait(isa.store(3, 0x100, 0, 1, 16, 16, 16), LOAD, 0),
             isa.load(3, 0x100, 0x100, 1, 16, 16, 16), isa.halt()]
        c = self.run_ok(d)
        self.assertEqual([(f.kind, f.space) for f in c.findings], [("RAW", "ddr")])
        d[2] = isa.wait(isa.load(3, 0x100, 0x100, 1, 16, 16, 16), STORE, 0)
        self.assertEqual(self.run_ok(d).races, 0)

    def test_strided_stores_do_not_overlap(self):
        """Interleaved column blocks share a bounding box but no bytes."""
        d = [isa.load(2, 0, 0, 4, 16, 64, 16), isa.wait(isa.store(3, 0, 0, 4, 16, 64, 16), LOAD, 0),
             isa.store(3, 16, 0, 4, 16, 64, 16), isa.halt()]
        self.assertEqual(self.run_ok(d).races, 0)

    def test_in_place_accumulation(self):
        """dst_addr may equal partial_addr (§5.5)."""
        d = gemm_program()
        d[4:4] = [isa.gemm(0, 64, 1, 64, 0), isa.epilogue(C.EP_ACC_I32, 0, 0, 0, 0, 16, 64, 0x40000, 0, 0x40000, 64)]
        self.assertEqual(self.run_ok(d).races, 0)

    def test_distance_is_loop_invariant(self):
        body = [isa.wait(isa.load(2, 0, 0, 1, 16, 16, 16), STORE, 0), isa.wait(isa.store(3, 0, 0, 1, 16, 16, 16), LOAD, 0)]
        self.assertEqual(self.run_ok([isa.loop(5, 2)] + body + [isa.halt()]).races, 0)

    def test_uninitialized_read(self):
        c = self.run_ok([isa.store(3, 0, 0x1000, 1, 16, 16, 16), isa.halt()])
        self.assertEqual(c.count, {"UNINIT": 1})


class NewOps(unittest.TestCase):
    def test_layernorm_needs_its_input_loaded(self):
        ln = isa.layernorm(False, 1, 16, 16, 0, 0x100, 16, 1 << 30, 34, 1, 1 << 30, 30, 0)
        d = [isa.load(2, 0, 0, 1, 16, 16, 16), ln, isa.halt()]
        m = machine(d)
        self.assertEqual(m.run(PROG), C.ST_DONE)
        self.assertEqual(m.checker.races, 1)
        d[1] = isa.wait(bytearray(ln), LOAD, 0)
        m = machine(d)
        m.run(PROG)
        self.assertEqual(m.checker.races, 0)

    def test_lut_reads_its_table(self):
        lut = isa.vec_lut(1, 16, 0, 0x200, 16, 16, 0x100)
        d = [isa.load(2, 0, 0, 1, 16, 16, 16), isa.load(2, 0, 0x100, 1, 256, 256, 256), isa.wait(lut, LOAD, 1), isa.halt()]
        m = machine(d)
        m.run(PROG)
        self.assertEqual([(f.kind, f.where) for f in m.checker.findings], [("RAW", 0x100)])

    def test_bad_fields(self):
        for d in (isa.layernorm(False, 1, 16, 16, 0, 0x100, 16, 1 << 30, 34, 0, 1 << 30, 30, 0),      # eps = 0
                  isa.layernorm(False, 1, 16, 16, 0, 0x100, 16, 1 << 30, 4, 1, 1 << 30, 30, 0),       # inv_shift < 8
                  isa.layernorm(False, 1, 2048, 2048, 0, 0x100, 2048, 1 << 30, 34, 1, 1 << 30, 30, 0),
                  isa.vec_lut(1, 8, 0, 0x200, 16, 16, 0x100)):
            self.assertEqual(machine([d, isa.halt()]).run(PROG), C.ST_ERROR | C.ERR_BAD_FIELD << 8)


class Rules(unittest.TestCase):
    def test_gemm_without_epilogue_deadlocks(self):
        m = machine(gemm_program()[:3] + [isa.halt()])
        self.assertEqual(m.run(PROG), C.ST_ERROR | C.ERR_WATCHDOG << 8)
        self.assertIn("no command can run", m.message)

    def test_vector_op_between_gemm_and_epilogue(self):
        d = gemm_program()
        d.insert(3, isa.eltwise(C.EW_PASS_A, 0, 0, 0, 0, 1, 16, 0, 0, 0x200, 16, 16, 16, 0))
        m = machine(d)
        m.run(PROG)
        self.assertGreaterEqual(m.checker.rule_violations, 1)

    def test_wait_targets_only_look_backward(self):
        """Mutual waits cannot hang: a target only counts commands issued earlier."""
        d = [isa.wait(isa.store(3, 0, 0, 1, 16, 16, 16), VECTOR, 0),
             isa.wait(isa.eltwise(C.EW_PASS_A, 0, 0, 0, 0, 1, 16, 0, 0, 0x200, 16, 16, 16, 0), STORE, 0), isa.halt()]
        self.assertEqual(machine(d).run(PROG), C.ST_DONE)

    def test_stream_mismatch(self):
        d = gemm_program()
        d[3] = isa.epilogue(C.EP_STORE_I32, 0, 0, 0, 0, 32, 64, 0x40000, 0)
        self.assertEqual(machine(d).run(PROG), C.ST_ERROR | C.ERR_STREAM_MISMATCH << 8)


class Errors(unittest.TestCase):
    def test_bad_magic(self):
        m = machine([isa.halt()])
        m.write_ddr(PROG, b"\0\0\0\0")
        self.assertEqual(m.run(PROG), C.ST_ERROR | C.ERR_BAD_MAGIC << 8)

    def test_region_unset(self):
        m = machine([isa.halt()])
        m.region[1] = 0
        self.assertEqual(m.run(PROG), C.ST_ERROR | C.ERR_REGION_UNSET << 8)
        self.assertEqual(m.err_info0, 0b10)

    def test_vmem_range(self):
        d = [isa.load(2, 0, (2 << 20) - 16, 2, 32, 32, 32), isa.halt()]
        self.assertEqual(machine(d).run(PROG), C.ST_ERROR | C.ERR_VMEM_RANGE << 8)

    def test_loop_body_past_chunk(self):
        raw = isa.header(isa.arch_word(), 0, 64, 2, 192) + isa.loop(2, 5) + isa.halt()
        m = machine([isa.halt()])
        m.write_ddr(PROG, raw)
        self.assertEqual(m.run(PROG), C.ST_ERROR | C.ERR_LOOP_BODY << 8)


class Programs(unittest.TestCase):
    def test_chunking_keeps_loops_whole(self):
        body = [isa.load(2, 0, 0, 1, 16, 16, 16)] * 100
        descs = body + body + [isa.loop(3, 100)] + body + [isa.halt()]
        m = machine(descs)
        self.assertEqual(m.run(PROG), C.ST_DONE, m.message)
        self.assertEqual(m.chunk_no, 1)
        self.assertEqual(m.desc_count, 200 + 1 + 1 + 300 + 1)     # + NEXT_CHUNK

    def test_order_does_not_change_a_race_free_program(self):
        ref = None
        for seed in range(8):
            m = machine(gemm_program(), order_seed=seed)
            m.run(PROG)
            out = m.read_ddr(R3, 1024)
            ref = ref or out
            self.assertEqual(out, ref)


if __name__ == "__main__":
    unittest.main()
