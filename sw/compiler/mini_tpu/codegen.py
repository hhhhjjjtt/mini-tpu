"""Compiles a quantized GPT into a program and DDR images (INTERFACE_SPEC.md).

Target holds the hardware limits. Emitter builds descriptors and works out their waits from
which unit last touched each vmem word, so ordering is right by construction; hazards.py
checks the result independently. The first version is fully unrolled, with one fixed vmem
buffer per tensor.
"""
import bisect
import hashlib
import struct
from dataclasses import dataclass

import numpy as np

from . import isa
from .isa import C, UNIT_OF


@dataclass(frozen=True)
class Target:
    n: int = C.TILE_N                       # GEMM tile width
    vmem_bytes: int = C.VMEM_KB << 10
    max_k: int = C.MAX_K
    max_row_bytes: int = 4096               # LOAD / STORE: an ISA limit
    max_len: int = C.MAX_LEN                # softmax, layernorm
    arch: int = None                        # program header
    config_hash: int = 0

    @classmethod
    def from_hardware(cls, hw):
        if (hw.tile_n, hw.vector_lanes) != (16, 16):
            raise ValueError("the compiler and simulator only support array_n of 8 or 16 and vector_lanes = 16 so far")
        return cls(hw.tile_n, hw.vmem_bytes, hw.max_k, 4096, hw.max_len, hw.arch_word, hw.hash)


def words(addr, rows, row_bytes, pitch):
    w0, n, p = addr >> 4, row_bytes >> 4, pitch >> 4
    return (w0 + np.arange(rows, dtype=np.int64)[:, None] * p + np.arange(n, dtype=np.int64)).ravel()


class Emitter:
    def __init__(self, target):
        self.t = target
        self.nv = target.vmem_bytes >> 4
        self.last_w = np.zeros((4, self.nv + 2 * target.max_k), dtype=np.int32)
        self.last_r = np.zeros_like(self.last_w)
        self.issued = [0, 0, 0, 0]
        self.gemms = []                     # MATRIX seq of each GEMM
        self.epilogues = 0
        self.half = 0
        self.descs = []

    def _half_rows(self, half, k):
        return self.nv + half * self.t.max_k + np.arange(k, dtype=np.int64)

    def _fence(self):
        self.descs.append(isa.fence())
        self.last_w[:] = 0
        self.last_r[:] = 0

    def _matrix_wait(self, seq, is_epilogue):
        """A VECTOR op must not wait on MATRIX (§4.2 rule 4). It needs none if a GEMM at or after seq
        has its epilogue earlier in the VECTOR stream, because that GEMM finished first."""
        i = bisect.bisect_left(self.gemms, seq)
        if i < self.epilogues:
            return
        if is_epilogue:
            raise ValueError("an epilogue overwrites data its own GEMM still reads")
        self._fence()

    def emit(self, d, reads=(), writes=(), after=None):
        """after: {unit: seq} waits the vmem tracking cannot see, such as a STORE then a LOAD of the same DDR."""
        unit = UNIT_OF[d[0]]
        need = np.zeros(4, dtype=np.int64)
        for v, seq in (after or {}).items():
            need[v] = max(need[v], seq)
        for idx in reads:
            need = np.maximum(need, self.last_w[:, idx].max(axis=1))
        for idx in writes:
            need = np.maximum(need, np.maximum(self.last_w[:, idx].max(axis=1), self.last_r[:, idx].max(axis=1)))
        need[unit] = 0
        if unit == 2 and need[1]:
            self._matrix_wait(int(need[1]), d[0] == C.OP_VEC_EPILOGUE)
            need[1] = 0
            if not self.last_w.any():       # fenced: nothing left to wait for
                need[:] = 0
        for v in range(4):
            if need[v]:
                dist = self.issued[v] - int(need[v])
                assert dist < 1 << 16
                isa.wait(d, v, dist)
        self.issued[unit] += 1
        seq = self.issued[unit]
        for idx in reads:
            self.last_r[unit, idx] = seq
        for idx in writes:
            self.last_w[unit, idx] = seq
        self.descs.append(d)
        return seq

    # ---- memory ----
    def load(self, region, off, vmem, rows, row_bytes, ddr_stride=None, vmem_stride=None, after=None):
        ds = row_bytes if ddr_stride is None else ddr_stride
        vs = row_bytes if vmem_stride is None else vmem_stride
        return self.emit(isa.load(region, off, vmem, rows, row_bytes, ds, vs), writes=[words(vmem, rows, row_bytes, vs)], after=after)

    def load_blob(self, region, off, vmem, nbytes):
        """A contiguous block of any size."""
        m = self.t.max_row_bytes
        if nbytes >= m:
            self.load(region, off, vmem, nbytes // m, m)
        if nbytes % m:
            self.load(region, off + nbytes // m * m, vmem + nbytes // m * m, 1, nbytes % m)

    def store(self, region, off, vmem, rows, row_bytes, ddr_stride=None, vmem_stride=None):
        ds = row_bytes if ddr_stride is None else ddr_stride
        vs = row_bytes if vmem_stride is None else vmem_stride
        return self.emit(isa.store(region, off, vmem, rows, row_bytes, ds, vs), reads=[words(vmem, rows, row_bytes, vs)])

    def load_wgt(self, off, k):
        """Fills the next weight half. Returns it."""
        self.half ^= 1
        self.emit(isa.load_wgt(0, off, self.half, k), writes=[self._half_rows(self.half, k)])
        return self.half

    def load_wgt_vm(self, src, pitch, k, transpose):
        self.half ^= 1
        area = words(src, 16, k, pitch) if transpose else words(src, k, 16, pitch)
        self.emit(isa.load_wgt_vm(src, pitch, k, self.half, transpose), reads=[area], writes=[self._half_rows(self.half, k)])
        return self.half

    def gemm(self, act, pitch, m_tiles, k, half, epilogue):
        """GEMM and its epilogue, which must stay adjacent (§4.2 rule 3).

        epilogue: dict of mode, dst, dst_pitch and optionally params, flags, partial, partial_pitch.
        """
        n = self.t.n
        rows = n * m_tiles
        seq = self.emit(isa.gemm(act, pitch, m_tiles, k, half), reads=[words(act, rows, k, pitch), self._half_rows(half, k)])
        self.gemms.append(seq)
        e = dict(epilogue)
        mode, dst, dp = e["mode"], e["dst"], e["dst_pitch"]
        flags, params = e.get("flags", 0), e.get("params", 0)
        partial, pp = e.get("partial", 0), e.get("partial_pitch", 0)
        i32 = mode in (C.EP_STORE_I32, C.EP_ACC_I32)
        reads = [] if i32 else [words(params, 1, 16 if flags & C.EP_F_PER_TENSOR else 256, 0)]
        if mode in (C.EP_ACC_I32, C.EP_ACC_REQUANT_I8):
            reads.append(words(partial, rows, 64, pp))
        self.emit(isa.epilogue(mode, flags, 0, -128, 127, rows, dp, dst, params, partial, pp), reads=reads,
                  writes=[words(dst, rows, 64 if i32 else 16, dp)])
        self.epilogues += 1

    # ---- vector ----
    def layernorm(self, rows, n, src, dst, prm):
        self.emit(isa.layernorm(True, rows, n, n * 4, src, dst, n, prm[0], prm[1], prm[2], prm[3], prm[4], 0),
                  reads=[words(src, rows, n * 4, n * 4)], writes=[words(dst, rows, n, n)])

    def softmax(self, rows, n, src, dst, in_ms, out_ms, mask=None):
        """CAUSAL from row 0, or with mask (an i32 row in vmem) instead."""
        reads = [words(src, rows, n * 4, n * 4)] + ([] if mask is None else [words(mask, 1, n * 4, 0)])
        self.emit(isa.softmax(True, rows, n, n * 4, src, dst, n, in_ms[0], in_ms[1], 0, out_ms[0], out_ms[1], 0,
                              mask=mask or 0, has_mask=mask is not None, causal=mask is None, q_pos=0),
                  reads=reads, writes=[words(dst, rows, n, n)])

    def lut(self, rows, cols, addr, table):
        area = words(addr, rows, cols, cols)
        self.emit(isa.vec_lut(rows, cols, addr, addr, cols, cols, table), reads=[area, words(table, 1, 256, 0)], writes=[area])

    def eltwise(self, op, b_bcast, rows, cols, a, b, out, params):
        """i32 in and out, row pitch = cols."""
        p = cols * 4
        breads = words(b, 1, p, 0) if b_bcast == C.EW_BCAST_ROW else words(b, rows, p, p)
        self.emit(isa.eltwise(op, C.DT_I32, C.DT_I32, C.DT_I32, b_bcast, rows, cols, a, b, out, p, p, p, params),
                  reads=[words(a, rows, p, p), breads, words(params, 1, 64, 0)], writes=[words(out, rows, p, p)])


class Blob:
    """A DDR image built front to back; every block is 16 B aligned. A block identical to an
    earlier one is not stored again, so programs compiled into one Blob share their weights."""
    def __init__(self):
        self.parts, self.size, self.seen = [], 0, {}

    def put(self, data):
        data = bytes(data)
        key = hashlib.blake2b(data, digest_size=16).digest()
        if key in self.seen:
            return self.seen[key]
        off = self.size
        pad = -len(data) % 16
        self.parts.append(data + b"\0" * pad)
        self.size += len(data) + pad
        self.seen[key] = off
        return off

    def bytes(self):
        return b"".join(self.parts)


def requant_tables(lin, bias=None):
    """§9.1: one 256 B table per 16 output columns."""
    n = lin.w.shape[1]
    t = np.zeros((n, 16), dtype=np.uint8)
    b = lin.bias if bias is None else bias
    t[:, 0:4] = b.astype("<i4").view(np.uint8).reshape(n, 4)
    t[:, 4:8] = lin.mult.astype("<i4").view(np.uint8).reshape(n, 4)
    t[:, 8] = lin.shift
    return t


def eltwise_params(a=(1 << 30, 30), b=(1 << 30, 30), mul_shift=0):
    p = bytearray(64)
    struct.pack_into("<iiB", p, C.EWP_A_ZP, 0, *a)
    struct.pack_into("<iiB", p, C.EWP_B_ZP, 0, *b)
    struct.pack_into("<iBB", p, C.EWP_OUT_MULT, 1 << 30, 30, mul_shift)
    struct.pack_into("<bbbB", p, C.EWP_OUT_ZP, 0, -128, 127, 0)
    return p


class Allocator:
    def __init__(self, size):
        self.top, self.size = 0, size

    def __call__(self, n):
        a = self.top
        self.top += (n + 255) & ~255
        if self.top > self.size:
            raise ValueError(f"vmem overflow: need {self.top} bytes of {self.size}")
        return a


R_WGT, R_PRM, R_IN, R_OUT = C.REGION_WEIGHTS, C.REGION_PARAMS, C.REGION_INPUT, C.REGION_OUTPUT
R_KV, R_KV_ROW = C.REGION_KV, C.REGION_KV_ROW
MASKED = -(1 << 31)                     # decode mask entry of a position after this token


@dataclass
class Compiled:
    program: bytes
    weights: bytes              # region 0
    params: bytes               # region 1
    input_bytes: int            # region 2. Prefill: i32 [S, D]. Decode: i32 [D], then the mask, i32 [ctx]
    output_bytes: int           # region 3. Prefill: i32 [S, vocab_pad]. Decode: i32 [vocab_pad]
    vmem_used: int
    descriptors: int
    kv_bytes: int = 0           # region 5


@dataclass
class Model:
    """Prefill and one decode program per context bucket, sharing weights.bin and params.bin."""
    prefill: Compiled
    decode: dict                # bucket L -> Compiled; L is how many cached positions the program reads
    ctx: int                    # longest context

    @property
    def kv_bytes(self):
        return self.prefill.kv_bytes


def default_ctx(m, t):
    """Longest context: the position embeddings, one softmax row, the mask in one LOAD row."""
    return min(len(m.wpe), t.max_len, t.max_row_bytes // 4) // 16 * 16


def buckets(ctx):
    """Decode programs: 16, 32, 64, ... up to ctx. Position p runs the smallest L > p."""
    out, L = [], 16
    while L < ctx:
        out.append(L)
        L *= 2
    return out + [ctx]


def kv_offset(cfg, ctx, layer, which):
    """KV cache: per layer K then V (which = 0, 1), each [ctx, d_model] i8, row = position."""
    return (2 * layer + which) * ctx * cfg.d_model


class _Builder:
    """One program: emitter, vmem, and the parts of a layer that prefill and decode share.

    rows: real activation rows. GEMMs always compute m_tiles * n rows; decode ignores all but row 0.
    """
    def __init__(self, m, t, wts, prm, rows):
        self.m, self.t, self.wts, self.prm, self.rows = m, t, wts, prm, rows
        cfg = m.cfg
        d, f, n = cfg.d_model, cfg.d_ff, t.n
        self.mt = -(-rows // n)
        R = self.mt * n
        self.em, self.alloc = Emitter(t), Allocator(t.vmem_bytes)
        a = self.alloc
        self.x, self.h, self.qkv, self.ctx = a(rows * d * 4), a(R * d), a(R * 3 * d), a(R * d)
        self.acc, self.hid = a(R * d * 4), a(R * f)
        self.partial = a(R * n * 4)
        self.table = a(max(3 * d, f) * 16)          # requant tables of the current GEMM
        self.rows_v = a(2 * d * 4)                  # bias row, multiplier row
        small = a(4 * 64 + 256 + 16)                # eltwise params, GELU table, per-tensor table
        self.ew_mul, self.ew_add, self.gelu, self.pv_table = small, small + 64, small + 256, small + 512

    def linear(self, lin, act, pitch, dst, dst_pitch, i32):
        """i32: bias row at rows_v, result stays in accumulators. Else requant through `table`."""
        t, em, n = self.t, self.em, self.t.n
        K, N = lin.w.shape
        chunks = [(k, min(t.max_k, K - k)) for k in range(0, K, t.max_k)]
        for c in range(0, N, n):
            tile = lin.w[:, c:c + n]
            for i, (k0, k) in enumerate(chunks):
                half = em.load_wgt(self.wts.put(tile[k0:k0 + k]), k)
                first, last = i == 0, i == len(chunks) - 1
                if i32:
                    col = dst + c * 4
                    e = {"mode": C.EP_ACC_I32, "dst": col, "dst_pitch": dst_pitch, "partial": self.rows_v + c * 4 if first else col,
                         "partial_pitch": 0 if first else dst_pitch}
                elif len(chunks) == 1:
                    e = {"mode": C.EP_REQUANT_I8, "dst": dst + c, "dst_pitch": dst_pitch, "params": self.table + c * 16}
                elif last:
                    e = {"mode": C.EP_ACC_REQUANT_I8, "dst": dst + c, "dst_pitch": dst_pitch, "params": self.table + c * 16,
                         "partial": self.partial, "partial_pitch": n * 4}
                elif first:
                    e = {"mode": C.EP_STORE_I32, "dst": self.partial, "dst_pitch": n * 4}
                else:
                    e = {"mode": C.EP_ACC_I32, "dst": self.partial, "dst_pitch": n * 4, "partial": self.partial, "partial_pitch": n * 4}
                em.gemm(act + k0, pitch, self.mt, k, half, e)

    def load_tables(self, lin):
        data = requant_tables(lin)
        self.em.load_blob(R_PRM, self.prm.put(data), self.table, data.size)

    def bias_row(self, lin):
        d = self.m.cfg.d_model
        self.em.load_blob(R_PRM, self.prm.put(lin.bias.astype("<i4")), self.rows_v, d * 4)

    def residual(self, branch, keep):
        """x = keep(x) + branch(acc); the accumulators already hold the bias."""
        em, prm, d, rows = self.em, self.prm, self.m.cfg.d_model, self.rows
        mult, shift = branch
        em.load(R_PRM, prm.put(eltwise_params(mul_shift=shift)), self.ew_mul, 1, 64)
        em.load(R_PRM, prm.put(eltwise_params(b=keep)), self.ew_add, 1, 64)
        em.load_blob(R_PRM, prm.put(mult.astype("<i4")), self.rows_v + d * 4, d * 4)
        em.eltwise(C.EW_MUL, C.EW_BCAST_ROW, rows, d, self.acc, self.rows_v + d * 4, self.acc, self.ew_mul)
        em.eltwise(C.EW_ADD, C.EW_BCAST_FULL, rows, d, self.acc, self.x, self.x, self.ew_add)

    def attention_in(self, y):
        """LayerNorm, then Q, K and V side by side in qkv."""
        d = self.m.cfg.d_model
        self.em.layernorm(self.rows, d, self.x, self.h, y.ln1)
        self.load_tables(y.attn)
        self.linear(y.attn, self.h, d, self.qkv, 3 * d, False)
        tbl = bytearray(16)
        struct.pack_into("<iiB", tbl, 0, 0, y.pv[0], y.pv[1])
        self.em.load(R_PRM, self.prm.put(tbl), self.pv_table, 1, 16)

    def attention_out(self, y):
        """Projection, residual, MLP, residual."""
        em, cfg, rows = self.em, self.m.cfg, self.rows
        d, f = cfg.d_model, cfg.d_ff
        self.bias_row(y.proj)
        self.linear(y.proj, self.ctx, d, self.acc, d * 4, True)
        self.residual(y.branch1, y.keep1)
        em.layernorm(rows, d, self.x, self.h, y.ln2)
        self.load_tables(y.fc)
        self.linear(y.fc, self.h, d, self.hid, f, False)
        em.load(R_PRM, self.prm.put(y.gelu), self.gelu, 1, 256)
        em.lut(rows, f, self.hid, self.gelu)
        self.bias_row(y.mlp)
        self.linear(y.mlp, self.hid, f, self.acc, d * 4, True)
        self.residual(y.branch2, y.keep2)

    def finish(self, regions, program_id, input_bytes, output_bytes, kv_bytes):
        em, t = self.em, self.t
        em.descs.append(isa.halt())
        program = isa.build_program(em.descs, regions, t.arch, program_id, t.config_hash)
        return Compiled(program, b"", b"", input_bytes, output_bytes, self.alloc.top, len(em.descs), kv_bytes)


def _prefill(m, t, wts, prm, ctx):
    cfg = m.cfg
    S, d, f, dh, H = cfg.seq, cfg.d_model, cfg.d_ff, cfg.d_head, cfg.n_head
    n, vpad = t.n, cfg.vocab_pad
    assert S % n == 0 and d % n == 0 and f % n == 0 and dh % n == 0, "dimensions must be multiples of the array width"
    assert S <= t.max_k and S <= t.max_len and d <= t.max_len, "S must fit one GEMM and one softmax row"
    assert d * 4 <= t.max_row_bytes, "a residual row must fit one LOAD"
    assert S <= ctx, "the KV cache must hold the prompt"
    b = _Builder(m, t, wts, prm, S)
    em, alloc, mt = b.em, b.alloc, b.mt
    scores, probs = alloc(S * S * 4), alloc(S * S)
    logit = [alloc(S * n * 4), alloc(S * n * 4)]

    em.load(R_IN, 0, b.x, S, d * 4)
    for l, y in enumerate(m.layers):
        b.attention_in(y)
        for which in (0, 1):
            em.store(R_KV, kv_offset(cfg, ctx, l, which), b.qkv + (1 + which) * d, S, d, d, 3 * d)
        for i in range(H):
            q, k, v = b.qkv + i * dh, b.qkv + d + i * dh, b.qkv + 2 * d + i * dh
            for blk in range(mt):
                half = em.load_wgt_vm(k + blk * n * 3 * d, 3 * d, dh, True)
                em.gemm(q, 3 * d, mt, dh, half, {"mode": C.EP_STORE_I32, "dst": scores + blk * n * 4, "dst_pitch": S * 4})
            em.softmax(S, S, scores, probs, y.qk, (m.out_mult, m.out_shift))
            for c in range(0, dh, n):
                half = em.load_wgt_vm(v + c, 3 * d, S, False)
                em.gemm(probs, S, mt, S, half, {"mode": C.EP_REQUANT_I8, "flags": C.EP_F_PER_TENSOR, "dst": b.ctx + i * dh + c,
                                                "dst_pitch": d, "params": b.pv_table})
        b.attention_out(y)

    em.layernorm(S, d, b.x, b.h, m.lnf)
    head = m.head
    chunks = [(k, min(t.max_k, d - k)) for k in range(0, d, t.max_k)]
    bias_off = prm.put(head.bias.astype("<i4"))
    for j, c in enumerate(range(0, vpad, n)):
        buf = logit[j & 1]
        em.load(R_PRM, bias_off + c * 4, b.rows_v, 1, n * 4)
        for i, (k0, k) in enumerate(chunks):
            half = em.load_wgt(wts.put(head.w[k0:k0 + k, c:c + n]), k)
            em.gemm(b.h + k0, d, mt, k, half, {"mode": C.EP_ACC_I32, "dst": buf, "dst_pitch": n * 4, "partial": b.rows_v if i == 0 else buf,
                                               "partial_pitch": 0 if i == 0 else n * 4})
        em.store(R_OUT, c * 4, buf, S, n * 4, vpad * 4)
    regions = 1 << R_WGT | 1 << R_PRM | 1 << R_IN | 1 << R_OUT | 1 << R_KV
    return b.finish(regions, 0, S * d * 4, S * vpad * 4, 2 * cfg.n_layer * ctx * d)


def _check_mask_exact(m):
    """A masked score must act exactly like a CAUSAL one (§5.7): MASKED plus any score the cache
    can produce stays below every real score by more than E() can see. True if |t| < 2^30 - 2^20."""
    dh = m.cfg.d_head
    for y in m.layers:
        mult, shift = y.qk
        bound = dh * 128 * 128 * mult / 2.0 ** shift
        assert bound < (1 << 30) - (1 << 20), f"scores up to {bound:.3g} in Q16: the decode mask would not be exact"


def _decode(m, t, wts, prm, ctx, L):
    """One token at the position the ARM chose: R2 holds its embedding and the mask, R6 points at
    its KV cache row. Reads L cached positions; the mask blocks those after this token."""
    cfg = m.cfg
    d, dh, H = cfg.d_model, cfg.d_head, cfg.n_head
    n, vpad = t.n, cfg.vocab_pad
    assert L % n == 0 and L <= ctx and L <= t.max_len and L * 4 <= t.max_row_bytes
    b = _Builder(m, t, wts, prm, 1)
    em, alloc = b.em, b.alloc
    R = n                                           # rows the matrix unit computes
    G = min(t.max_row_bytes // (n * 4), vpad // n)  # head tiles per STORE
    kbuf, vbuf = alloc(L * d), alloc(L * d)
    scores, probs, mask = alloc(R * L * 4), alloc(R * L), alloc(L * 4)
    hbias = [alloc(G * n * 4), alloc(G * n * 4)]
    logit = [alloc(R * G * n * 4), alloc(R * G * n * 4)]
    chunks = [(k, min(t.max_k, L - k)) for k in range(0, L, t.max_k)]

    em.load(R_IN, 0, b.x, 1, d * 4)
    em.load(R_IN, d * 4, mask, 1, L * 4)
    zero = prm.put(bytes(R * max(d, L)))
    em.load_blob(R_PRM, zero, b.h, R * d)           # rows past 0 are only ever read by the matrix unit
    em.load_blob(R_PRM, zero, probs, R * L)
    for l, y in enumerate(m.layers):
        b.attention_in(y)
        for which, buf in ((0, kbuf), (1, vbuf)):
            off = kv_offset(cfg, ctx, l, which)
            seq = em.store(R_KV_ROW, off, b.qkv + (1 + which) * d, 1, d)
            em.load(R_KV, off, buf, L, d, after={3: seq})         # includes the row just stored
        for i in range(H):
            q = b.qkv + i * dh
            for blk in range(L // n):
                half = em.load_wgt_vm(kbuf + blk * n * d + i * dh, d, dh, True)
                em.gemm(q, 3 * d, 1, dh, half, {"mode": C.EP_STORE_I32, "dst": scores + blk * n * 4, "dst_pitch": L * 4})
            em.softmax(1, L, scores, probs, y.qk, (m.out_mult, m.out_shift), mask=mask)
            pt = {"flags": C.EP_F_PER_TENSOR, "params": b.pv_table}
            for c in range(0, dh, n):
                dst = b.ctx + i * dh + c
                for j, (k0, k) in enumerate(chunks):
                    half = em.load_wgt_vm(vbuf + k0 * d + i * dh + c, d, k, False)
                    if len(chunks) == 1:
                        e = {"mode": C.EP_REQUANT_I8, "dst": dst, "dst_pitch": d, **pt}
                    elif j == len(chunks) - 1:
                        e = {"mode": C.EP_ACC_REQUANT_I8, "dst": dst, "dst_pitch": d, "partial": b.partial, "partial_pitch": n * 4, **pt}
                    elif j == 0:
                        e = {"mode": C.EP_STORE_I32, "dst": b.partial, "dst_pitch": n * 4}
                    else:
                        e = {"mode": C.EP_ACC_I32, "dst": b.partial, "dst_pitch": n * 4, "partial": b.partial, "partial_pitch": n * 4}
                    em.gemm(probs + k0, L, 1, k, half, e)
        b.attention_out(y)

    em.layernorm(1, d, b.x, b.h, m.lnf)
    head = m.head
    hk = [(k, min(t.max_k, d - k)) for k in range(0, d, t.max_k)]
    bias_off = prm.put(head.bias.astype("<i4"))
    for g, g0 in enumerate(range(0, vpad, G * n)):
        cols = min(G * n, vpad - g0)
        hb, buf = hbias[g & 1], logit[g & 1]
        em.load(R_PRM, bias_off + g0 * 4, hb, 1, cols * 4)
        for c in range(0, cols, n):
            for i, (k0, k) in enumerate(hk):
                half = em.load_wgt(wts.put(head.w[k0:k0 + k, g0 + c:g0 + c + n]), k)
                em.gemm(b.h + k0, d, 1, k, half, {"mode": C.EP_ACC_I32, "dst": buf + c * 4, "dst_pitch": G * n * 4,
                                                  "partial": hb + c * 4 if i == 0 else buf + c * 4,
                                                  "partial_pitch": 0 if i == 0 else G * n * 4})
        em.store(R_OUT, g0 * 4, buf, 1, cols * 4)
    regions = 1 << R_WGT | 1 << R_PRM | 1 << R_IN | 1 << R_OUT | 1 << R_KV | 1 << R_KV_ROW
    return b.finish(regions, L, d * 4 + ctx * 4, vpad * 4, 2 * cfg.n_layer * ctx * d)


def _shared(c, wts, prm):
    c.weights, c.params = wts, prm
    return c


def compile_gpt(m, target=None, ctx=None):
    """The prefill program alone, with its own weights and params."""
    t = target or Target()
    wts, prm = Blob(), Blob()
    c = _prefill(m, t, wts, prm, ctx or default_ctx(m, t))
    return _shared(c, wts.bytes(), prm.bytes())


def compile_model(m, target=None, ctx=None):
    """Prefill and the decode programs, all reading one weights.bin and one params.bin."""
    t = target or Target()
    ctx = ctx or default_ctx(m, t)
    _check_mask_exact(m)
    wts, prm = Blob(), Blob()
    pre = _prefill(m, t, wts, prm, ctx)
    dec = {L: _decode(m, t, wts, prm, ctx, L) for L in buckets(ctx)}
    w, p = wts.bytes(), prm.bytes()
    return Model(_shared(pre, w, p), {L: _shared(c, w, p) for L, c in dec.items()}, ctx)


def run(machine, compiled, x, bases):
    """Loads a compiled prefill into a Machine and runs it. bases: prog, regions 0..3, KV cache. Returns STATUS."""
    prog, r0, r1, r2, r3, kv = bases
    machine.region[:4] = [r0, r1, r2, r3]
    machine.region[R_KV] = kv
    machine.write_ddr(prog, compiled.program)
    machine.write_ddr(r0, compiled.weights)
    machine.write_ddr(r1, compiled.params)
    machine.write_ddr(r2, x.astype("<i4").tobytes())
    return machine.run(prog)
