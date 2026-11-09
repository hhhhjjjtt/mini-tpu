"""Functional simulator: runs program blobs bit-exactly (INTERFACE_SPEC.md §3-§11).

Machine.run() follows the firmware (hw/fw/main.c) for control ops and the issue engine for
unit ops, then executes commands in any order the wait conditions allow. It has no timing.
Set order_seed to randomize that order; a race-free program gives the same result anyway.
"""
import random
import struct
from collections import deque
from dataclasses import dataclass

import numpy as np

from . import golden
from .hazards import Checker
from .isa import C, DESC, MEM_OPS, UNIT_OF, arch_word, decode

FIFO_DEPTH = C.FIFO_DEPTH


class UnitError(Exception):
    def __init__(self, unit, code, info0=0):
        super().__init__(f"unit {unit}: error 0x{code:02x}")
        self.unit, self.code, self.info0 = unit, code, info0


class Deadlock(Exception):
    pass


@dataclass
class Command:
    op: int
    unit: int
    seq: int            # 1-based position in the unit's command stream
    f: dict
    targets: tuple      # absolute DONE target per unit, 0 = no wait
    addr: int           # resolved DDR address (memory ops)
    loc: tuple          # (chunk, descriptor index)


class Machine:
    def __init__(self, ddr_bytes=1 << 20, ddr_base=0, vmem_bytes=C.VMEM_KB << 10, check=True, order_seed=None,
                 magic=C.PROG_MAGIC, hw_arch=None, on_desc=None, on_break=None):
        self.ddr = np.zeros(ddr_bytes, dtype=np.uint8)
        self.ddr_base = ddr_base
        self.vmem = np.zeros(vmem_bytes, dtype=np.uint8)
        self.weights = np.zeros((C.WGT_HALVES, C.MAX_K, 16), dtype=np.int8)
        self.region = [0] * C.NUM_REGIONS
        self.magic = magic
        self.hw_arch = arch_word(vmem_kb=vmem_bytes >> 10) if hw_arch is None else hw_arch
        self.check = check
        self.rng = random.Random(order_seed) if order_seed is not None else None
        self.on_desc, self.on_break = on_desc, on_break
        self.checker = None
        self.valid = np.zeros((vmem_bytes >> 4) + C.WGT_HALVES * C.MAX_K, dtype=bool)  # words written so far
        self.reset_status()

    def reset_status(self):
        self.status = C.ST_IDLE
        self.err_desc = self.err_unit = self.err_info0 = self.err_info1 = 0
        self.desc_count = 0
        self.message = ""

    # ---- memory ----
    def write_ddr(self, addr, data):
        o = addr - self.ddr_base
        self.ddr[o:o + len(data)] = np.frombuffer(bytes(data), dtype=np.uint8)

    def read_ddr(self, addr, n):
        o = addr - self.ddr_base
        return self.ddr[o:o + n].tobytes()

    def _ddr_view(self, addr, rows, row_bytes, stride):
        o = addr - self.ddr_base
        if o < 0 or o + (rows - 1) * stride + row_bytes > self.ddr.size:
            raise UnitError(0, C.ERR_UNIT_AXI, addr & 0xFFFFFFFF)
        return np.lib.stride_tricks.as_strided(self.ddr[o:], (rows, row_bytes), (stride, 1))

    def _vm(self, addr, rows, row_bytes, pitch, unit):
        """Writable [rows, row_bytes] view of vmem."""
        if addr + (rows - 1) * pitch + row_bytes > self.vmem.size:
            raise UnitError(unit, C.ERR_VMEM_RANGE, addr)
        return np.lib.stride_tricks.as_strided(self.vmem[addr:], (rows, row_bytes), (pitch, 1))

    def _vm_i32(self, addr, rows, cols, pitch, unit):
        return self._vm(addr, rows, cols * 4, pitch, unit).copy().view(np.int32)

    def _vm_i8(self, addr, rows, cols, pitch, unit):
        return self._vm(addr, rows, cols, pitch, unit).copy().view(np.int8)

    # ---- firmware: program ----
    def run(self, prog_base):
        """Runs one program. Returns STATUS."""
        self.reset_status()
        self.status = C.ST_RUNNING
        self.prog_base = prog_base
        self.loop_add = [0] * C.NUM_REGIONS
        self.loops = []
        self.chunk_no = 0
        self.issued = [0] * 4
        self.done = [0] * 4
        self.fifo = [deque() for _ in range(4)]
        if self.check:
            self.checker = Checker(self.vmem.size, self.valid)
        try:
            self._program()
        except UnitError as e:
            self._fail(e.code, e.unit, e.info0, 0, 0)
        except Deadlock as e:
            self.message = str(e)
            self._fail(C.ERR_WATCHDOG, C.UNIT_FW, 0, 0, 0)
        return self.status

    def _fail(self, code, unit, info0, info1, idx):
        self.err_desc = self.chunk_no << 16 | idx
        self.err_unit, self.err_info0, self.err_info1 = unit, info0, info1
        self.status = C.ST_ERROR | code << 8

    def _program(self):
        h = self.read_ddr(self.prog_base, C.PROG_HDR_SIZE)
        magic = struct.unpack_from("<I", h, C.HDR_MAGIC)[0]
        if magic != self.magic:
            return self._fail(C.ERR_BAD_MAGIC, C.UNIT_FW, magic, 0, 0)
        fmt = struct.unpack_from("<H", h, C.HDR_FORMAT)[0]
        if fmt != C.PROG_FORMAT:
            return self._fail(C.ERR_VERSION, C.UNIT_FW, fmt, 0, 0)
        arch = struct.unpack_from("<I", h, C.HDR_ARCH)[0]
        hw = self.hw_arch
        if arch & 0xFF != hw & 0xFF or (arch >> 8 & 0xFF) & ~(hw >> 8 & 0xFF) or arch >> 16 > hw >> 16:
            return self._fail(C.ERR_ARCH, C.UNIT_FW, arch, hw, 0)
        missing = sum(1 << r for r in range(8) if h[C.HDR_REGIONS] >> r & 1 and self.region[r] == 0)
        if missing:
            return self._fail(C.ERR_REGION_UNSET, C.UNIT_FW, missing, 0, 0)
        off, length = struct.unpack_from("<IH", h, C.HDR_FIRST_OFF)
        if not self._chunk_ok(off, length):
            return self._fail(C.ERR_CHUNK, C.UNIT_FW, off, length, 0)
        while True:
            chunk = self.read_ddr(self.prog_base + off, length * DESC)
            last = chunk[(length - 1) * DESC:]
            nxt = None
            if last[0] == C.OP_NEXT_CHUNK:
                f = decode(last)
                nxt = (f["offset"], f["len"])
                if not self._chunk_ok(*nxt):
                    return self._fail(C.ERR_CHUNK, C.UNIT_FW, *nxt, length - 1)
            elif last[0] != C.OP_HALT:
                return self._fail(C.ERR_CHUNK, C.UNIT_FW, last[0], length, length - 1)
            if not self._chunk(chunk, length):
                return
            off, length = nxt
            self.chunk_no += 1

    @staticmethod
    def _chunk_ok(off, length):
        return 1 <= length <= C.CHUNK_MAX_DESCS and off & 0xF == 0

    # ---- firmware: chunk interpreter (run_chunk in main.c) ----
    def _chunk(self, chunk, length):
        """Returns True at NEXT_CHUNK, False at HALT or error."""
        i = 0
        while i < length:
            d = chunk[i * DESC:(i + 1) * DESC]
            op = d[0]
            f = decode(d)
            if self.on_desc:
                self.on_desc(d, f)
            self.desc_count += 1
            if op in UNIT_OF:
                if op in MEM_OPS and f["region"] > 7:
                    self.desc_count -= 1
                    self._fail(C.ERR_BAD_FIELD, C.UNIT_FW, C.D_REGION, f["region"], i)
                    return False
                self._issue(op, f, i)
                i += 1
            elif op == C.OP_NOP:
                i += 1
            elif op == C.OP_LOOP:
                if not self._loop(f, i, length):
                    return False
                i += 1
            elif op == C.OP_FENCE:
                self._drain()
                i += 1
            elif op == C.OP_BREAK:
                self._drain()
                if self.on_break:
                    self.on_break(self, f["break_id"])
                i += 1
            elif op == C.OP_NEXT_CHUNK:
                if i != length - 1:
                    self._fail(C.ERR_CHUNK, C.UNIT_FW, i, length, i)
                    return False
                return True
            elif op == C.OP_HALT:
                self._drain()
                self.status = C.ST_DONE
                return False
            else:
                self._fail(C.ERR_BAD_OPCODE, C.UNIT_FW, op, 0, i)
                return False
            while self.loops and i == self.loops[-1]["end"]:
                lp = self.loops[-1]
                lp["remaining"] -= 1
                if lp["remaining"]:
                    for r in range(8):
                        if lp["stride"][r]:
                            lp["acc"][r] += lp["stride"][r]
                            self.loop_add[r] += lp["stride"][r]
                    i = lp["start"]
                    break
                for r in range(8):
                    self.loop_add[r] -= lp["acc"][r]
                self.loops.pop()
        self._fail(C.ERR_CHUNK, C.UNIT_FW, i, length, min(i, 255))
        return False

    def _loop(self, f, i, length):
        if f["count"] == 0 or f["body_len"] == 0:
            self._fail(C.ERR_BAD_FIELD, C.UNIT_FW, C.LOOP_COUNT if f["count"] == 0 else C.LOOP_BODY_LEN, 0, i)
            return False
        if len(self.loops) == C.LOOP_MAX_DEPTH:
            self._fail(C.ERR_LOOP_DEPTH, C.UNIT_FW, 0, 0, i)
            return False
        end = i + 1 + f["body_len"]
        if end > length - 1 or (self.loops and end > self.loops[-1]["end"]):
            self._fail(C.ERR_LOOP_BODY, C.UNIT_FW, end, length, i)
            return False
        self.loops.append({"start": i + 1, "end": end, "remaining": f["count"], "stride": f["strides"], "acc": [0] * 8})
        return True

    # ---- issue engine (§8.7) ----
    def _issue(self, op, f, i):
        unit = UNIT_OF[op]
        targets = tuple(max(0, self.issued[u] - f["wait_dist"][u]) if f["wait_mask"] >> u & 1 else 0 for u in range(4))
        addr = 0
        if op in MEM_OPS:
            r = f["region"]
            addr = self.region[r] + (f["ddr_offset"] + self.loop_add[r] & 0xFFFFFFFF)
        self.issued[unit] += 1
        cmd = Command(op, unit, self.issued[unit], f, targets, addr, (self.chunk_no, i))
        if self.checker:
            self.checker.issue(cmd)
        while len(self.fifo[unit]) == FIFO_DEPTH:
            if not self._step():
                raise Deadlock(self._state(f"issue blocked on unit {unit}"))
        self.fifo[unit].append(cmd)

    def _state(self, why):
        heads = [(q[0].op, q[0].targets) if q else None for q in self.fifo]
        return f"{why}: done {self.done}, heads {heads}"

    def _ready(self, cmd):
        return all(self.done[u] >= t for u, t in enumerate(cmd.targets))

    def _step(self):
        """Runs every command that can start now. Returns True if any did."""
        progress = False
        units = [0, 1, 2, 3]
        if self.rng:
            self.rng.shuffle(units)
        for u in units:
            q = self.fifo[u]
            if q and self._ready(q[0]) and self._fire(u, q[0]):
                progress = True
        return progress

    def _drain(self):
        while any(self.fifo):
            if not self._step():
                raise Deadlock(self._state("no command can run"))
        if self.checker:
            self.checker.fence()

    def _fire(self, u, cmd):
        if cmd.op == C.OP_VEC_EPILOGUE:
            return False                        # runs together with its GEMM
        if cmd.op == C.OP_GEMM:
            self._check_gemm(cmd.f)
            vq = self.fifo[2]
            if not vq or vq[0].op != C.OP_VEC_EPILOGUE or not self._ready(vq[0]):
                return False
            epi = vq.popleft()
            self.fifo[1].popleft()
            self._gemm_epilogue(cmd.f, epi.f)
            self.done[1] += 1
            self.done[2] += 1
            return True
        self.fifo[u].popleft()
        {C.OP_LOAD: self._load, C.OP_STORE: self._store, C.OP_LOAD_WGT: self._load_wgt,
         C.OP_LOAD_WGT_VM: self._load_wgt_vm, C.OP_VEC_ELTWISE: self._eltwise, C.OP_VEC_SOFTMAX: self._softmax,
         C.OP_VEC_LAYERNORM: self._layernorm, C.OP_VEC_LUT: self._lut}[cmd.op](cmd)
        self.done[u] += 1
        return True

    # ---- LOAD / STORE ----
    @staticmethod
    def _check_dma(cmd, unit):
        f = cmd.f
        bad = (f["rows"] == 0 or f["row_bytes"] == 0 or f["row_bytes"] & 15 or f["row_bytes"] > 4096 or cmd.addr & 15
               or f["ddr_stride"] & 15 or f["vmem"] & 15 or f["vmem_stride"] & 15)
        if bad:
            raise UnitError(unit, C.ERR_BAD_FIELD)

    def _load(self, cmd):
        self._check_dma(cmd, 0)
        f = cmd.f
        dst = self._vm(f["vmem"], f["rows"], f["row_bytes"], f["vmem_stride"], 0)
        dst[...] = self._ddr_view(cmd.addr, f["rows"], f["row_bytes"], f["ddr_stride"])

    def _store(self, cmd):
        self._check_dma(cmd, 3)
        f = cmd.f
        src = self._vm(f["vmem"], f["rows"], f["row_bytes"], f["vmem_stride"], 3)
        self._ddr_view(cmd.addr, f["rows"], f["row_bytes"], f["ddr_stride"])[...] = src

    def _load_wgt(self, cmd):
        f = cmd.f
        k = f["k"]
        if k == 0 or k & 15 or k > C.MAX_K or f["half"] >= C.WGT_HALVES or cmd.addr & 15:
            raise UnitError(0, C.ERR_BAD_FIELD)
        self.weights[f["half"], :k] = self._ddr_view(cmd.addr, k, 16, 16).view(np.int8)

    # ---- MATRIX ----
    @staticmethod
    def _addr_ok(a):
        return a & 15 == 0 and a >> 21 == 0

    def _check_gemm(self, f):
        k = f["k"]
        if (k == 0 or k & 15 or k > C.MAX_K or f["m_tiles"] == 0 or f["half"] >= C.WGT_HALVES or f["act_pitch"] & 15
                or not self._addr_ok(f["act"])):
            raise UnitError(1, C.ERR_BAD_FIELD)

    def _load_wgt_vm(self, cmd):
        f = cmd.f
        k = f["k"]
        if k == 0 or k & 15 or k > C.MAX_K or f["half"] >= C.WGT_HALVES or f["src_pitch"] & 15 or not self._addr_ok(f["src"]):
            raise UnitError(1, C.ERR_BAD_FIELD)
        if f["flags"] & C.WV_F_TRANSPOSE:
            self.weights[f["half"], :k] = self._vm_i8(f["src"], 16, k, f["src_pitch"], 1).T
        else:
            self.weights[f["half"], :k] = self._vm_i8(f["src"], k, 16, f["src_pitch"], 1)

    def _gemm_epilogue(self, g, e):
        rows = 16 * g["m_tiles"]
        if self._epilogue_bad(e):
            raise UnitError(2, C.ERR_BAD_FIELD)
        if e["rows"] != rows:
            raise UnitError(2, C.ERR_STREAM_MISMATCH, min(rows, e["rows"]))
        a = self._vm_i8(g["act"], rows, g["k"], g["act_pitch"], 1)
        c = golden.gemm(a, self.weights[g["half"], :g["k"]])

        mode = e["mode"]
        if mode in (C.EP_ACC_I32, C.EP_ACC_REQUANT_I8):
            c = golden.wrap32(c.astype(np.int64) + self._vm_i32(e["partial"], rows, 16, e["partial_pitch"], 2))
        if mode in (C.EP_STORE_I32, C.EP_ACC_I32):
            self._vm(e["dst"], rows, 64, e["dst_pitch"], 2)[...] = c.view(np.uint8)
            return
        tbl = self._vm(e["params"], 1 if e["flags"] & C.EP_F_PER_TENSOR else 16, 16, 16, 2).copy()
        bias, mult = tbl[:, 0:4].copy().view(np.int32)[:, 0], tbl[:, 4:8].copy().view(np.int32)[:, 0]
        out = golden.requant(c, bias, mult, tbl[:, 8], e["out_zp"], e["clamp_min"], e["clamp_max"],
                             bool(e["flags"] & C.EP_F_DOUBLE_ROUND))
        self._vm(e["dst"], rows, 16, e["dst_pitch"], 2)[...] = out.view(np.uint8)

    def _epilogue_bad(self, e):
        acc = e["mode"] in (C.EP_ACC_I32, C.EP_ACC_REQUANT_I8)
        return (e["mode"] > 3 or e["rows"] == 0 or e["dst_pitch"] & 15 or not self._addr_ok(e["dst"])
                or not self._addr_ok(e["params"]) or (acc and (not self._addr_ok(e["partial"]) or e["partial_pitch"] & 15)))

    # ---- VECTOR ----
    def _params(self, addr):
        b = self._vm(addr, 1, 64, 64, 2).tobytes()
        u = struct.unpack_from
        return {"a_zp": u("<i", b, C.EWP_A_ZP)[0], "a_mult": u("<i", b, C.EWP_A_MULT)[0], "a_shift": b[C.EWP_A_SHIFT],
                "b_zp": u("<i", b, C.EWP_B_ZP)[0], "b_mult": u("<i", b, C.EWP_B_MULT)[0], "b_shift": b[C.EWP_B_SHIFT],
                "out_mult": u("<i", b, C.EWP_OUT_MULT)[0], "out_shift": b[C.EWP_OUT_SHIFT], "mul_shift": b[C.EWP_MUL_SHIFT],
                "out_zp": u("<b", b, C.EWP_OUT_ZP)[0], "clamp_min": u("<b", b, C.EWP_CLAMP_MIN)[0],
                "clamp_max": u("<b", b, C.EWP_CLAMP_MAX)[0], "double_round": bool(b[C.EWP_FLAGS] & 1)}

    def _operand(self, addr, rows, cols, pitch, i32):
        if i32:
            return self._vm_i32(addr, rows, cols, pitch, 2)
        return self._vm_i8(addr, rows, cols, pitch, 2).astype(np.int32)

    def _eltwise(self, cmd):
        f = cmd.f
        op, at, bt, ot, bc = f["op"], f["types"] & 0xF, f["types"] >> 4, f["out_type"], f["b_bcast"]
        pitches = f["a_pitch"] | f["b_pitch"] | f["out_pitch"]
        pass_a = op == C.EW_PASS_A
        bad = (op > 5 or at > 1 or ot > 1 or (not pass_a and (bt > 1 or bc > 2)) or f["rows"] == 0 or f["cols"] == 0
               or f["cols"] & 15 or not self._addr_ok(f["a"]) or not self._addr_ok(f["out"])
               or not self._addr_ok(f["params"]) or (not pass_a and not self._addr_ok(f["b"])) or pitches & 15)
        if bad:
            raise UnitError(2, C.ERR_BAD_FIELD)
        rows, cols = f["rows"], f["cols"]
        a = self._operand(f["a"], rows, cols, f["a_pitch"], at == C.DT_I32)
        b = None
        if not pass_a:
            if bc == C.EW_BCAST_SCALAR:
                b = self._operand(f["b"], 1, 16, 16, False) if bt == C.DT_I8 else self._operand(f["b"], 1, 4, 16, True)
                b = b[:, :1]
            elif bc == C.EW_BCAST_ROW:
                b = self._operand(f["b"], 1, cols, 0, bt == C.DT_I32)
            else:
                b = self._operand(f["b"], rows, cols, f["b_pitch"], bt == C.DT_I32)
        out = golden.eltwise(op, a, b, self._params(f["params"]), ot == C.DT_I32, C)
        self._vm(f["out"], rows, cols * out.itemsize, f["out_pitch"], 2)[...] = out.view(np.uint8)

    def _layernorm(self, cmd):
        f = cmd.f
        n = f["len"]
        bad = (f["in_type"] > 1 or f["out_type"] != 0 or f["rows"] == 0 or n == 0 or n & 15 or n > C.MAX_LEN or f["in_pitch"] & 15
               or f["out_pitch"] & 15 or not self._addr_ok(f["in_addr"]) or not self._addr_ok(f["out"]) or f["eps"] == 0
               or f["inv_shift"] < 8 or f["inv_shift"] > 63 or f["c_shift"] > 63)
        if bad:
            raise UnitError(2, C.ERR_BAD_FIELD)
        x = self._operand(f["in_addr"], f["rows"], n, f["in_pitch"], f["in_type"] == C.DT_I32)
        out = golden.layernorm(x, f["inv_mult"], f["inv_shift"], f["eps"], f["c_mult"], f["c_shift"], f["out_zp"])
        self._vm(f["out"], f["rows"], n, f["out_pitch"], 2)[...] = out.view(np.uint8)

    def _lut(self, cmd):
        f = cmd.f
        bad = (f["rows"] == 0 or f["cols"] == 0 or f["cols"] & 15 or not self._addr_ok(f["in_addr"]) or not self._addr_ok(f["out"])
               or not self._addr_ok(f["table"]) or f["in_pitch"] & 15 or f["out_pitch"] & 15)
        if bad:
            raise UnitError(2, C.ERR_BAD_FIELD)
        table = self._vm_i8(f["table"], 1, 256, 256, 2)[0]
        x = self._vm_i8(f["in_addr"], f["rows"], f["cols"], f["in_pitch"], 2)
        self._vm(f["out"], f["rows"], f["cols"], f["out_pitch"], 2)[...] = golden.lut(x, table).view(np.uint8)

    def _softmax(self, cmd):
        f = cmd.f
        n, has_mask = f["len"], f["flags"] & C.SM_F_HAS_MASK
        bad = (f["in_type"] > 1 or f["out_type"] != 0 or f["rows"] == 0 or n == 0 or n & 15 or n > C.MAX_LEN or f["in_pitch"] & 15
               or f["out_pitch"] & 15 or not self._addr_ok(f["in_addr"]) or not self._addr_ok(f["out"])
               or (has_mask and not self._addr_ok(f["mask"])) or f["in_shift"] > 63 or f["out_shift"] > 63)
        if bad:
            raise UnitError(2, C.ERR_BAD_FIELD)
        x = self._operand(f["in_addr"], f["rows"], n, f["in_pitch"], f["in_type"] == C.DT_I32)
        mask = self._vm_i32(f["mask"], 1, n, 0, 2)[0] if has_mask else None
        q_pos = f["q_pos"] if f["flags"] & C.SM_F_CAUSAL else None
        out = golden.softmax(x, mask, f["in_mult"], f["in_shift"], f["in_zp"], f["out_mult"], f["out_shift"], f["out_zp"], q_pos)
        self._vm(f["out"], f["rows"], n, f["out_pitch"], 2)[...] = out.view(np.uint8)
