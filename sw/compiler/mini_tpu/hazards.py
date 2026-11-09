"""Static race checker for programs (INTERFACE_SPEC.md §4.2).

The units run concurrently, so every RAW, WAR and WAW pair must be ordered by waits or by a
unit's own in-order execution; hardware does not check. Each command gets a vector clock:
clock[u] = how many commands of unit u are guaranteed done before it starts. An access is
safe if every earlier conflicting access is covered by that clock. Issue order and wait
targets do not depend on timing, so one pass over the issue stream proves the race-freedom
of every possible schedule.
"""
from dataclasses import dataclass

import numpy as np

from .isa import C

WEIGHT_ROWS = C.MAX_K


@dataclass
class Finding:
    kind: str           # RAW, WAR, WAW, UNINIT, RULE
    space: str          # vmem, weights, ddr, program
    where: int          # byte address, half * 512 + row, or -1
    earlier: tuple      # (unit, seq) or None
    later: tuple        # (unit, seq)
    note: str = ""

    def __str__(self):
        return f"{self.kind} {self.space} @0x{self.where:x}: {self.earlier} -> {self.later} {self.note}".rstrip()


def _words(addr, rows, row_bytes, pitch):
    """vmem word indices of a 2D block."""
    w0, n, p = addr >> 4, max(row_bytes >> 4, 1), pitch >> 4
    return (w0 + np.arange(rows, dtype=np.int64)[:, None] * p + np.arange(n, dtype=np.int64)).ravel()


class Checker:
    MAX_FINDINGS = 200

    def __init__(self, vmem_bytes, written):
        """written: bool per word, shared across programs because vmem keeps its contents."""
        self.nv = vmem_bytes >> 4
        n = self.nv + 2 * WEIGHT_ROWS
        self.last_w = np.zeros((4, n), dtype=np.int32)      # seq of the newest writer, per unit
        self.last_r = np.zeros((4, n), dtype=np.int32)
        self.written = written
        self.findings = []
        self.count = {}
        self.end = [[] for _ in range(4)]                   # end clock of each command, per unit
        self.barrier = [0, 0, 0, 0]
        self.prev_matrix = None                             # (opcode, act clock)
        self.pending_gemm = []                              # end clocks of GEMMs without an epilogue
        self.ddr_w, self.ddr_r = {}, {}                     # 1 MB bucket -> access records

    def _add(self, kind, space, where, earlier, later, note=""):
        self.count[kind] = self.count.get(kind, 0) + 1
        if len(self.findings) < self.MAX_FINDINGS:
            self.findings.append(Finding(kind, space, where, earlier, later, note))

    @property
    def races(self):
        return sum(self.count.get(k, 0) for k in ("RAW", "WAR", "WAW"))

    @property
    def rule_violations(self):
        return self.count.get("RULE", 0)

    def fence(self):
        """FENCE and BREAK: every command issued so far is done before any later one starts."""
        self.barrier = [len(e) for e in self.end]

    # ---- clocks ----
    def _clocks(self, cmd):
        """Returns (start clock, act clock). The act clock is for GEMM activation reads, which
        cross-GEMM prefill may start before the previous GEMM ends (§10.5)."""
        waited = list(self.barrier)
        for v in range(4):
            t = cmd.targets[v]
            if t:
                waited = [max(a, b) for a, b in zip(waited, self.end[v][t - 1])]
        u, seq = cmd.unit, cmd.seq
        prev_end = self.end[u][-1] if seq > 1 else [0, 0, 0, 0]
        start = [max(a, b) for a, b in zip(waited, prev_end)]
        start[u] = seq - 1
        act = start
        if cmd.op == C.OP_GEMM and self.prev_matrix and self.prev_matrix[0] == C.OP_GEMM:
            act = [max(a, b) for a, b in zip(waited, self.prev_matrix[1])]
            act[u] = seq - 1
        return start, act

    def _rules(self, cmd, me):
        f = cmd.f
        if f["wait_mask"] >> cmd.unit & 1:
            self._add("RULE", "program", -1, None, me, "unit waits on itself (rule 2)")
        if cmd.unit == 2:
            if f["wait_mask"] >> 1 & 1:
                self._add("RULE", "program", -1, None, me, "VECTOR waits on MATRIX (rule 4)")
            if self.pending_gemm and cmd.op != C.OP_VEC_EPILOGUE:
                self._add("RULE", "program", -1, None, me, "VEC op between GEMM and its epilogue (rule 3)")
            if cmd.op == C.OP_VEC_EPILOGUE and not self.pending_gemm:
                self._add("RULE", "program", -1, None, me, "epilogue before its GEMM (rule 3)")

    # ---- per-word checks ----
    def _read(self, idx, clock, me, space="vmem"):
        for u in range(4):
            bad = self.last_w[u, idx] > clock[u]
            if bad.any():
                i = idx[np.argmax(bad)]
                self._add("RAW", space, self._where(i), (u, int(self.last_w[u, i])), me)
        if not self.written[idx].all():
            i = idx[np.argmin(self.written[idx])]
            self._add("UNINIT", space, self._where(i), None, me)
        self.last_r[me[0], idx] = me[1]

    def _write(self, idx, clock, me, space="vmem"):
        clock = list(clock)
        clock[me[0]] = me[1]                                # in place: a command may read what it writes
        for u in range(4):
            for arr, kind in ((self.last_w, "WAW"), (self.last_r, "WAR")):
                bad = arr[u, idx] > clock[u]
                if bad.any():
                    i = idx[np.argmax(bad)]
                    self._add(kind, space, self._where(i), (u, int(arr[u, i])), me)
        self.last_w[me[0], idx] = me[1]
        self.written[idx] = True

    def _where(self, i):
        return int(i) << 4 if i < self.nv else int(i) - self.nv

    def _wrows(self, half, k):
        return self.nv + half * WEIGHT_ROWS + np.arange(k, dtype=np.int64)

    # ---- DDR: 2D blocks kept as records; only stores can conflict ----
    @staticmethod
    def _overlap(a, b):
        if a[1] <= b[0] or b[1] <= a[0]:
            return False
        sa = a[0] + np.arange(a[2], dtype=np.int64) * a[4]
        sb = b[0] + np.arange(b[2], dtype=np.int64) * b[4]
        return bool(((sa[:, None] < sb[None, :] + b[3]) & (sb[None, :] < sa[:, None] + a[3])).any())

    def _ddr_check(self, table, kind, rec, buckets, clock, me):
        for bucket in buckets:
            for old in table.get(bucket, ()):
                if old[5] > clock[old[6]] and self._overlap(rec, old):
                    self._add(kind, "ddr", rec[0], (old[6], old[5]), me)

    def _ddr_access(self, addr, rows, row_bytes, stride, clock, me, write):
        rec = (addr, addr + (rows - 1) * stride + row_bytes, rows, row_bytes, stride)
        buckets = range(rec[0] >> 20, ((rec[1] - 1) >> 20) + 1)
        if write:
            self._ddr_check(self.ddr_r, "WAR", rec, buckets, clock, me)
            self._ddr_check(self.ddr_w, "WAW", rec, buckets, clock, me)
        else:
            self._ddr_check(self.ddr_w, "RAW", rec, buckets, clock, me)
        table = self.ddr_w if write else self.ddr_r
        for bucket in buckets:
            table.setdefault(bucket, []).append(rec + (me[1], me[0]))

    # ---- one command, in issue order ----
    def issue(self, cmd):
        me = (cmd.unit, cmd.seq)
        f, op = cmd.f, cmd.op
        self._rules(cmd, me)
        start, act = self._clocks(cmd)
        end = list(start)
        end[cmd.unit] = cmd.seq

        if op in (C.OP_LOAD, C.OP_STORE):
            idx = _words(f["vmem"], f["rows"], f["row_bytes"], f["vmem_stride"])
            if idx.size and idx.max() < self.nv:
                write = op == C.OP_STORE
                self._ddr_access(cmd.addr, f["rows"], f["row_bytes"], f["ddr_stride"], start, me, write)
                (self._read if write else self._write)(idx, start, me)
        elif op == C.OP_LOAD_WGT:
            self._ddr_access(cmd.addr, 1, 16 * f["k"], 0, start, me, False)
            self._write(self._wrows(f["half"] & 1, min(f["k"], WEIGHT_ROWS)), start, me, "weights")
        elif op == C.OP_GEMM:
            k = min(f["k"], WEIGHT_ROWS)
            self._read(_words(f["act"], 16 * f["m_tiles"], f["k"], f["act_pitch"]), act, me)
            self._read(self._wrows(f["half"] & 1, k), start, me, "weights")
            self.pending_gemm.append(end)
        elif op == C.OP_LOAD_WGT_VM:
            k = min(f["k"], WEIGHT_ROWS)
            if f["flags"] & C.WV_F_TRANSPOSE:
                src = _words(f["src"], 16, f["k"], f["src_pitch"])
            else:
                src = _words(f["src"], f["k"], 16, f["src_pitch"])
            self._read(src, start, me)
            self._write(self._wrows(f["half"] & 1, k), start, me, "weights")
        elif op == C.OP_VEC_EPILOGUE:
            i32_out = f["mode"] in (C.EP_STORE_I32, C.EP_ACC_I32)
            if not i32_out:                                 # the i32 modes fetch the table but ignore it
                self._read(_words(f["params"], 1, 16 if f["flags"] & C.EP_F_PER_TENSOR else 256, 0), start, me)
            if f["mode"] in (C.EP_ACC_I32, C.EP_ACC_REQUANT_I8):
                self._read(_words(f["partial"], f["rows"], 64, f["partial_pitch"]), start, me)
            self._write(_words(f["dst"], f["rows"], 64 if i32_out else 16, f["dst_pitch"]), start, me)
            if self.pending_gemm:
                end = [max(a, b) for a, b in zip(end, self.pending_gemm.pop(0))]
        elif op == C.OP_VEC_ELTWISE:
            ab, bb = (f["types"] & 0xF) == C.DT_I32, (f["types"] >> 4) == C.DT_I32
            self._read(_words(f["params"], 1, 64, 0), start, me)
            self._read(_words(f["a"], f["rows"], f["cols"] * (4 if ab else 1), f["a_pitch"]), start, me)
            if f["op"] != C.EW_PASS_A:
                n = f["cols"] * (4 if bb else 1)
                if f["b_bcast"] == C.EW_BCAST_SCALAR:
                    self._read(_words(f["b"], 1, 16, 0), start, me)
                elif f["b_bcast"] == C.EW_BCAST_ROW:
                    self._read(_words(f["b"], 1, n, 0), start, me)
                else:
                    self._read(_words(f["b"], f["rows"], n, f["b_pitch"]), start, me)
            self._write(_words(f["out"], f["rows"], f["cols"] * (4 if f["out_type"] == C.DT_I32 else 1), f["out_pitch"]),
                        start, me)
        elif op == C.OP_VEC_SOFTMAX:
            e = 4 if f["in_type"] == C.DT_I32 else 1
            self._read(_words(f["in_addr"], f["rows"], f["len"] * e, f["in_pitch"]), start, me)
            if f["flags"] & C.SM_F_HAS_MASK:
                self._read(_words(f["mask"], 1, f["len"] * 4, 0), start, me)
            self._write(_words(f["out"], f["rows"], f["len"], f["out_pitch"]), start, me)
        elif op == C.OP_VEC_LAYERNORM:
            e = 4 if f["in_type"] == C.DT_I32 else 1
            self._read(_words(f["in_addr"], f["rows"], f["len"] * e, f["in_pitch"]), start, me)
            self._write(_words(f["out"], f["rows"], f["len"], f["out_pitch"]), start, me)
        elif op == C.OP_VEC_LUT:
            self._read(_words(f["table"], 1, 256, 0), start, me)
            self._read(_words(f["in_addr"], f["rows"], f["cols"], f["in_pitch"]), start, me)
            self._write(_words(f["out"], f["rows"], f["cols"], f["out_pitch"]), start, me)

        self.end[cmd.unit].append(end)
        if cmd.unit == 1:
            self.prev_matrix = (op, act)
