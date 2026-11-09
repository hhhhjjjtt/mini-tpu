"""Random programs on the simulator and on the RTL (hw/sim/tb/tb_blob.sv), DDR compared byte for byte.

Covers what the board tests do not: every VEC_ELTWISE op, type and broadcast, all epilogue modes,
the bias-row form of ACC_I32 (partial_pitch = 0), and extreme multipliers, shifts and clamps.

usage: rtl_diff.py [--seed N] [--scale F]     (scale 1.0 is about 5 minutes)
"""
import argparse
import math
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mini_tpu import golden, isa
from mini_tpu.isa import C
from mini_tpu.sim import Machine

ROOT = Path(__file__).resolve().parents[3]
SIM_DIR = ROOT / "hw" / "sim"
WORK = SIM_DIR / "build" / "blob"            # tb_blob.sv input and DDR dump
DDR = 1 << 20
REGIONS = {0: 0x40000, 1: 0x50000, 2: 0x60000, 3: 0xA0000}      # DDR model errors from 0xF0000
LOAD, MATRIX, VECTOR, STORE = range(4)
LOG2E = 1.4426950408889634


class Image:
    def __init__(self, rng):
        self.rng = rng
        self.ddr = np.zeros(DDR, dtype=np.uint8)
        self.top = dict(REGIONS)
        self.prog_top = 0x1000
        self.programs = []                      # (address, case name)

    def alloc(self, region, n):
        off = self.top[region] - REGIONS[region]
        self.top[region] += (n + 15) & ~15
        limit = {0: 0x50000, 1: 0x60000, 2: 0xA0000, 3: 0xE0000}[region]
        assert self.top[region] <= limit, f"region {region} full"
        return off

    def put(self, region, data):
        data = np.frombuffer(bytes(data), dtype=np.uint8)
        off = self.alloc(region, data.size)
        self.ddr[REGIONS[region] + off:REGIONS[region] + off + data.size] = data
        return off

    def program(self, name, descs):
        blob = isa.build_program(descs)
        addr = self.prog_top
        self.prog_top += (len(blob) + 63) & ~63
        assert self.prog_top <= 0x40000
        self.ddr[addr:addr + len(blob)] = np.frombuffer(blob, dtype=np.uint8)
        self.programs.append((addr, name))


def i8(rng, *shape):
    return rng.integers(-128, 128, shape, dtype=np.int8)


def i32(rng, *shape):
    bits = int(rng.choice([8, 16, 24, 30, 31]))
    return rng.integers(-(1 << bits), 1 << bits, shape, dtype=np.int64).astype(np.int32)


def rescale(rng):
    """A (mult, shift) pair: identity, ordinary, tiny shift, or a negative multiplier."""
    style = rng.integers(4)
    if style == 0:
        return 1 << 30, 30
    if style == 1:
        return int(rng.integers(1 << 30, 1 << 31)), int(rng.integers(20, 46))
    if style == 2:
        return int(rng.integers(-5, 6)), int(rng.integers(0, 4))
    return -int(rng.integers(1 << 28, 1 << 31)), int(rng.integers(30, 63))


def eltwise_case(img):
    rng = img.rng
    rows, cols = int(rng.integers(1, 5)), 16 * int(rng.integers(1, 5))
    at, bt, ot = (int(rng.integers(2)) for _ in range(3))
    op, bc = int(rng.integers(6)), int(rng.integers(3))
    width = lambda t: cols * (4 if t else 1)
    pad = lambda: 16 * int(rng.integers(3))
    ap, bp, op_pitch = width(at) + pad(), width(bt) + pad(), width(ot) + pad()

    def tensor(t, n_rows, pitch):
        buf = np.zeros((n_rows, pitch), dtype=np.uint8)
        data = i32(rng, n_rows, cols) if t else i8(rng, n_rows, cols)
        buf[:, :width(t)] = data.view(np.uint8)
        return buf

    (am, ash), (bm, bsh), (om, osh) = rescale(rng), rescale(rng), rescale(rng)
    p = bytearray(64)
    struct.pack_into("<iiB", p, C.EWP_A_ZP, int(rng.integers(-100, 100)), am, ash)
    struct.pack_into("<iiB", p, C.EWP_B_ZP, int(rng.integers(-100, 100)), bm, bsh)
    struct.pack_into("<iBB", p, C.EWP_OUT_MULT, om, osh, int(rng.integers(0, 41)))
    lo, hi = sorted(int(x) for x in rng.integers(-128, 128, 2))
    struct.pack_into("<bbbB", p, C.EWP_OUT_ZP, int(rng.integers(-128, 128)), lo, hi, int(rng.integers(2)))

    va, vb, vp, vo = 0x00000, 0x40000, 0x80000, 0xC0000     # separate banks
    d = [isa.load(1, img.put(1, p), vp, 1, 64, 64, 64),
         isa.load(2, img.put(2, tensor(at, rows, ap)), va, rows, ap, ap, ap)]
    if op != C.EW_PASS_A:
        if bc == C.EW_BCAST_FULL:
            d.append(isa.load(2, img.put(2, tensor(bt, rows, bp)), vb, rows, bp, bp, bp))
        elif bc == C.EW_BCAST_ROW:
            d.append(isa.load(2, img.put(2, tensor(bt, 1, bp)), vb, 1, bp, bp, bp))
        else:
            d.append(isa.load(2, img.put(2, i32(rng, 4)), vb, 1, 16, 16, 16))
    d.append(isa.wait(isa.eltwise(op, at, bt, ot, bc, rows, cols, va, vb, vo, ap, bp, op_pitch, vp), LOAD, 0))
    w = width(ot)
    d.append(isa.wait(isa.store(3, img.alloc(3, rows * w), vo, rows, w, w, op_pitch), VECTOR, 0))
    img.program(f"eltwise op={op} a={at} b={bt} out={ot} bcast={bc} {rows}x{cols}", d + [isa.halt()])


def requant_table(rng):
    t = bytearray(256)
    for j in range(16):
        struct.pack_into("<iiB", t, 16 * j, int(rng.integers(-10000, 10001)), int(rng.integers(1 << 30, 1 << 31)),
                         int(rng.integers(30, 41)))
    return t


def epilogue_case(img):
    rng = img.rng
    variant = int(rng.integers(4))
    m_tiles, rows = int(rng.integers(1, 3)), 0
    rows = 16 * m_tiles
    chunks = 1 if variant < 2 else int(rng.integers(2, 4))
    ks = [16 * int(rng.integers(1, 5)) for _ in range(chunks)]
    ktot = sum(ks)
    flags = int(rng.integers(4))
    zp, (lo, hi) = int(rng.integers(-128, 128)), sorted(int(x) for x in rng.integers(-128, 128, 2))
    va, vt, vb, vp, vo = 0x00000, 0x80000, 0x80400, 0xC0000, 0x100000

    d = [isa.load(1, img.put(1, requant_table(rng)), vt, 1, 256, 256, 256),
         isa.load(2, img.put(2, i8(rng, rows, ktot)), va, rows, ktot, ktot, ktot)]
    if variant == 3:
        d.append(isa.load(2, img.put(2, i32(rng, 16)), vb, 1, 64, 64, 64))
    i32_out = variant in (1, 3)
    pitch = 64 if i32_out else 16
    koff = 0
    for c, k in enumerate(ks):
        half = c & 1
        d.append(isa.wait(isa.load_wgt(0, img.put(0, i8(rng, k, 16)), half, k), MATRIX, 1))
        d.append(isa.wait(isa.gemm(va + koff, ktot, m_tiles, k, half), LOAD, 0))
        last = c == chunks - 1
        if variant == 0:
            e = isa.epilogue(C.EP_REQUANT_I8, flags, zp, lo, hi, rows, 16, vo, vt)
        elif variant == 1:
            e = isa.epilogue(C.EP_STORE_I32, flags, zp, lo, hi, rows, 64, vo, vt)
        elif variant == 2:                                  # K-split ending in requant
            mode = C.EP_STORE_I32 if c == 0 else C.EP_ACC_REQUANT_I8 if last else C.EP_ACC_I32
            e = isa.epilogue(mode, flags, zp, lo, hi, rows, 16 if last else 64, vo if last else vp, vt, vp, 64)
        else:                                               # bias row: first chunk adds it, pitch 0
            first = c == 0
            e = isa.epilogue(C.EP_ACC_I32, flags, zp, lo, hi, rows, 64, vo, vt, vb if first else vo, 0 if first else 64)
        d.append(isa.wait(e, LOAD, 0))
        koff += k
    w = 64 if i32_out else 16
    d.append(isa.wait(isa.store(3, img.alloc(3, rows * w), vo, rows, w, w, pitch), VECTOR, 0))
    img.program(f"epilogue variant={variant} flags={flags} k={ks} m={m_tiles}", d + [isa.halt()])


def softmax_case(img):
    rng = img.rng
    rows, n = int(rng.integers(1, 5)), 16 * int(rng.integers(1, 9))
    is32, has_mask, causal = bool(rng.integers(2)), bool(rng.integers(2)), bool(rng.integers(2))
    q_pos = int(rng.integers(0, n + 1))
    scale = float(rng.choice([0.0015, 0.02, 0.08, 0.3])) if not is32 else float(rng.choice([0.0015, 0.0002]))
    in_mult, in_sh = golden.real_to_ms(scale * LOG2E * 65536.0)
    out_inv, out_zp = (127.0, 0) if rng.integers(2) else (256.0, -128)
    out_mult, out_sh = golden.real_to_ms(out_inv / 65536.0)
    in_zp = int(rng.integers(-5, 6))
    data = (rng.integers(-20000, 20001, (rows, n), dtype=np.int64).astype(np.int32) if is32 else i8(rng, rows, n))
    pitch = n * (4 if is32 else 1) + 16 * int(rng.integers(3))
    buf = np.zeros((rows, pitch), dtype=np.uint8)
    buf[:, :data.nbytes // rows] = data.view(np.uint8)
    vi, vm, vo = 0x00000, 0x40000, 0x80000
    d = [isa.load(2, img.put(2, buf), vi, rows, pitch, pitch, pitch)]
    if has_mask:
        kinds = rng.integers(0, 4, n)
        real = np.where(kinds == 0, -10000.0, np.where(kinds == 1, -rng.random(n) * 4, 0.0))
        mask = np.floor(real * LOG2E * 65536.0 + 0.5).astype(np.int32)
        d.append(isa.load(2, img.put(2, mask), vm, 1, n * 4, 0, 0))
    d.append(isa.wait(isa.softmax(is32, rows, n, pitch, vi, vo, n, in_mult, in_sh, in_zp, out_mult, out_sh, out_zp, vm, has_mask,
                                  causal, q_pos), LOAD, 0))
    d.append(isa.wait(isa.store(3, img.alloc(3, rows * n), vo, rows, n, n, n), VECTOR, 0))
    img.program(f"softmax i32={is32} mask={has_mask} causal={causal}/{q_pos} {rows}x{n} scale={scale}", d + [isa.halt()])


def layernorm_case(img):
    rng = img.rng
    rows = int(rng.integers(1, 5))
    n = int(rng.choice([16, 48, 64, 128, 768, 1024]))
    is32 = bool(rng.integers(2))
    if is32:
        bits = int(rng.choice([4, 10, 16, 20, 22]))         # 22 saturates
        data = rng.integers(-(1 << bits), 1 << bits, (rows, n), dtype=np.int64).astype(np.int32)
        data += np.int32(rng.integers(-300, 300))
    else:
        data = i8(rng, rows, n)
    if rng.integers(4) == 0:
        data[0] = data[0, 0]                                # variance 0: eps alone sets the scale
    width = n * (4 if is32 else 1)
    pitch = width + 16 * int(rng.integers(3))
    op_pitch = n + 16 * int(rng.integers(2))
    buf = np.zeros((rows, pitch), dtype=np.uint8)
    buf[:, :width] = data.view(np.uint8)
    inv_m, inv_s, eps, c_m, c_s = golden.layernorm_params(n, float(rng.choice([1e-4, 0.01, 0.1])), float(rng.choice([0.03, 0.1, 0.3])))
    vi, vo = 0x00000, 0x80000
    d = [isa.load(2, img.put(2, buf), vi, rows, width, pitch, pitch),
         isa.wait(isa.layernorm(is32, rows, n, pitch, vi, vo, op_pitch, inv_m, inv_s, eps, c_m, c_s, int(rng.integers(-5, 6))), LOAD, 0),
         isa.wait(isa.store(3, img.alloc(3, rows * n), vo, rows, n, n, op_pitch), VECTOR, 0)]
    img.program(f"layernorm i32={is32} {rows}x{n}", d + [isa.halt()])


def lut_case(img):
    rng = img.rng
    rows, cols = int(rng.integers(1, 5)), 16 * int(rng.integers(1, 7))
    pitch, op_pitch = cols + 16 * int(rng.integers(3)), cols + 16 * int(rng.integers(3))
    buf = np.zeros((rows, pitch), dtype=np.uint8)
    buf[:, :cols] = i8(rng, rows, cols).view(np.uint8)
    vi, vt, vo = 0x00000, 0x40000, 0x80000
    d = [isa.load(1, img.put(1, i8(rng, 256)), vt, 1, 256, 256, 256),
         isa.load(2, img.put(2, buf), vi, rows, pitch, pitch, pitch),
         isa.wait(isa.vec_lut(rows, cols, vi, vo, pitch, op_pitch, vt), LOAD, 0),
         isa.wait(isa.store(3, img.alloc(3, rows * cols), vo, rows, cols, cols, op_pitch), VECTOR, 0)]
    img.program(f"lut {rows}x{cols}", d + [isa.halt()])


def attention_case(img):
    """K^T and V through LOAD_WGT_VM, in i32 and i8 steps."""
    rng = img.rng
    s, dh = 16 * int(rng.integers(1, 3)), 16 * int(rng.integers(1, 5))
    vq, vk, vv, vs, vo, vpr = 0x00000, 0x40000, 0x80000, 0xC0000, 0x100000, 0x140000
    d = [isa.load(2, img.put(2, i8(rng, 16, dh)), vq, 16, dh, dh, dh),
         isa.load(2, img.put(2, i8(rng, s, dh)), vk, s, dh, dh, dh),
         isa.load(2, img.put(2, i8(rng, s, 16)), vv, s, 16, 16, 16),
         isa.load(2, img.put(2, i8(rng, 16, s)), vpr, 16, s, s, s)]
    for b in range(s // 16):
        d.append(isa.wait(isa.load_wgt_vm(vk + 16 * b * dh, dh, dh, b & 1, transpose=True), LOAD, 0))
        d.append(isa.gemm(vq, dh, 1, dh, b & 1))
        d.append(isa.epilogue(C.EP_STORE_I32, 0, 0, 0, 0, 16, s * 4, vs + 64 * b, 0))
    d.append(isa.load_wgt_vm(vv, 16, s, 0))
    d.append(isa.gemm(vpr, s, 1, s, 0))
    d.append(isa.epilogue(C.EP_STORE_I32, 0, 0, 0, 0, 16, 64, vo, 0))
    d.append(isa.wait(isa.store(3, img.alloc(3, 16 * s * 4), vs, 16, s * 4, s * 4, s * 4), VECTOR, 1))
    d.append(isa.wait(isa.store(3, img.alloc(3, 16 * 64), vo, 16, 64, 64, 64), VECTOR, 0))
    img.program(f"attention s={s} d={dh}", d + [isa.halt()])


def loop_case(img):
    """Nested LOOP with region strides, reads and writes walking through DDR."""
    rng = img.rng
    src = img.put(2, rng.integers(0, 256, 64 * 6, dtype=np.uint8))
    dst = img.alloc(3, 64 * 6)
    copy = [isa.wait(isa.load(2, src, 0, 1, 64, 64, 64), STORE, 0), isa.wait(isa.store(3, dst, 0, 1, 64, 64, 64), LOAD, 0)]
    d = [isa.loop(2, 3, {2: 192, 3: 192}), isa.loop(3, 2, {2: 64, 3: 64})] + copy
    img.program("nested LOOP", d + [isa.halt()])


def chunk_case(img):
    """Enough descriptors to need NEXT_CHUNK."""
    rng = img.rng
    src = img.put(2, rng.integers(0, 256, 16 * 300, dtype=np.uint8))
    d = [isa.load(2, src + 16 * i, 16 * i, 1, 16, 16, 16) for i in range(300)]
    d.append(isa.wait(isa.store(3, img.alloc(3, 16 * 300), 0, 300, 16, 16, 16), LOAD, 0))
    img.program("two chunks", d + [isa.halt()])


def build(seed, scale):
    img = Image(np.random.default_rng(seed))
    plan = [(eltwise_case, 30), (epilogue_case, 12), (softmax_case, 12), (layernorm_case, 12), (lut_case, 6), (attention_case, 3), (loop_case, 1), (chunk_case, 1)]
    for case, n in plan:
        for _ in range(max(1, round(n * scale))):
            case(img)
    return img


def simulate(img):
    m = Machine(ddr_bytes=DDR)
    m.ddr[:] = img.ddr
    for r, base in REGIONS.items():
        m.region[r] = base
    results = []
    for addr, name in img.programs:
        status = m.run(addr)
        c = m.checker
        if status != C.ST_DONE or c.races or c.rule_violations or c.count.get("UNINIT"):
            sys.exit(f"bad test program '{name}': status {status:#x}, {[str(f) for f in c.findings[:5]]}")
        results.append((status, m.desc_count))
    return m.ddr, results


def write_blob(img):
    WORK.mkdir(parents=True, exist_ok=True)
    words = img.ddr.reshape(-1, 16)[:, ::-1]
    (WORK / "ddr.hex").write_text("\n".join(w.tobytes().hex() for w in words) + "\n")
    lines = [f"R {r:x} {b:x}" for r, b in REGIONS.items()] + [f"P {a:x} 0" for a, _ in img.programs]
    (WORK / "run.txt").write_text("\n".join(lines) + "\n")


def read_ddr_out():
    lines = (l.split("//")[0].strip() for l in (WORK / "ddr_out.hex").read_text().splitlines())
    rows = [l for l in lines if l and not l.startswith("@")]
    assert len(rows) == DDR // 16, len(rows)
    return np.frombuffer(bytes.fromhex("".join(rows)), dtype=np.uint8).reshape(-1, 16)[:, ::-1].ravel()


def run_rtl():
    rel = WORK.relative_to(SIM_DIR).as_posix()
    t = time.time()
    r = subprocess.run(["make", "-s", "sim_blob", f"BLOB={rel}"], cwd=SIM_DIR, capture_output=True, text=True)
    out = r.stdout + r.stderr
    (WORK / "sim.log").write_text(out)
    if "TB DONE" not in out:
        sys.exit("RTL run failed:\n" + "\n".join(out.splitlines()[-15:]))
    res = [(int(s, 16), int(cyc), int(dc)) for s, cyc, dc in re.findall(r"RESULT \S+ (\S+) (\d+) (\d+)", out)]
    return res, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    img = build(a.seed, a.scale)
    sim_ddr, sim_res = simulate(img)
    write_blob(img)
    rtl_res, secs = run_rtl()
    rtl_ddr = read_ddr_out()

    bad = 0
    for (addr, name), (st, dc), (rst, cyc, rdc) in zip(img.programs, sim_res, rtl_res):
        if st != rst or dc != rdc:
            print(f"STATUS DIFF {name}: sim {st:#x}/{dc} rtl {rst:#x}/{rdc}")
            bad += 1
    if len(rtl_res) != len(img.programs):
        print(f"RTL ran {len(rtl_res)} of {len(img.programs)} programs")
        bad += 1
    diff = np.flatnonzero(sim_ddr != rtl_ddr)
    print(f"seed {a.seed}: {len(img.programs)} programs, {sum(r[1] for r in rtl_res)} RTL cycles, {secs:.0f} s, "
          f"{img.top[3] - REGIONS[3]} output bytes, {diff.size} DDR bytes differ, {bad} status differences")
    if diff.size:
        print("first differences:", [(hex(int(i)), int(sim_ddr[i]), int(rtl_ddr[i])) for i in diff[:8]])
    sys.exit(1 if diff.size or bad else 0)


if __name__ == "__main__":
    main()
