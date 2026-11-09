"""Cycles per GEMM on the RTL for the patterns decode and prefill use (hw/sim/tb/tb_blob.sv).

Runs each pattern with the testbench's event trace and reports the steady-state interval
between consecutive GEMM completions (median over the middle of the run), so program start-up
and the final STORE of all outputs do not count. Outputs are checked against the simulator,
so the timing is of correct programs.

  decode      LOAD_WGT then GEMM per 16 output columns, one M-tile: what decode does
  reuse       the same GEMMs with two weight halves loaded once: no LOAD_WGT in the loop
  load only   the LOAD_WGTs alone: the weight DMA (interval between LOAD_WGT completions)
  prefill     LOAD_WGT then GEMM with 8 M-tiles: weights used by 128 rows

usage: rtl_gemm_timing.py [--stalls]     (--stalls: the DDR model's random stalls, as in rtl_diff)
"""
import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rtl_diff
from mini_tpu import isa
from mini_tpu.codegen import Emitter, Target
from mini_tpu.isa import C
from mini_tpu.sim import Machine

DDR = 1 << 20
PROG, WGT, PRM, ACT, OUT = 0x1000, 0x40000, 0x78000, 0x80000, 0xA0000
ACT_VM, OUT_VM = 0, 0x40000
CASES = [("decode", 512, 1, 24), ("decode", 256, 1, 24), ("reuse", 512, 1, 24), ("load only", 512, 1, 24),
         ("prefill", 512, 8, 12)]


def program(kind, n, k, m_tiles):
    em = Emitter(Target())
    rows = 16 * m_tiles
    em.load(C.REGION_INPUT, 0, ACT_VM, rows, k)
    tile = 16 * k
    gemms = 0 if kind == "load only" else n
    if kind == "reuse":
        halves = [em.load_wgt(0, k), em.load_wgt(tile, k)]
    for i in range(n):
        if kind == "reuse":
            half = halves[i & 1]
        else:
            half = em.load_wgt(i * tile, k)
        if gemms:
            em.gemm(ACT_VM, k, m_tiles, k, half, {"mode": C.EP_STORE_I32, "dst": OUT_VM + i * rows * 64, "dst_pitch": 64})
    if gemms:
        em.store(C.REGION_OUTPUT, 0, OUT_VM, gemms * rows, 64)
    em.descs.append(isa.halt())
    return isa.build_program(em.descs)


def traces(log):
    """Trace events of each program, in run order: [[(cycle, event), ...], ...]."""
    out, cur = [], []
    for line in log.splitlines():
        if line.startswith("T "):
            _, c, e = line.split()
            cur.append((int(c), e))
        elif line.startswith("RESULT "):
            out.append(cur)
            cur = []
    return out


def period(events, done):
    """Median interval between `done` events, without the first 4 and the last 2."""
    t = [c for c, e in events if e == done]
    gaps = np.diff(t)[4:-2]
    return statistics.median(gaps.tolist()) if len(gaps) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stalls", action="store_true", help="keep the DDR model's random stalls")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    img = np.zeros(DDR, dtype=np.uint8)
    img[WGT:WGT + 24 * 16 * 512] = rng.integers(0, 256, 24 * 16 * 512, dtype=np.uint8)
    img[ACT:ACT + 128 * 512] = rng.integers(0, 256, 128 * 512, dtype=np.uint8)

    runs, lines, addr = [], [f"R {r:x} {b:x}" for r, b in ((0, WGT), (1, PRM), (2, ACT), (3, OUT))], PROG
    for kind, k, m_tiles, n in CASES:
        blob = program(kind, n, k, m_tiles)
        img[addr:addr + len(blob)] = np.frombuffer(blob, dtype=np.uint8)
        runs.append((kind, k, m_tiles, n, addr))
        lines.append(f"P {addr:x} 0")
        addr += -(-len(blob) // 4096) * 4096
    assert addr <= WGT

    m = Machine(ddr_bytes=DDR)
    m.ddr[:] = img
    m.region[:4] = [WGT, PRM, ACT, OUT]
    for kind, k, m_tiles, n, at in runs:
        assert m.run(at) == C.ST_DONE and m.checker.races == 0, (kind, n, m.message)

    rtl_diff.WORK.mkdir(parents=True, exist_ok=True)
    (rtl_diff.WORK / "ddr.hex").write_text("\n".join(w.tobytes().hex() for w in img.reshape(-1, 16)[:, ::-1]) + "\n")
    (rtl_diff.WORK / "run.txt").write_text("\n".join(lines) + "\n")
    res, secs = rtl_diff.run_rtl(("" if a.stalls else "+nostall") + " +trace")
    diff = int(np.count_nonzero(rtl_diff.read_ddr_out() != m.ddr))
    bad = diff or len(res) != len(runs) or any(st != C.ST_DONE for st, _, _ in res)
    ev = traces((rtl_diff.WORK / "sim.log").read_text())

    print(f"DDR model {'with' if a.stalls else 'without'} random stalls; {len(res)} programs in {secs:.0f} s; "
          f"{diff} DDR bytes differ from the simulator")
    print(f"{'pattern':10} {'k':>4} {'M-tiles':>7} {'cycles per GEMM':>16} {'per k-row':>10}")
    for (kind, k, m_tiles, n, _), events in zip(runs, ev):
        per = period(events, "D0" if kind == "load only" else "D1")
        note = "  (per LOAD_WGT)" if kind == "load only" else ""
        print(f"{kind:10} {k:4} {m_tiles:7} {per:16.0f} {per / k:10.2f}{note}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
