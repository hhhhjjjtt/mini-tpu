"""Runs output/ in the simulator, from the files alone, the way the ARM uses them.

Prefill: golden_input.bin must give golden_output.bin. Decode: the prompt from golden_ids.bin,
then --steps tokens one at a time, must give the rows of golden_decode.bin at those positions.

usage: python -m mini_tpu.check [--steps N]
"""
import argparse
import struct
import sys

import numpy as np

from . import config, session
from .isa import C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2, help="decode steps to check (default 2)")
    a = ap.parse_args()
    c = config.load()
    d = c.output
    model, dims = session.load(d)

    hw = c.hardware
    stored = struct.unpack_from("<Q", model.prefill.program, C.HDR_COMPILER_HASH)[0]
    if stored != hw.hash:
        sys.exit(f"output was built for another hardware config: {stored:#x}, config.toml is {hw.hash:#x}; run make")

    s = session.Session(model, dims, vmem_bytes=hw.vmem_bytes)
    x = np.fromfile(d / "golden_input.bin", dtype="<i4").reshape(dims.seq, dims.d_model)
    out = s.prefill_x(x)
    expected = np.fromfile(d / "golden_output.bin", dtype="<i4").reshape(out.shape)
    diff = int(np.count_nonzero(out != expected))
    print(f"{d.name}: prefill {s.runs[0][1]} descriptors, {diff} of {out.size} outputs differ")

    ids = np.fromfile(d / "golden_ids.bin", dtype="<i4").astype(np.int64)
    g = np.fromfile(d / "golden_decode.bin", dtype="<i4")
    n, count = int(g[0]), int(g[1])
    pos = [int(p) for p in g[2:2 + count]]
    rows = dict(zip(pos, g[2 + count:].reshape(count, dims.vocab_pad)))
    got = {n - 1: s.prefill(ids[:n])}
    for p in range(n, n + a.steps):
        got[p] = s.step(ids[p], p)
    checked = [p for p in got if p in rows]
    ddiff = sum(int(np.count_nonzero(got[p] != rows[p])) for p in checked)
    diff += ddiff
    print(f"decode: prompt {n}, positions {checked}: {ddiff} of {len(checked) * dims.vocab_pad} logits differ")

    races, rules, uninit = (sum(r[i] for r in s.runs) for i in (2, 3, 4))
    print(f"races {races}, rule violations {rules}, uninitialized reads {uninit}")
    sys.exit(1 if diff or races or rules else 0)


if __name__ == "__main__":
    main()
