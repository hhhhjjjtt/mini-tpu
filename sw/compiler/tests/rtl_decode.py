"""Decode on the RTL (hw/sim/tb/tb_blob.sv), with the tiny GPT: two prefills, each followed by decode
steps, crossing the 16, 32 and 64 buckets. All of DDR must match the simulator, every logit the reference.

Each run gets its own input and output area, so DDR keeps every result; region 6 moves per token.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rtl_diff
import test_decode
from mini_tpu import codegen, model
from mini_tpu.isa import C
from mini_tpu.session import mask_row
from mini_tpu.sim import Machine

DDR = 1 << 20
RUNS = [(14, [14, 15, 16]), (31, [31, 32])]      # (prompt length, decode positions)


def main():
    q, ids, c = test_decode.build()
    cfg, pre = q.cfg, c.prefill
    ref = model.forward(q, q.embed(ids))

    img = np.zeros(DDR, dtype=np.uint8)
    top = 0x1000

    def put(data, size=None):
        nonlocal top
        addr = top
        if data is not None:
            img[addr:addr + len(data)] = np.frombuffer(data, dtype=np.uint8)
        top += -(-(size or len(data)) // 4096) * 4096
        return addr

    prog = {"prefill": put(pre.program), **{L: put(x.program) for L, x in c.decode.items()}}
    base = {codegen.R_WGT: put(pre.weights), codegen.R_PRM: put(pre.params), codegen.R_KV: put(None, c.kv_bytes)}
    runs = []                                   # (program, region 2, region 3, region 6 or None, position)
    for n, steps in RUNS:
        x = q.embed(np.concatenate([ids[:n], np.zeros(cfg.seq - n, dtype=np.int64)]))
        runs.append(("prefill", put(x.astype("<i4").tobytes()), put(None, pre.output_bytes), None, n - 1))
        for p in steps:
            L = min(L for L in c.decode if L > p)
            data = q.embed(ids[p:p + 1], p)[0].astype("<i4").tobytes() + mask_row(c.ctx, p).tobytes()
            runs.append((L, put(data), put(None, c.decode[L].output_bytes), base[codegen.R_KV] + p * cfg.d_model, p))
    assert top <= DDR, f"needs {top} bytes of DDR"

    m = Machine(ddr_bytes=DDR)
    m.ddr[:] = img
    lines = [f"R {r:x} {a:x}" for r, a in base.items()]
    for name, r2, r3, r6, p in runs:
        regs = {codegen.R_IN: r2, codegen.R_OUT: r3} | ({codegen.R_KV_ROW: r6} if r6 is not None else {})
        lines += [f"R {r:x} {a:x}" for r, a in regs.items()] + [f"P {prog[name]:x} 0"]
        for r, a in (base | regs).items():
            m.region[r] = a
        status = m.run(prog[name])
        assert status == C.ST_DONE and m.checker.races == 0, (name, p, m.message)

    rtl_diff.WORK.mkdir(parents=True, exist_ok=True)
    (rtl_diff.WORK / "ddr.hex").write_text("\n".join(w.tobytes().hex() for w in img.reshape(-1, 16)[:, ::-1]) + "\n")
    (rtl_diff.WORK / "run.txt").write_text("\n".join(lines) + "\n")
    res, secs = rtl_diff.run_rtl()
    ddr = rtl_diff.read_ddr_out()

    bad = int(any(st != C.ST_DONE for st, _, _ in res)) or len(res) != len(runs)
    for (name, r2, r3, r6, p), (st, cycles, descs) in zip(runs, res):
        rows = cfg.seq if name == "prefill" else 1
        out = ddr[r3:r3 + rows * cfg.vocab_pad * 4].view("<i4").reshape(rows, -1)[p if name == "prefill" else 0]
        diff = int(np.count_nonzero(out != ref[p]))
        bad |= diff > 0
        print(f"  {'prefill' if name == 'prefill' else f'decode_{name}':12} position {p:2}: status {st:#x}, {cycles:7} cycles, "
              f"{descs:4} descriptors, {diff} of {cfg.vocab_pad} logits differ from the reference")
    diff = int(np.count_nonzero(ddr != m.ddr))
    print(f"RTL: {len(res)} runs in {secs:.0f} s; {diff} DDR bytes differ from the simulator")
    sys.exit(1 if bad or diff else 0)


if __name__ == "__main__":
    main()
