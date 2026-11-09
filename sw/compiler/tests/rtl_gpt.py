"""The tiny GPT on the RTL (hw/sim/tb/tb_blob.sv): all of DDR must match the simulator, logits the reference."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rtl_diff
import test_tiny_gpt as tiny
from mini_tpu import model
from mini_tpu.isa import C


def main():
    _, q, ids, c = tiny.build()
    m, status, out = tiny.simulate(q, ids, c)
    assert status == C.ST_DONE, m.message
    ref = model.forward(q, q.embed(ids))
    assert (out == ref).all()

    prog, *regions = tiny.BASES
    rtl_diff.WORK.mkdir(parents=True, exist_ok=True)
    start = np.zeros(1 << 20, dtype=np.uint8)
    for addr, data in ((prog, c.program), (regions[0], c.weights), (regions[1], c.params), (regions[2], q.embed(ids).astype("<i4").tobytes())):
        start[addr:addr + len(data)] = np.frombuffer(data, dtype=np.uint8)
    words = start.reshape(-1, 16)[:, ::-1]
    (rtl_diff.WORK / "ddr.hex").write_text("\n".join(w.tobytes().hex() for w in words) + "\n")
    lines = [f"R {r:x} {b:x}" for r, b in zip((0, 1, 2, 3, C.REGION_KV), regions)] + [f"P {prog:x} 0"]
    (rtl_diff.WORK / "run.txt").write_text("\n".join(lines) + "\n")

    res, secs = rtl_diff.run_rtl()
    (st, cycles, descs), = res
    ddr = rtl_diff.read_ddr_out()
    rtl_out = ddr[regions[3]:regions[3] + c.output_bytes].view("<i4").reshape(out.shape)
    diff = np.count_nonzero(ddr != m.ddr)
    print(f"RTL: status {st:#x}, {cycles} cycles, {descs} descriptors, {secs:.0f} s; {diff} DDR bytes differ from the simulator, "
          f"{np.count_nonzero(rtl_out != ref)} of {ref.size} logits differ from the reference")
    sys.exit(1 if st != C.ST_DONE or diff or (rtl_out != ref).any() else 0)


if __name__ == "__main__":
    main()
