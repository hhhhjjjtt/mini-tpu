"""Simulator speed on one GPT-2 c_fc layer: A [256 x 768] . W [768 x 3072], K split 512 + 256."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mini_tpu import isa
from mini_tpu.isa import C
from mini_tpu.sim import Machine

M, K, N = 256, 768, 3072
LOAD, MATRIX, VECTOR, STORE = range(4)
R0, R1, R2, R3 = 0x100000, 0x80000, 0x90000, 0x400000
VA, VT, VP, VO = 0x000000, 0x40000, 0x50000, 0x80000


def program():
    d = [isa.load(1, 0, VT, 1, 256, 256, 256), isa.load(2, 0, VA, M, K, K, K)]
    off, half = 0, 0
    for n in range(N // 16):
        for c, (k0, k) in enumerate(((0, 512), (512, 256))):
            d.append(isa.wait(isa.load_wgt(0, off, half, k), MATRIX, 1))
            d.append(isa.wait(isa.gemm(VA + k0, K, M // 16, k, half), LOAD, 0))
            if c == 0:
                d.append(isa.wait(isa.epilogue(C.EP_STORE_I32, 0, 0, 0, 0, M, 64, VP, VT), LOAD, 0))
            else:
                d.append(isa.epilogue(C.EP_ACC_REQUANT_I8, 0, 0, -128, 127, M, N, VO + 16 * n, VT, VP, 64))
            off += 16 * k
            half ^= 1
    d.append(isa.wait(isa.store(3, 0, VO, M, N, N, N), VECTOR, 0))
    return d + [isa.halt()]


def main():
    rng = np.random.default_rng(0)
    descs = program()
    blob = isa.build_program(descs)
    for check in (False, True):
        m = Machine(ddr_bytes=8 << 20, check=check)
        m.region[:4] = [R0, R1, R2, R3]
        m.write_ddr(0x1000, blob)
        m.write_ddr(R0, rng.integers(0, 256, K * N, dtype=np.uint8).tobytes())
        m.write_ddr(R1, rng.integers(0, 256, 256, dtype=np.uint8).tobytes())
        m.write_ddr(R2, rng.integers(0, 256, M * K, dtype=np.uint8).tobytes())
        t = time.time()
        status = m.run(0x1000)
        s = time.time() - t
        print(f"check={check}: status {status:#x}, {m.desc_count} descriptors, {s:.2f} s, {s / m.desc_count * 1e3:.3f} ms each, "
              f"{M * K * N / s / 1e9:.1f} GMAC/s; about {65000 * s / m.desc_count:.0f} s for a 65,000-descriptor program")
        if check:
            print("races", m.checker.races, "rules", m.checker.rule_violations)


if __name__ == "__main__":
    main()
