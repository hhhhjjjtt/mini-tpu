"""Runs one program in the simulator. Called by mtpu_stub.c from the build directory.

Also re-encodes every descriptor it meets, so the C encoders test the Python ones, and
appends one line per run to summary.txt.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from mini_tpu.isa import reencode
from mini_tpu.sim import Machine

prog_off, sync_bytes, buf_phys, *regions = (int(x) for x in Path("run.cfg").read_text().split())
stats = {"descs": 0, "encode_mismatch": 0}


def check_encoder(d, f):
    stats["descs"] += 1
    if bytes(reencode(f)) != bytes(d):
        stats["encode_mismatch"] += 1
        print(f"encoder mismatch, opcode 0x{d[0]:02x}:\n C  {bytes(d).hex()}\n py {bytes(reencode(f)).hex()}")


m = Machine(ddr_bytes=sync_bytes, ddr_base=buf_phys, on_desc=check_encoder)
m.ddr[:] = np.fromfile("run.in", dtype=np.uint8)
m.region = regions
status = m.run(buf_phys + prog_off)
m.ddr.tofile("run.out")
Path("run.regs").write_text(f"{status} {m.desc_count} {m.err_desc} {m.err_unit} {m.err_info0} {m.err_info1}")

c = m.checker
for finding in c.findings:
    if finding.kind != "UNINIT":
        print("finding:", finding)
with open("summary.txt", "a") as f:
    f.write(f"status={status:#x} descs={stats['descs']} encode_mismatch={stats['encode_mismatch']} "
            f"races={c.races} rules={c.rule_violations} uninit={c.count.get('UNINIT', 0)} {m.message}\n")
