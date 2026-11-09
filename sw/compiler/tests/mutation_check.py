"""Shows rtl_diff can fail: injects one bug at a time into the simulator and compares with saved RTL output.

Run after rtl_diff.py, with the same seed and scale. Every mutation must change DDR.
usage: mutation_check.py [--seed N] [--scale F]
"""
import argparse
import sys
from contextlib import contextmanager
from unittest import mock

import numpy as np

import rtl_diff
from mini_tpu import golden, sim
from mini_tpu.golden import apply_scale, exp2neg, wrap32

MUTATIONS = {}


def mutation(name):
    def add(fn):
        MUTATIONS[name] = fn
        return fn
    return add


@mutation("double rounding ignored")
@contextmanager
def _():
    with mock.patch.object(golden, "apply_scale", lambda v, m, s, dr=False: apply_scale(v, m, s, False)):
        yield


@mutation("shift rounds down instead of to nearest")
@contextmanager
def _():
    def floor_scale(v, m, s, dr=False):
        return wrap32(np.right_shift(np.asarray(v, dtype=np.int64) * np.asarray(m, dtype=np.int64), np.asarray(s, dtype=np.int64)))
    with mock.patch.object(golden, "apply_scale", floor_scale):
        yield


@mutation("exp2neg without the +128 rounding")
@contextmanager
def _():
    with mock.patch.object(golden, "exp2neg", lambda z: exp2neg(np.asarray(z) - 128)):
        yield


@mutation("eltwise MAX behaves as MIN")
@contextmanager
def _():
    with mock.patch.object(golden.np, "maximum", np.minimum):
        yield


@mutation("clamp not applied")
@contextmanager
def _():
    with mock.patch.object(golden, "clamp8", lambda v, lo, hi: np.asarray(v, dtype=np.int64).astype(np.int8)):
        yield


@mutation("LOAD_WGT_VM ignores TRANSPOSE")
@contextmanager
def _():
    real = sim.Machine._load_wgt_vm

    def no_t(self, cmd):
        cmd.f["flags"] = 0
        cmd.f["src_pitch"] = 16
        real(self, cmd)
    with mock.patch.object(sim.Machine, "_load_wgt_vm", no_t):
        yield


@mutation("ACC partial pitch 0 reads successive rows")
@contextmanager
def _():
    real = sim.Machine._gemm_epilogue

    def bad(self, g, e):
        real(self, g, dict(e, partial_pitch=e["partial_pitch"] or 64))
    with mock.patch.object(sim.Machine, "_gemm_epilogue", bad):
        yield


@mutation("softmax mask ignored")
@contextmanager
def _():
    real = golden.softmax
    with mock.patch.object(golden, "softmax", lambda x, mask, *a: real(x, None, *a)):
        yield


@mutation("loop strides ignored")
@contextmanager
def _():
    real = sim.Machine._loop

    def bad(self, f, i, length):
        return real(self, dict(f, strides=(0,) * 8), i, length)
    with mock.patch.object(sim.Machine, "_loop", bad):
        yield


@mutation("CAUSAL ignored")
@contextmanager
def _():
    real = golden.softmax
    with mock.patch.object(golden, "softmax", lambda x, mask, *a: real(x, mask, *a[:6])):
        yield


@mutation("CAUSAL limit off by one")
@contextmanager
def _():
    real = golden.softmax

    def bad(x, mask, *a):
        return real(x, mask, *a[:6], None if a[6] is None else a[6] + 1)
    with mock.patch.object(golden, "softmax", bad):
        yield


@mutation("LayerNorm mean without fractional bits")
@contextmanager
def _():
    with mock.patch.object(golden, "LN_MEAN_FRAC", 0):
        yield


@mutation("LayerNorm input not saturated")
@contextmanager
def _():
    with mock.patch.object(golden, "LN_MAX", 1 << 30):
        yield


@mutation("LayerNorm square root one too large")
@contextmanager
def _():
    real = golden.isqrt32
    with mock.patch.object(golden, "isqrt32", lambda d: real(d) + 1):
        yield


@mutation("LUT index off by one")
@contextmanager
def _():
    with mock.patch.object(golden, "lut", lambda x, t: t[(x.view(np.uint8).astype(np.int64) + 1) & 255]):
        yield


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    img = rtl_diff.build(a.seed, a.scale)
    rtl = rtl_diff.read_ddr_out()
    clean = rtl_diff.simulate(img)[0]
    print(f"no mutation: {np.count_nonzero(clean != rtl)} bytes differ")
    missed = 0
    for name, ctx in MUTATIONS.items():
        with ctx():
            n = np.count_nonzero(rtl_diff.simulate(img)[0] != rtl)
        print(f"{n:8} bytes differ: {name}")
        missed += n == 0
    sys.exit(1 if missed else 0)


if __name__ == "__main__":
    main()
