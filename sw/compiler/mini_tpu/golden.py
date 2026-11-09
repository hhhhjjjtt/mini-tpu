"""Bit-exact vector and matrix arithmetic, on NumPy arrays (INTERFACE_SPEC.md §5.5-§5.7, §9).

Every function matches hw/rtl/vector_unit/vector_unit.sv, including its 32-bit wrap.
"""
import math

import numpy as np

I32, I64 = np.int32, np.int64

# LUT[k] = floor(65536 * 2^(-k/256) + 0.5), the ROM in exp2neg.sv
EXP2_LUT = np.array([math.floor(65536.0 * 2.0 ** (-i / 256.0) + 0.5) for i in range(256)], dtype=I64)


def wrap32(x):
    return np.asarray(x, dtype=I64).astype(I32)


def apply_scale(v, mult, shift, double_round=False):
    """(v * mult + round) >> shift, keeping the low 32 bits. v, mult, shift broadcast."""
    v = np.asarray(v, dtype=I64)
    mult = np.asarray(mult, dtype=I64)
    shift = np.asarray(shift, dtype=I64)
    rnd = np.where(shift == 0, I64(0), np.left_shift(I64(1), np.maximum(shift - 1, 0)))
    if np.any(double_round):
        step = np.where(v < 0, I64(-(1 << 30)), I64(1 << 30))
        rnd = rnd + np.where((shift > 31) & np.asarray(double_round), step, I64(0))
    return wrap32(np.right_shift(v * mult + rnd, shift))


def clamp8(v, lo, hi):
    """Low byte of clamp(v, lo, hi), tested in the RTL's order (lo first)."""
    v = np.asarray(v, dtype=I64)
    return np.where(v < lo, lo, np.where(v > hi, hi, v)).astype(np.int8)


def gemm(a, w):
    """i8 x i8 -> i32. Products summed over K <= 512 stay below 2^24, so float32 is exact."""
    return (a.astype(np.float32) @ w.astype(np.float32)).astype(I32)


def requant(x, bias, mult, shift, out_zp, clamp_min, clamp_max, double_round):
    """§9.1. x: [rows, 16] i32; bias, mult, shift: [16] or scalar."""
    v = wrap32(x.astype(I64) + np.asarray(bias, dtype=I64))
    r = wrap32(apply_scale(v, mult, shift, double_round).astype(I64) + out_zp)
    return clamp8(r, clamp_min, clamp_max)


def eltwise(op, a, b, p, out_i32, ops):
    """§5.6, §9.2. a: [rows, cols] i32 before zero point; b broadcastable, or None for PASS_A.

    p: dict of the params block. ops: namespace with the EW_* opcodes.
    """
    dr = p["double_round"]
    va = apply_scale(wrap32(a.astype(I64) - p["a_zp"]), p["a_mult"], p["a_shift"], dr).astype(I64)
    if op == ops.EW_PASS_A:
        r = va
    else:
        vb = apply_scale(wrap32(b.astype(I64) - p["b_zp"]), p["b_mult"], p["b_shift"], dr).astype(I64)
        if op == ops.EW_ADD:
            r = va + vb
        elif op == ops.EW_SUB:
            r = va - vb
        elif op == ops.EW_MAX:
            r = np.maximum(va, vb)
        elif op == ops.EW_MIN:
            r = np.minimum(va, vb)
        else:                                               # MUL: same formula, dr = 0
            r = apply_scale(va, vb, p["mul_shift"])
    r = wrap32(r)
    if out_i32:
        return r
    s = wrap32(apply_scale(r, p["out_mult"], p["out_shift"], dr).astype(I64) + p["out_zp"])
    return clamp8(s, p["clamp_min"], p["clamp_max"])


def exp2neg(z):
    """2^-z for z >= 0: Q16 in, Q0.16 out (exp2neg.sv)."""
    zr = np.asarray(z, dtype=I64) + 128
    ip = zr >> 16
    return np.where(ip > 16, I64(0), EXP2_LUT[(zr >> 8) & 255] >> np.minimum(ip, 16))


def softmax(x, mask, in_mult, in_shift, in_zp, out_mult, out_shift, out_zp, q_pos=None):
    """§5.7. x: [rows, len] i32; mask: [len] i32 or None. Returns i8 [rows, len].

    q_pos: CAUSAL, row r sees columns 0 .. q_pos + r; None = off.
    """
    t = apply_scale(wrap32(x.astype(I64) - in_zp), in_mult, in_shift).astype(I64)
    if mask is not None:
        t = np.clip(t + mask.astype(I64), -(1 << 31), (1 << 31) - 1)
    rows, length = t.shape
    if q_pos is not None:
        t = np.where(np.arange(length) > q_pos + np.arange(rows)[:, None], I64(-(1 << 31)), t)
    mx = np.full(rows, -(1 << 31), dtype=I64)
    total = np.zeros(rows, dtype=I64)
    for b in range(0, length, 16):
        blk = t[:, b:b + 16]
        mnew = np.maximum(blk.max(axis=1), mx)
        total = ((total * exp2neg(mnew - mx)) >> 16) + exp2neg(mnew[:, None] - blk).sum(axis=1)
        mx = mnew
    recip = (I64(1) << 46) // total
    p = apply_scale(exp2neg(mx[:, None] - t), recip[:, None], 30)
    return clamp8(wrap32(apply_scale(p, out_mult, out_shift).astype(I64) + out_zp), -128, 127)


def lut(x, table):
    """§5.7.2. x: i8; table: 256 x i8, indexed by the input byte."""
    return table[x.view(np.uint8)]


LN_FRAC = 24                                                # u = x_hat / n in Q24
LN_MEAN_FRAC = 8                                            # fractional bits of the mean
LN_MAX = (1 << 20) - 1                                      # input saturation


def bit_length(v):
    """Position of the top set bit + 1, for v >= 1 (a priority encoder)."""
    v = np.asarray(v, dtype=I64).copy()
    n = np.zeros_like(v)
    for s in (32, 16, 8, 4, 2, 1):
        hi = v >> s
        n = np.where(hi > 0, n + s, n)
        v = np.where(hi > 0, hi, v)
    return n + 1


def isqrt32(d):
    """floor(sqrt(d)) for d < 2^32, one result bit per step."""
    d = np.asarray(d, dtype=I64)
    r = np.zeros_like(d)
    for bit in range(15, -1, -1):
        t = r | 1 << bit
        r = np.where(t * t <= d, t, r)
    return r


def layernorm(x, inv_mult, inv_shift, eps, c_mult, c_shift, out_zp):
    """Integer (x - mean) / sqrt(var + eps), scaled to i8. x: [rows, n] i32 -> i8. See docs/VEC_LAYERNORM.md.

    inv_mult, inv_shift: 1 / n. eps: n^2 * real_eps / in_scale^2, >= 1.
    c_mult, c_shift: n / (out_scale * 2^LN_FRAC).
    """
    x = np.clip(np.asarray(x, dtype=I64), -LN_MAX, LN_MAX)
    n = x.shape[1]
    s1, s2 = x.sum(axis=1), (x * x).sum(axis=1)             # pass 1, exact
    d = n * s2 - s1 * s1 + eps                              # n^2 * (var + eps)
    mean = apply_scale(s1, inv_mult, inv_shift - LN_MEAN_FRAC).astype(I64)
    sh = bit_length(d) - 32                                 # normalize to [2^30, 2^32), even shift
    sh = sh + (sh & 1)
    dn = np.where(sh >= 0, d >> np.maximum(sh, 0), d << np.maximum(-sh, 0))
    recip = (I64(1) << 46) // (2 * isqrt32(dn))             # the softmax divider: 2^45 / sqrt(dn)
    a = 45 + (sh >> 1) - LN_FRAC + LN_MEAN_FRAC
    u = apply_scale((x << LN_MEAN_FRAC) - mean[:, None], recip[:, None], a[:, None])    # pass 2
    return clamp8(wrap32(apply_scale(u, c_mult, c_shift).astype(I64) + out_zp), -128, 127)


def layernorm_params(n, in_scale, out_scale, real_eps=1e-5):
    """Compile-time fields of layernorm(), after inv_mult.. in its argument order."""
    inv_mult, inv_shift = real_to_ms(1.0 / n)
    eps = min(max(round(n * n * real_eps / in_scale ** 2), 1), (1 << 32) - 1)
    c_mult, c_shift = real_to_ms(n / (out_scale * 2.0 ** LN_FRAC))
    return inv_mult, inv_shift, eps, c_mult, c_shift


def real_to_ms(r):
    """Real scale -> (mult, shift), mult in [2^30, 2^31)."""
    e = 30 - math.floor(math.log2(r))
    v = math.floor(r * 2.0 ** e + 0.5)
    if v >= 1 << 31:
        v, e = v // 2, e - 1
    return v, e
