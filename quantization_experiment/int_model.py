"""GPT-2 on the mini-tpu's integer arithmetic.

Integer: GEMMs, requant (apply_scale), residual adds, softmax (§5.7, plus the planned CAUSAL
flag), GELU as an i8 -> i8 table, LayerNorm (docs/VEC_LAYERNORM.md). The torch functions below are GPU ports of
sw/compiler/mini_tpu/golden.py; check_ports() proves them bit-exact against it.

usage: int_model.py [S:LEVELS ...]     default 256:127 1024:255; results in results/int_S<S>_P<LEVELS>.json
"""
import json
import math
import sys
import time

import numpy as np
import torch

from gpt2_quant import *
from gpt2_quant import D, DH, H, L, ROOT, gelu_new, layernorm

sys.path.insert(0, str(ROOT.parent / "sw" / "compiler"))
from mini_tpu import golden  # noqa: E402

LOG2E = 1.4426950408889634
RES_BITS = 16
LUT = torch.tensor(golden.EXP2_LUT, device=DEV)
INT_MIN = -(1 << 31)


# ---- ports of golden.py ----
def wrap32(x):
    return ((x + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)


def apply_scale(v, mult, shift):
    mult = torch.as_tensor(mult, dtype=torch.int64, device=v.device)
    shift = torch.as_tensor(shift, dtype=torch.int64, device=v.device)
    rnd = torch.where(shift == 0, torch.zeros_like(shift), torch.ones_like(shift) << (shift - 1).clamp_min(0))
    return wrap32((v * mult + rnd) >> shift)


def exp2neg(z):
    zr = z + 128
    ip = zr >> 16
    return torch.where(ip > 16, torch.zeros_like(zr), LUT[(zr >> 8) & 255] >> ip.clamp_max(16))


def softmax(x, in_mult, in_shift, out_mult, out_shift, out_zp, keep=None):
    """x: [..., n] i32 scores, n a multiple of 16. keep: bool, False = masked (CAUSAL flag)."""
    t = apply_scale(x, in_mult, in_shift)
    if keep is not None:
        t = torch.where(keep, t, torch.full_like(t, INT_MIN))
    mx = torch.full_like(t[..., 0], INT_MIN)
    total = torch.zeros_like(mx)
    for b in range(0, t.shape[-1], 16):
        blk = t[..., b:b + 16]
        mnew = torch.maximum(blk.amax(-1), mx)
        total = ((total * exp2neg(mnew - mx)) >> 16) + exp2neg(mnew[..., None] - blk).sum(-1)
        mx = mnew
    recip = (1 << 46) // total
    p = apply_scale(exp2neg(mx[..., None] - t), recip[..., None], 30)
    return (apply_scale(p, out_mult, out_shift) + out_zp).clamp(-128, 127)


def bit_length(v):
    n = torch.zeros_like(v)
    for s in (32, 16, 8, 4, 2, 1):
        hi = v >> s
        n = torch.where(hi > 0, n + s, n)
        v = torch.where(hi > 0, hi, v)
    return n + 1


def isqrt32(d):
    r = torch.zeros_like(d)
    for bit in range(15, -1, -1):
        t = r | 1 << bit
        r = torch.where(t * t <= d, t, r)
    return r


def layernorm_int(x, inv_mult, inv_shift, eps, c_mult, c_shift, out_zp=0):
    """x: [..., n] i32."""
    x = x.clamp(-golden.LN_MAX, golden.LN_MAX)
    n = x.shape[-1]
    s1, s2 = x.sum(-1), (x * x).sum(-1)
    d = n * s2 - s1 * s1 + eps
    mean = apply_scale(s1, inv_mult, inv_shift - golden.LN_MEAN_FRAC)
    sh = bit_length(d) - 32
    sh = sh + (sh & 1)
    dn = torch.where(sh >= 0, d >> sh.clamp_min(0), d << (-sh).clamp_min(0))
    recip = (1 << 46) // (2 * isqrt32(dn))
    a = 45 + (sh >> 1) - golden.LN_FRAC + golden.LN_MEAN_FRAC
    u = apply_scale((x << golden.LN_MEAN_FRAC) - mean[..., None], recip[..., None], a[..., None])
    return (apply_scale(u, c_mult, c_shift) + out_zp).clamp(-128, 127)


def check_ports():
    rng = np.random.default_rng(0)
    n = 0
    for _ in range(200):
        v = rng.integers(-(1 << 31), 1 << 31, 4096, dtype=np.int64)
        m = rng.integers(-(1 << 31), 1 << 31, 4096, dtype=np.int64)
        s = rng.integers(0, 63, 4096, dtype=np.int64)
        a = golden.apply_scale(v, m, s).astype(np.int64)
        b = apply_scale(torch.tensor(v, device=DEV), torch.tensor(m, device=DEV), torch.tensor(s, device=DEV)).cpu().numpy()
        assert (a == b).all()
        n += a.size
    for i in range(200):
        rows, length = 8, 16 * int(rng.integers(1, 33))
        x = rng.integers(-20000, 20000, (rows, length), dtype=np.int64)
        im, ish = golden.real_to_ms(float(rng.choice([0.0015, 0.01, 0.1])) * LOG2E * 65536.0)
        om, osh = golden.real_to_ms(float(rng.choice([127.0, 255.0])) / 65536.0)
        q = int(rng.integers(0, length + 1))
        a = golden.softmax(x.astype(np.int32), None, im, ish, 0, om, osh, -128, q).astype(np.int64)
        keep = torch.arange(length, device=DEV) <= q + torch.arange(rows, device=DEV)[:, None]
        b = softmax(torch.tensor(x, device=DEV), im, ish, om, osh, -128, keep).cpu().numpy()
        assert (a == b).all()
        n += a.size
    for i in range(300):
        bits = int(rng.integers(1, 21))
        x = rng.integers(-(1 << bits), 1 << bits, (8, 16 * int(rng.integers(1, 65))), dtype=np.int64)
        if i % 10 == 0:
            x[:] = x[:, :1]                                 # constant rows: variance 0
        prm = golden.layernorm_params(x.shape[1], float(rng.choice([0.0001, 0.01, 0.1])), float(rng.choice([0.05, 0.2])))
        a = golden.layernorm(x.astype(np.int32), *prm, 0).astype(np.int64)
        b = layernorm_int(torch.tensor(x, device=DEV), *prm).cpu().numpy()
        assert (a == b).all()
        n += a.size
    print(f"ports match golden.py on {n} values")


# ---- compile: scales, integer weights, multipliers ----
def ms(r):
    """Real scale -> (mult, shift), shift <= 62."""
    m, s = golden.real_to_ms(float(r))
    if s > 62:
        m, s = m >> (s - 62), 62
    return m, s


def ms_per_lane(r):
    m, s = zip(*(ms(x) for x in r.tolist()))
    return torch.tensor(m, device=DEV), torch.tensor(s, device=DEV)


def ms_shared_shift(r):
    """Per-column multipliers with one shift: VEC_ELTWISE MUL by a broadcast row."""
    s = min(62, 30 - math.floor(math.log2(float(r.max()))))
    return torch.round(r.double() * 2.0 ** s).long(), s


class Linear:
    def __init__(self, w, b, s_in):
        self.s_w = (w.abs().amax(0).clamp_min(1e-8) / 127).double()
        self.w = (w / self.s_w.float()).round().clamp(-127, 127)            # float32 holding integers
        self.bias = None if b is None else torch.round(b.double() / (s_in * self.s_w)).long()
        self.s_in = s_in

    def acc(self, a):
        """a: float32 holding i8. K chunks of 512 keep float32 exact (§5.4 K-split)."""
        out = 0
        for k in range(0, a.shape[-1], 512):
            out = out + (a[..., k:k + 512] @ self.w[k:k + 512]).long()
        return out if self.bias is None else wrap32(out + self.bias)

    def requant(self, a, s_out):
        """REQUANT_I8. s_out: scalar or per-column tensor."""
        m, s = ms_per_lane(self.s_in * self.s_w / s_out)
        self.req = (m, s)
        return self

    def to_i8(self, a):
        return apply_scale(self.acc(a), *self.req).clamp(-128, 127).float()


class IntModel:
    def __init__(self, sd, obs, levels, int_ln=True):
        self.int_ln = int_ln
        p = prepare(sd, Cfg("fold", fold_ln=True))
        sc = lambda site, bits=8, how="mse": obs.clip_value(site, bits, how) / (2 ** (bits - 1) - 1)
        res = lambda site: sc(site, RES_BITS, "max")
        self.p, self.levels = p, levels
        self.s_embed = res("embed")
        self.out_mult, self.out_shift = golden.real_to_ms(levels / 65536.0)
        self.out_zp = 0 if levels == 127 else -128
        self.s_p = 1.0 / levels
        self.layers = []
        s_res = self.s_embed
        for l in range(L):
            w = p[l]
            s_ln1, s_ln2 = sc(f"{l}.ln1"), sc(f"{l}.ln2")
            s_q, s_k, s_v = sc(f"{l}.q"), sc(f"{l}.k"), sc(f"{l}.v")
            s_ctx, s_fc, s_gelu = sc(f"{l}.ctx"), sc(f"{l}.fc"), sc(f"{l}.gelu")
            s_r1, s_r2 = res(f"{l}.res1"), res(f"{l}.res2")
            attn = Linear(*w["attn"], s_ln1)
            s_qkv = torch.tensor([s_q] * D + [s_k] * D + [s_v] * D, device=DEV, dtype=torch.float64)
            attn.requant(None, s_qkv)
            proj, fc, mlp = Linear(*w["proj"], s_ctx), Linear(*w["fc"], s_ln2), Linear(*w["mlp"], s_gelu)
            fc.requant(None, torch.tensor(s_fc, device=DEV, dtype=torch.float64))
            # GELU table over all i8 inputs
            q = torch.arange(-128, 128, device=DEV, dtype=torch.float32)
            table = (gelu_new(q * s_fc) / s_gelu).round().clamp(-128, 127)
            self.layers.append({
                "s_ln1": s_ln1, "s_ln2": s_ln2, "s_res_in": s_res, "s_r1": s_r1, "attn": attn, "proj": proj, "fc": fc, "mlp": mlp,
                "qk": ms(s_q * s_k / math.sqrt(DH) * LOG2E * 65536.0), "pv": ms(self.s_p * s_v / s_ctx), "table": table,
                "keep1": ms(s_res / s_r1), "branch1": ms_shared_shift(proj.s_in * proj.s_w / s_r1),
                "keep2": ms(s_r1 / s_r2), "branch2": ms_shared_shift(mlp.s_in * mlp.s_w / s_r2), "s_r2": s_r2,
            })
            s_res = s_r2
        self.s_res_out, self.s_lnf = s_res, sc("lnf")
        self.head = Linear(*p["head"], self.s_lnf)

    def ln(self, x, s_res, s_out):
        """VEC_LAYERNORM, or the float reference on the same integer residual."""
        if self.int_ln:
            return layernorm_int(x, *golden.layernorm_params(D, s_res, s_out)).float()
        return (layernorm(x.double() * s_res, None, True) / s_out).round().clamp(-128, 127).float()

    def block(self, x, y):
        B, S, _ = x.shape
        mask = torch.ones(S, S, dtype=torch.bool, device=DEV).tril()
        pad = -S % 16
        if pad:
            mask = torch.nn.functional.pad(mask, (0, pad))
        h = self.ln(x, y["s_res_in"], y["s_ln1"])
        qkv = y["attn"].to_i8(h)
        q, k, v = (t.view(B, S, H, DH).transpose(1, 2) for t in qkv.split(D, dim=-1))
        acc = (q @ k.transpose(-1, -2)).long()                              # STORE_I32 scores
        if pad:
            acc = torch.nn.functional.pad(acc, (0, pad))
        pr = softmax(acc, *y["qk"], self.out_mult, self.out_shift, self.out_zp, keep=mask)[..., :S]
        pr = pr - self.out_zp                                               # the correction GEMM removes the zero point
        c = 0
        for t in range(0, S, 512):
            c = c + (pr[..., t:t + 512].float() @ v[..., t:t + 512, :]).long()
        c = apply_scale(c, *y["pv"]).clamp(-128, 127).float().transpose(1, 2).reshape(B, S, D)
        x = apply_scale(x, *y["keep1"]) + apply_scale(y["proj"].acc(c), *y["branch1"])
        h = self.ln(x, y["s_r1"], y["s_ln2"])
        f = y["fc"].to_i8(h)
        g = y["table"][f.long() + 128]
        return apply_scale(x, *y["keep2"]) + apply_scale(y["mlp"].acc(g), *y["branch2"])

    def __call__(self, ids):
        S = ids.shape[1]
        x = self.p["wte"][ids] + self.p["wpe"][:S]
        x = (x.double() / self.s_embed).round().clamp(-32767, 32767).long()
        for y in self.layers:
            x = self.block(x, y)
        h = self.ln(x, self.s_res_out, self.s_lnf)
        return (h @ self.head.w) * (self.head.s_in * self.head.s_w).float() + self.p["head"][1]


@torch.no_grad()
def evaluate_int(sd, model, test, S):
    n = test.numel() // S
    wins = test[: n * S].view(n, S).to(DEV)
    bs = max(1, 1024 // S)
    p_ref, fp_cfg = prepare(sd, Cfg("fp")), Cfg("fp")
    nll = agree = kl = count = 0.0
    for i in range(0, n, bs):
        ids = wins[i:i + bs]
        lp = model(ids)[:, :-1].log_softmax(-1)
        fp = forward(ids, p_ref, fp_cfg, lambda s, x, b, k="act": x)[:, :-1].log_softmax(-1)
        tgt = ids[:, 1:]
        nll += -lp.gather(-1, tgt[..., None]).sum().item()
        agree += (lp.argmax(-1) == fp.argmax(-1)).sum().item()
        kl += (fp.exp() * (fp - lp)).sum().item()
        count += tgt.numel()
    return {"ppl": math.exp(nll / count), "top1_agree": agree / count, "kl": kl / count, "tokens": int(count)}


@torch.no_grad()
def generate(model, tok, prompt, n=40):
    ids = torch.tensor([tok(prompt)["input_ids"]], device=DEV)
    for _ in range(n):
        ids = torch.cat([ids, model(ids)[:, -1:].argmax(-1)], dim=1)
    return tok.decode(ids[0])


def main():
    from transformers import GPT2TokenizerFast
    check_ports()
    sd = load_gpt2()
    train, test = data()
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    prompts = ["The history of the Roman Empire began when", "Scientists have recently discovered that",
               "The best way to learn programming is"]
    for spec in sys.argv[1:] or ["256:127", "1024:255"]:
        S, levels = (int(v) for v in spec.split(":"))
        obs = calibrate(sd, train, S, fold=True)
        out = {"S": S, "levels": levels}
        for name, int_ln in (("float LN", False), ("integer LN", True)):
            model = IntModel(sd, obs, levels, int_ln)
            t = time.time()
            out[name] = evaluate_int(sd, model, test, S)
            r = out[name]
            print(f"S={S} P levels={levels} {name}: ppl {r['ppl']:.2f}, top1 {r['top1_agree']:.3f}, KL {r['kl']:.4f} ({time.time() - t:.0f} s)", flush=True)
        out["samples"] = [generate(model, tok, p) for p in prompts]
        for s in out["samples"]:
            print(repr(s), flush=True)
        (ROOT / "results" / f"int_S{S}_P{levels}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
