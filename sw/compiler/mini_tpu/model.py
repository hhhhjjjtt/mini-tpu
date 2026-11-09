"""GPT model: quantization to integers, and the integer forward pass a program must reproduce.

Weights arrive folded: LayerNorm gamma / beta already sit in the next linear layer, so
LayerNorm itself has no parameters. `scales` holds one real scale per tensor, by site name
("embed", "lnf", and per layer "<l>.ln1", q, k, v, ctx, res1, ln2, fc, gelu, res2).
"""
import math
from dataclasses import dataclass, field

import numpy as np

from . import golden

LOG2E = 1.4426950408889634
P_LEVELS = 127                  # softmax output, zero point 0: valid up to S = 256 (docs/QUANTIZATION.md)
RES_BITS = 16
K_CHUNK = 512


@dataclass(frozen=True)
class GPTConfig:
    n_layer: int
    d_model: int
    n_head: int
    d_ff: int
    vocab: int
    seq: int

    @property
    def d_head(self):
        return self.d_model // self.n_head

    @property
    def vocab_pad(self):
        return -(-self.vocab // 16) * 16


def gelu_new(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def ms(r):
    """Real scale -> (mult, shift), shift <= 62."""
    m, s = golden.real_to_ms(float(r))
    if s > 62:
        m, s = m >> (s - 62), 62
    return m, s


def ms_shared_shift(r):
    """Per-column multipliers with one shift, for VEC_ELTWISE MUL by a broadcast row."""
    s = min(62, 30 - math.floor(math.log2(float(r.max()))))
    return np.round(r.astype(np.float64) * 2.0 ** s).astype(np.int64).astype(np.int32), s


@dataclass
class Linear:
    w: np.ndarray               # [K, N] i8
    bias: np.ndarray            # [N] i32, accumulator scale
    s_w: np.ndarray             # [N] float64
    s_in: float
    mult: np.ndarray = None     # requant to i8, per column
    shift: np.ndarray = None

    def to_i8(self, s_out):
        r = self.s_in * self.s_w / s_out
        pairs = [ms(x) for x in np.broadcast_to(r, self.s_w.shape)]
        self.mult = np.array([p[0] for p in pairs], dtype=np.int32)
        self.shift = np.array([p[1] for p in pairs], dtype=np.uint8)
        return self


def quant_linear(w, b, s_in):
    w = np.asarray(w, dtype=np.float64)
    s_w = np.maximum(np.abs(w).max(axis=0), 1e-8) / 127
    wi = np.clip(np.round(w / s_w), -127, 127).astype(np.int8)
    b = np.zeros(w.shape[1]) if b is None else np.asarray(b, dtype=np.float64)
    bias = np.clip(np.round(b / (s_in * s_w)), -(1 << 31), (1 << 31) - 1).astype(np.int32)
    return Linear(wi, bias, s_w, s_in)


@dataclass
class Layer:
    attn: Linear
    proj: Linear
    fc: Linear
    mlp: Linear
    ln1: tuple                  # golden.layernorm_params
    ln2: tuple
    qk: tuple                   # softmax in_mult, in_shift
    pv: tuple                   # P.V requant mult, shift
    keep1: tuple                # residual rescale, old site to new
    keep2: tuple
    branch1: tuple              # (per-column multipliers, shift)
    branch2: tuple
    gelu: np.ndarray            # 256 x i8, indexed by the input byte


@dataclass
class QuantModel:
    cfg: GPTConfig
    s_embed: float
    layers: list
    lnf: tuple
    head: Linear                # padded to vocab_pad columns
    out_mult: int = 0
    out_shift: int = 0
    wte: np.ndarray = field(default=None, repr=False)
    wpe: np.ndarray = field(default=None, repr=False)

    def embed(self, ids, pos=0):
        """The host's part: token + position embedding on the residual grid, ids at positions pos, pos + 1, ...
        Returns i32 [len(ids), D]."""
        x = self.wte[ids].astype(np.float64) + self.wpe[pos:pos + len(ids)]
        return np.clip(np.round(x / self.s_embed), -32767, 32767).astype(np.int32)

    def logits(self, acc):
        """i32 accumulators -> float logits."""
        return acc[:, :self.cfg.vocab] * (self.head.s_in * self.head.s_w[:self.cfg.vocab])


def quantize(cfg, weights, scales):
    """weights: {"wte", "wpe", "head": (W, b), "layers": [{"attn", "proj", "fc", "mlp": (W, b)}]}."""
    dh, d = cfg.d_head, cfg.d_model
    layers = []
    s_res = scales["embed"]
    for l, w in enumerate(weights["layers"]):
        sc = lambda name: scales[f"{l}.{name}"]
        attn = quant_linear(*w["attn"], sc("ln1"))
        attn.to_i8(np.repeat([sc("q"), sc("k"), sc("v")], d))
        proj = quant_linear(*w["proj"], sc("ctx"))
        fc = quant_linear(*w["fc"], sc("ln2")).to_i8(sc("fc"))
        mlp = quant_linear(*w["mlp"], sc("gelu"))
        s1, s2 = sc("res1"), sc("res2")
        q = np.arange(256, dtype=np.int64).astype(np.uint8).view(np.int8).astype(np.float64)
        table = np.clip(np.round(gelu_new(q * sc("fc")) / sc("gelu")), -128, 127).astype(np.int8)
        layers.append(Layer(
            attn, proj, fc, mlp,
            golden.layernorm_params(d, s_res, sc("ln1")), golden.layernorm_params(d, s1, sc("ln2")),
            ms(sc("q") * sc("k") / math.sqrt(dh) * LOG2E * 65536.0), ms(sc("v") / P_LEVELS / sc("ctx")),
            ms(s_res / s1), ms(s1 / s2),
            ms_shared_shift(proj.s_in * proj.s_w / s1), ms_shared_shift(mlp.s_in * mlp.s_w / s2), table))
        s_res = s2
    hw, hb = weights["head"]
    pad = cfg.vocab_pad - cfg.vocab
    head = quant_linear(np.pad(hw, ((0, 0), (0, pad))), None if hb is None else np.pad(hb, (0, pad)), scales["lnf"])
    om, osh = golden.real_to_ms(P_LEVELS / 65536.0)
    return QuantModel(cfg, scales["embed"], layers, golden.layernorm_params(d, s_res, scales["lnf"]), head, om, osh,
                      weights["wte"], weights["wpe"])


# ---- reference: what a compiled program must produce, bit for bit ----
def _acc(a, lin):
    """a: i8 [rows, K]; K chunks as in the program."""
    out = np.zeros((a.shape[0], lin.w.shape[1]), dtype=np.int64)
    for k in range(0, a.shape[1], K_CHUNK):
        out += golden.gemm(a[:, k:k + K_CHUNK], lin.w[k:k + K_CHUNK])
    return golden.wrap32(out + lin.bias)


def _requant(a, lin):
    z = np.zeros_like(lin.bias)
    return golden.requant(_acc(a, lin), z, lin.mult, lin.shift, 0, -128, 127, False)


def _branch(x, acc, keep, branch):
    return golden.wrap32(golden.apply_scale(x, *keep).astype(np.int64) + golden.apply_scale(acc, *branch))


def forward(m, x, rows=None):
    """x: i32 [S, D] from embed(). Returns i32 accumulators [S, vocab_pad], or only for `rows`."""
    cfg = m.cfg
    d, dh, S = cfg.d_model, cfg.d_head, x.shape[0]
    for y in m.layers:
        h = golden.layernorm(x, *y.ln1, 0)
        qkv = _requant(h, y.attn)
        ctx = np.zeros((S, d), dtype=np.int8)
        for i in range(cfg.n_head):
            c = slice(i * dh, (i + 1) * dh)
            q, k, v = qkv[:, c], qkv[:, d:][:, c], qkv[:, 2 * d:][:, c]
            scores = golden.gemm(q, k.T)
            p = golden.softmax(scores, None, *y.qk, 0, m.out_mult, m.out_shift, 0, 0)
            ctx[:, c] = golden.requant(golden.gemm(p, v), 0, *y.pv, 0, -128, 127, False)
        x = _branch(x, _acc(ctx, y.proj), y.keep1, y.branch1)
        h = golden.layernorm(x, *y.ln2, 0)
        g = golden.lut(_requant(h, y.fc), y.gelu)
        x = _branch(x, _acc(g, y.mlp), y.keep2, y.branch2)
    return _acc(golden.layernorm(x if rows is None else x[rows], *m.lnf, 0), m.head)


# ---- float model, for calibration and for measuring quantization loss ----
def float_forward(cfg, weights, ids, observe=None, head=True):
    """Folded weights, ids [S]. observe(site, tensor) sees every quantized tensor."""
    see = observe or (lambda site, x: None)
    S, d, dh = len(ids), cfg.d_model, cfg.d_head

    def ln(x):
        mu = x.mean(-1, keepdims=True)
        return (x - mu) / np.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + 1e-5)

    x = weights["wte"][ids].astype(np.float64) + weights["wpe"][:S]
    see("embed", x)
    mask = np.triu(np.full((S, S), -np.inf), 1)
    for l, w in enumerate(weights["layers"]):
        h = ln(x)
        see(f"{l}.ln1", h)
        qkv = h @ w["attn"][0] + w["attn"][1]
        q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]
        see(f"{l}.q", q), see(f"{l}.k", k), see(f"{l}.v", v)
        ctx = np.zeros_like(q)
        for i in range(cfg.n_head):
            c = slice(i * dh, (i + 1) * dh)
            s = q[:, c] @ k[:, c].T / math.sqrt(dh) + mask
            p = np.exp(s - s.max(-1, keepdims=True))
            ctx[:, c] = (p / p.sum(-1, keepdims=True)) @ v[:, c]
        see(f"{l}.ctx", ctx)
        x = x + ctx @ w["proj"][0] + w["proj"][1]
        see(f"{l}.res1", x)
        h = ln(x)
        see(f"{l}.ln2", h)
        f = h @ w["fc"][0] + w["fc"][1]
        see(f"{l}.fc", f)
        g = gelu_new(f)
        see(f"{l}.gelu", g)
        x = x + g @ w["mlp"][0] + w["mlp"][1]
        see(f"{l}.res2", x)
    h = ln(x)
    see("lnf", h)
    if not head:
        return None
    hw, hb = weights["head"]
    return h @ hw + (0 if hb is None else hb)


def calibrate_max(cfg, weights, token_batches):
    """Scales from the largest value at each site, for models without a calibration set."""
    amax = {}

    def see(site, x):
        amax[site] = max(amax.get(site, 0.0), float(np.abs(x).max()))

    for ids in token_batches:
        float_forward(cfg, weights, ids, see)
    res = lambda s: s == "embed" or s.endswith(("res1", "res2"))
    return {s: v / ((1 << (RES_BITS - 1)) - 1 if res(s) else 127) for s, v in amax.items()}


def random_gpt(cfg, seed=0, ctx=None):
    """A GPT with random weights at GPT-2-like scale, already folded. ctx: position embeddings (default seq)."""
    rng = np.random.default_rng(seed)
    n = lambda *shape, s=0.08: rng.normal(0, s, shape)
    d, f = cfg.d_model, cfg.d_ff
    layers = [{"attn": (n(d, 3 * d), n(3 * d, s=0.02)), "proj": (n(d, d), n(d, s=0.02)),
               "fc": (n(d, f), n(f, s=0.02)), "mlp": (n(f, d, s=0.05), n(d, s=0.02))} for _ in range(cfg.n_layer)]
    wte = n(cfg.vocab, d, s=0.5)
    w = {"wte": wte, "wpe": n(cfg.seq, d, s=0.2), "head": (wte.T.copy(), n(cfg.vocab, s=0.02)), "layers": layers}
    if ctx and ctx > cfg.seq:
        w["wpe"] = np.vstack([w["wpe"], n(ctx - cfg.seq, d, s=0.2)])
    return w
