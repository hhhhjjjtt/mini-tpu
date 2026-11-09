"""GPT-2 forward pass with fake quantization at every point the mini-tpu stores a value.

Fake quantization (round to the integer grid, keep floats) models the storage formats.
The hardware's integer arithmetic itself (requant rounding, softmax ROM) is not modelled.
"""
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).parent
os.environ.setdefault("HF_HOME", str(ROOT.parent / ".cache"))

import torch

D, H, DH, L = 768, 12, 64, 12
EPS = 1e-5
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class Cfg:
    name: str
    w: int | None = None        # weight bits, per output channel
    fold_ln: bool = False       # fold LayerNorm gamma/beta into the next GEMM
    act: int | None = None      # GEMM inputs: LN out, attention context, GELU out
    qkv: int | None = None      # Q, K, V (operands of the attention GEMMs)
    scores: int | None = None   # Q.K^T before softmax; None = i32 (exact)
    p_levels: int | None = None  # softmax output levels: 127 (zp 0) or 255 (zp -128)
    p_max: float = 1.0          # softmax output clip
    resid: int | None = None    # residual stream bits; None = wide (i32)
    branch: int | None = None   # attention / MLP branch output before the residual add
    gelu_in: int | None = None  # LUT input bits; None = exact GELU input
    clip: str = "max"           # activation range: max, pct, mse
    resid_clip: str | None = None  # residual range; None = same as clip
    act_skip: tuple = ()        # GEMM inputs kept wide: ln1, ctx, ln2, gelu, lnf
    w_skip: tuple = ()          # weights kept wide: attn, proj, fc, mlp, head
    smooth: float | None = None  # SmoothQuant alpha on LN outputs (needs a per-column multiply)


def load_gpt2():
    from transformers import GPT2LMHeadModel
    sd = GPT2LMHeadModel.from_pretrained("gpt2").state_dict()
    return {k: v.detach().float().to(DEV) for k, v in sd.items()}


def quant_weight(w, bits):
    """w: [in, out]; symmetric, one scale per output channel."""
    if bits is None:
        return w
    qmax = 2 ** (bits - 1) - 1
    s = w.abs().amax(dim=0, keepdim=True).clamp_min(1e-12) / qmax
    return (w / s).round().clamp(-qmax, qmax) * s


def prepare(sd, cfg, ch=None):
    """Weights as the compiler would emit them: LN folded, then quantized.

    ch: per-column LN output ranges, needed when cfg.smooth is set.
    """
    p = {"wte": sd["transformer.wte.weight"], "wpe": sd["transformer.wpe.weight"]}

    def gemm(name, w, b, ln=None, site=None):
        if cfg.fold_ln and ln is not None:
            g, beta = ln
            b = (b if b is not None else 0) + beta @ w
            w = g[:, None] * w
        if cfg.smooth is not None and site is not None:
            a = cfg.smooth
            s = (ch[site].clamp_min(1e-5) ** a / w.abs().amax(dim=1).clamp_min(1e-5) ** (1 - a)).clamp_min(1e-5)
            p["smooth", site] = s
            w = s[:, None] * w
        return (w if name in cfg.w_skip else quant_weight(w, cfg.w)), b

    for l in range(L):
        t = f"transformer.h.{l}."
        ln1 = (sd[t + "ln_1.weight"], sd[t + "ln_1.bias"])
        ln2 = (sd[t + "ln_2.weight"], sd[t + "ln_2.bias"])
        p[l] = {
            "ln1": ln1, "ln2": ln2,
            "attn": gemm("attn", sd[t + "attn.c_attn.weight"], sd[t + "attn.c_attn.bias"], ln1, f"{l}.ln1"),
            "proj": gemm("proj", sd[t + "attn.c_proj.weight"], sd[t + "attn.c_proj.bias"]),
            "fc": gemm("fc", sd[t + "mlp.c_fc.weight"], sd[t + "mlp.c_fc.bias"], ln2, f"{l}.ln2"),
            "mlp": gemm("mlp", sd[t + "mlp.c_proj.weight"], sd[t + "mlp.c_proj.bias"]),
        }
    lnf = (sd["transformer.ln_f.weight"], sd["transformer.ln_f.bias"])
    p["lnf"] = lnf
    p["head"] = gemm("head", p["wte"].t().contiguous(), None, lnf, "lnf")
    return p


class Observer:
    """Collects per-site ranges and a value sample during calibration."""
    def __init__(self, per_batch=20000):
        self.per_batch, self.amax, self.amax_rest, self.sample, self.ch = per_batch, {}, {}, {}, {}

    def __call__(self, site, x):
        self.amax[site] = max(self.amax.get(site, 0.0), x.abs().max().item())
        if x.dim() == 3:  # [B, S, C]: range without the first position
            self.amax_rest[site] = max(self.amax_rest.get(site, 0.0), x[:, 1:].abs().max().item())
        if site.endswith(("ln1", "ln2", "lnf")):
            c = x.abs().amax(dim=(0, 1))
            self.ch[site] = torch.maximum(self.ch[site], c) if site in self.ch else c
        flat = x.reshape(-1)
        idx = torch.randint(flat.numel(), (self.per_batch,), device=x.device)
        self.sample.setdefault(site, []).append(flat[idx].abs().cpu())
        return x

    def clip_value(self, site, bits, how):
        amax = self.amax[site]
        if how == "max":
            return amax
        v = torch.cat(self.sample[site]).to(DEV)
        if how == "pct":
            return min(amax, torch.quantile(v[:2_000_000], 0.9999).item())
        assert how == "mse"
        qmax = 2 ** (bits - 1) - 1
        best, best_e = amax, float("inf")
        for r in torch.linspace(0.02, 1.0, 50).tolist():
            s = amax * r / qmax
            e = ((v / s).round().clamp_max(qmax) * s - v).pow(2).mean().item()
            if e < best_e:
                best, best_e = amax * r, e
        return best


class Quant:
    """Applies fake quantization with the scales calibrated by an Observer."""
    def __init__(self, obs, cfg):
        self.obs, self.cfg, self.scale = obs, cfg, {}

    def __call__(self, site, x, bits, kind="act"):
        if bits is None:
            return x
        key = (site, bits)
        if key not in self.scale:
            how = (self.cfg.resid_clip or self.cfg.clip) if kind == "resid" else self.cfg.clip
            self.scale[key] = self.obs.clip_value(site, bits, how) / (2 ** (bits - 1) - 1)
        s, qmax = self.scale[key], 2 ** (bits - 1) - 1
        return (x / s).round().clamp(-qmax, qmax) * s


def layernorm(x, ln, fold):
    mu = x.mean(-1, keepdim=True)
    y = (x - mu) * torch.rsqrt((x - mu).pow(2).mean(-1, keepdim=True) + EPS)
    return y if fold else y * ln[0] + ln[1]


def gelu_new(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x.pow(3))))


def forward(ids, p, cfg, q, hidden_only=False):
    """q(site, x, bits[, kind]) is a Quant, or a wrapper around an Observer."""
    B, S = ids.shape

    def gemm_in(site, kind, y):
        return q(site, y, None if kind in cfg.act_skip else cfg.act)

    def ln_in(site, kind, y):
        s = p.get(("smooth", site))
        return gemm_in(site, kind, y if s is None else y / s)

    mask = torch.ones(S, S, dtype=torch.bool, device=ids.device).tril()
    x = q("embed", p["wte"][ids] + p["wpe"][:S], cfg.resid, "resid")
    for l in range(L):
        w = p[l]
        h = ln_in(f"{l}.ln1", "ln1", layernorm(x, w["ln1"], cfg.fold_ln))
        qkv = h @ w["attn"][0] + w["attn"][1]
        a, k, v = qkv.split(D, dim=-1)
        a, k, v = q(f"{l}.q", a, cfg.qkv), q(f"{l}.k", k, cfg.qkv), q(f"{l}.v", v, cfg.qkv)
        a, k, v = (t.view(B, S, H, DH).transpose(1, 2) for t in (a, k, v))
        s = q(f"{l}.scores", a @ k.transpose(-1, -2) / math.sqrt(DH), cfg.scores)
        pr = s.masked_fill(~mask, float("-inf")).softmax(-1)
        if cfg.p_levels:
            pr = (pr.clamp_max(cfg.p_max) / cfg.p_max * cfg.p_levels).round() / cfg.p_levels * cfg.p_max
        c = (pr @ v).transpose(1, 2).reshape(B, S, D)
        c = gemm_in(f"{l}.ctx", "ctx", c)
        o = q(f"{l}.attn_out", c @ w["proj"][0] + w["proj"][1], cfg.branch)
        x = q(f"{l}.res1", x + o, cfg.resid, "resid")
        h = ln_in(f"{l}.ln2", "ln2", layernorm(x, w["ln2"], cfg.fold_ln))
        f = q(f"{l}.fc", h @ w["fc"][0] + w["fc"][1], cfg.gelu_in)
        g = gemm_in(f"{l}.gelu", "gelu", gelu_new(f))
        o = q(f"{l}.mlp_out", g @ w["mlp"][0] + w["mlp"][1], cfg.branch)
        x = q(f"{l}.res2", x + o, cfg.resid, "resid")
    h = ln_in("lnf", "lnf", layernorm(x, p["lnf"], cfg.fold_ln))
    if hidden_only:
        return h
    b = p["head"][1]
    return h @ p["head"][0] + (b if b is not None else 0)


def calibrate(sd, tokens, S, fold, windows=64, batch=4, smooth=None):
    """Observe every site on the float model."""
    torch.manual_seed(0)                                    # the value samples set the MSE clip
    wins = tokens[: windows * S].view(windows, S).to(DEV)

    @torch.no_grad()
    def run(cfg, ch):
        p, obs = prepare(sd, cfg, ch), Observer()
        for i in range(0, windows, batch):
            forward(wins[i:i + batch], p, cfg, lambda site, x, bits, kind="act": obs(site, x), hidden_only=True)
        return obs

    cfg = Cfg("calib", fold_ln=fold)
    obs = run(cfg, None)
    if smooth is not None:
        ch = obs.ch
        obs = run(replace(cfg, smooth=smooth), ch)
        obs.ch = ch
    return obs


def data():
    """WikiText-2 raw: train tokens for calibration, test tokens for evaluation."""
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.model_max_length = 10 ** 9
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    out = []
    for split, limit in (("train", 2_000_000), ("test", None)):
        text = "\n\n".join(ds[split]["text"])
        out.append(torch.tensor(tok(text[:limit] if limit else text)["input_ids"]))
    return out


@torch.no_grad()
def evaluate(sd, cfg, obs, test, S, ref=None, tokens_per_batch=2048):
    """Perplexity on non-overlapping S-token windows, and agreement with the float model."""
    p = prepare(sd, cfg, obs.ch)
    q = Quant(obs, cfg)
    n = test.numel() // S
    wins = test[: n * S].view(n, S).to(DEV)
    bs = max(1, tokens_per_batch // S)
    nll = agree = kl = count = 0.0
    fp_cfg = Cfg("fp")
    p_ref = prepare(sd, fp_cfg) if ref is None else ref
    for i in range(0, n, bs):
        ids = wins[i:i + bs]
        lp = forward(ids, p, cfg, q)[:, :-1].log_softmax(-1)
        fp = forward(ids, p_ref, fp_cfg, lambda s, x, b, k="act": x)[:, :-1].log_softmax(-1)
        tgt = ids[:, 1:]
        nll += -lp.gather(-1, tgt[..., None]).sum().item()
        agree += (lp.argmax(-1) == fp.argmax(-1)).sum().item()
        kl += (fp.exp() * (fp - lp)).sum().item()
        count += tgt.numel()
    return {"ppl": math.exp(nll / count), "top1_agree": agree / count, "kl": kl / count, "tokens": int(count)}


__all__ = ["Cfg", "replace", "load_gpt2", "prepare", "forward", "calibrate", "data", "evaluate", "Quant", "DEV"]
