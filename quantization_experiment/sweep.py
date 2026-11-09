"""Turns on one hardware storage format at a time and measures the damage.

usage: sweep.py [S ...]     results go to results/sweep_S<S>.json
"""
import json
import sys
import time
import torch
from gpt2_quant import *
from gpt2_quant import ROOT

W8 = Cfg("w8", w=8, fold_ln=True)
A8 = replace(W8, act=8, clip="mse")
ALL = replace(A8, qkv=8, p_levels=127, gelu_in=8)      # everything except the residual

LADDER = [
    Cfg("fp32"),
    W8,
    # GEMM input activations: range choice and LN folding
    replace(W8, name="+act8 max, LN not folded", fold_ln=False, act=8),
    replace(W8, name="+act8 max", act=8),
    replace(W8, name="+act8 pct", act=8, clip="pct"),
    replace(A8, name="+act8 mse"),
    replace(A8, name="+act8 mse, LN not folded", fold_ln=False),
    # each remaining format alone, on top of w8 + act8
    replace(A8, name="+qkv8", qkv=8),
    replace(A8, name="+scores8", scores=8),
    replace(A8, name="+P i8 zp0 (127)", p_levels=127),
    replace(A8, name="+P i8 zp-128 (255)", p_levels=255),
    replace(A8, name="+GELU LUT i8", gelu_in=8),
    # residual stream width, everything else on
    replace(ALL, name="all, resid i32"),
    replace(ALL, name="all, resid 16", resid=16),
    replace(ALL, name="all, resid 12", resid=12),
    replace(ALL, name="all, resid 8 max", resid=8, branch=8, resid_clip="max"),
    replace(ALL, name="all, resid 8 mse", resid=8, branch=8),
]


def stats(obs):
    rows = []
    for site in ("embed", "0.res2", "2.res2", "5.res2", "10.res2", "11.res2", "0.ln1", "5.ln1", "lnf",
                 "5.q", "5.k", "5.v", "5.ctx", "5.fc", "5.gelu", "5.mlp_out"):
        v = torch.cat(obs.sample[site]).float()
        rows.append({"site": site, "max": obs.amax[site], "max_no_pos0": obs.amax_rest.get(site),
                     "p99.99": torch.quantile(v[:2_000_000], 0.9999).item(),
                     "rms": v.pow(2).mean().sqrt().item()})
    return rows


def main():
    sd = load_gpt2()
    train, test = data()
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    for S in [int(a) for a in sys.argv[1:]] or [256, 1024]:
        obs = {f: calibrate(sd, train, S, fold=f) for f in (False, True)}
        st = stats(obs[True])
        print(f"\n=== ranges, S={S} (LN outputs folded)")
        print(f"{'site':10}{'max':>10}{'max,no pos0':>13}{'p99.99':>10}{'rms':>9}")
        for r in st:
            m = r["max_no_pos0"]
            print(f"{r['site']:10}{r['max']:10.1f}{m if m is None else round(m, 1):>13}{r['p99.99']:10.2f}{r['rms']:9.3f}")
        print(f"\n=== S={S}\n{'config':32}{'ppl':>10}{'d ppl':>9}{'top1':>8}{'KL':>9}")
        res, base = [], None
        for cfg in LADDER:
            t = time.time()
            r = evaluate(sd, cfg, obs[cfg.fold_ln], test, S)
            base = base or r["ppl"]
            print(f"{cfg.name:32}{r['ppl']:10.2f}{r['ppl'] - base:+9.2f}{r['top1_agree']:8.3f}{r['kl']:9.4f}  ({time.time() - t:.0f}s)", flush=True)
            res.append({"name": cfg.name, **r})
        (out_dir / f"sweep_S{S}.json").write_text(json.dumps({"S": S, "ranges": st, "results": res}, indent=1))


if __name__ == "__main__":
    main()
