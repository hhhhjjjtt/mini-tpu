"""Round 2: fixes for the problems round 1 found, and where the remaining loss comes from.

usage: sweep2.py [S ...]     results go to results/sweep2_S<S>.json
"""
import json
import sys
from gpt2_quant import *
from gpt2_quant import ROOT

A8 = Cfg("base", w=8, fold_ln=True, act=8, clip="mse")
ALL = replace(A8, qkv=8, p_levels=127, p_max=0.5, gelu_in=8)

PLAIN = [
    Cfg("fp32"),
    replace(ALL, name="all, P clip 0.5"),
    # softmax output with zp = 0, no correction GEMM: trade range for resolution
    *[replace(A8, name=f"P 127 levels, clip {m}", p_levels=127, p_max=m) for m in (1.0, 0.5, 0.25, 0.125)],
    replace(A8, name="P 255 levels", p_levels=255),
    # residual width with the full range kept (position 0 is about 3000)
    *[replace(ALL, name=f"all, resid {b} max", resid=b, resid_clip="max") for b in (24, 20, 16, 14)],
    # which format costs the most: leave one out of the all-on config
    *[replace(ALL, name=f"all except act {k}", act_skip=(k,)) for k in ("ln1", "ctx", "ln2", "gelu", "lnf")],
    *[replace(ALL, name=f"all except weights {k}", w_skip=(k,)) for k in ("attn", "proj", "fc", "mlp", "head")],
]


def main():
    sd = load_gpt2()
    train, test = data()
    for S in [int(a) for a in sys.argv[1:]] or [1024]:
        obs = calibrate(sd, train, S, fold=True)
        print(f"\n=== S={S}\n{'config':38}{'ppl':>10}{'d ppl':>9}{'top1':>8}{'KL':>9}", flush=True)
        res, base = [], None

        def run(cfg, o):
            nonlocal base
            r = evaluate(sd, cfg, o, test, S)
            base = base or r["ppl"]
            print(f"{cfg.name:38}{r['ppl']:10.2f}{r['ppl'] - base:+9.2f}{r['top1_agree']:8.3f}{r['kl']:9.4f}", flush=True)
            res.append({"name": cfg.name, **r})

        for cfg in PLAIN:
            run(cfg, obs)
        for a in (0.5, 0.75):
            sm = calibrate(sd, train, S, fold=True, smooth=a)
            run(Cfg(f"smooth {a}, no quant (must equal fp32)", fold_ln=True, smooth=a), sm)
            run(replace(ALL, name=f"all + SmoothQuant {a}", smooth=a), sm)
        (ROOT / "results" / f"sweep2_S{S}.json").write_text(json.dumps({"S": S, "results": res}, indent=1))


if __name__ == "__main__":
    main()
