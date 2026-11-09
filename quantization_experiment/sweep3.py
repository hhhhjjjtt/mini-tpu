"""Round 3: the final format set, its sensitivity, and generated text.

usage: sweep3.py [S ...]     results go to results/sweep3_S<S>.json
"""
import json
import sys
import torch
from gpt2_quant import *
from gpt2_quant import ROOT

FINAL = Cfg("final, P 255", w=8, fold_ln=True, act=8, clip="mse", qkv=8, p_levels=255,
            gelu_in=8, resid=16, resid_clip="max")

CONFIGS = [
    Cfg("fp32"),
    FINAL,
    replace(FINAL, name="final, P 127", p_levels=127),
    replace(FINAL, name="final except qkv", qkv=None),
    replace(FINAL, name="final except GELU input", gelu_in=None),
    *[replace(FINAL, name=f"final except act {k}", act_skip=(k,)) for k in ("ln1", "ctx", "ln2", "gelu", "lnf")],
    *[replace(FINAL, name=f"final except weights {k}", w_skip=(k,)) for k in ("attn", "proj", "fc", "mlp", "head")],
]
PROMPTS = ["The history of the Roman Empire began when",
           "Scientists have recently discovered that",
           "The best way to learn programming is"]


@torch.no_grad()
def generate(sd, cfg, obs, tok, prompt, n=40):
    p, q = prepare(sd, cfg, obs.ch), Quant(obs, cfg)
    ids = torch.tensor([tok(prompt)["input_ids"]], device=DEV)
    for _ in range(n):
        ids = torch.cat([ids, forward(ids, p, cfg, q)[:, -1:].argmax(-1)], dim=1)
    return tok.decode(ids[0])


def main():
    from transformers import GPT2TokenizerFast
    sd = load_gpt2()
    train, test = data()
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    for S in [int(a) for a in sys.argv[1:]] or [256, 1024]:
        obs = calibrate(sd, train, S, fold=True)
        print(f"\n=== S={S}\n{'config':34}{'ppl':>10}{'d ppl':>9}{'top1':>8}{'KL':>9}", flush=True)
        res, base = [], None

        def run(cfg, o):
            nonlocal base
            r = evaluate(sd, cfg, o, test, S)
            base = base or r["ppl"]
            print(f"{cfg.name:34}{r['ppl']:10.2f}{r['ppl'] - base:+9.2f}{r['top1_agree']:8.3f}{r['kl']:9.4f}", flush=True)
            res.append({"name": cfg.name, **r})

        for cfg in CONFIGS:
            run(cfg, obs)
        for a in (0.5, 0.75):
            run(replace(FINAL, name=f"final + SmoothQuant {a}", smooth=a), calibrate(sd, train, S, fold=True, smooth=a))
        samples = []
        for prompt in PROMPTS:
            fp, qt = generate(sd, Cfg("fp"), obs, tok, prompt), generate(sd, FINAL, obs, tok, prompt)
            print(f"\n[fp32 ] {fp!r}\n[final] {qt!r}", flush=True)
            samples.append({"fp32": fp, "final": qt})
        (ROOT / "results" / f"sweep3_S{S}.json").write_text(json.dumps({"S": S, "results": res, "samples": samples}, indent=1))


if __name__ == "__main__":
    main()
