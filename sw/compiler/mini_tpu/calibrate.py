"""Stage 1: Hugging Face GPT-2 files + calibration text -> folded weights and scales.

Slow, and independent of the hardware, so make keeps its result until the model changes.
Needs NumPy and the `tokenizers` package. MSE clips come from histograms of every value.

usage: python -m mini_tpu.calibrate
"""
import json
import struct
import sys

import numpy as np

from . import config, model

BINS = 8192
CLIPS = np.linspace(0.02, 1.0, 50)


def read_safetensors(path):
    """Plain reader, so stage 1 needs no PyTorch."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if info["dtype"] != "F32":
            raise ValueError(f"{name}: only F32 is supported")
        a, b = info["data_offsets"]
        out[name.removeprefix("transformer.")] = data[a:b].view(np.float32).reshape(info["shape"])
    return out


def load_folded(path):
    """-> GPTConfig fields, and weights with LayerNorm gamma / beta folded into the next linear."""
    hp = json.loads((path / "config.json").read_text(encoding="utf-8"))
    sd = read_safetensors(path / "model.safetensors")
    d = hp["n_embd"]

    def fold(ln, w, b):
        g, beta = sd[ln + ".weight"].astype(np.float64), sd[ln + ".bias"].astype(np.float64)
        w = w.astype(np.float64)
        return g[:, None] * w, (0 if b is None else b.astype(np.float64)) + beta @ w

    layers = []
    for l in range(hp["n_layer"]):
        t = f"h.{l}."
        layers.append({
            "attn": fold(t + "ln_1", sd[t + "attn.c_attn.weight"], sd[t + "attn.c_attn.bias"]),
            "proj": (sd[t + "attn.c_proj.weight"].astype(np.float64), sd[t + "attn.c_proj.bias"].astype(np.float64)),
            "fc": fold(t + "ln_2", sd[t + "mlp.c_fc.weight"], sd[t + "mlp.c_fc.bias"]),
            "mlp": (sd[t + "mlp.c_proj.weight"].astype(np.float64), sd[t + "mlp.c_proj.bias"].astype(np.float64)),
        })
    wte = sd["wte.weight"]
    weights = {"wte": wte, "wpe": sd["wpe.weight"], "head": fold("ln_f", wte.T, None), "layers": layers}
    return {"n_layer": hp["n_layer"], "d_model": d, "n_head": hp["n_head"], "d_ff": hp.get("n_inner") or 4 * d,
            "vocab": hp["vocab_size"]}, weights


def mse_clip(hist, amax):
    """Best clip for 8 bits, from a histogram of |x| over [0, amax]."""
    v = (np.arange(BINS) + 0.5) * (amax / BINS)
    best, best_e = amax, np.inf
    for r in CLIPS:
        s = amax * r / 127
        e = (hist * (np.minimum(np.round(v / s), 127) * s - v) ** 2).sum()
        if e < best_e:
            best, best_e = amax * r, e
    return best


def calibrate(cfg, weights, windows):
    amax, hists = {}, {}

    def see_max(site, x):
        amax[site] = max(amax.get(site, 0.0), float(np.abs(x).max()))

    def see_hist(site, x):
        h, _ = np.histogram(np.abs(x), bins=BINS, range=(0.0, amax[site]))
        hists[site] = hists.get(site, 0) + h

    for pass_, see in (("ranges", see_max), ("histograms", see_hist)):
        for i, ids in enumerate(windows):
            model.float_forward(cfg, weights, ids, see, head=False)
            print(f"\r{pass_}: {i + 1} / {len(windows)}", end="", flush=True)
        print()
    res = lambda s: s == "embed" or s.endswith(("res1", "res2"))
    return {s: amax[s] / ((1 << (model.RES_BITS - 1)) - 1) if res(s) else mse_clip(hists[s], amax[s]) / 127 for s in amax}


def save(path, fields, weights, scales, tokens):
    a = {"wte": weights["wte"], "wpe": weights["wpe"], "head_w": weights["head"][0].astype(np.float32),
         "head_b": weights["head"][1].astype(np.float32), "tokens": tokens, "fields": json.dumps(fields),
         "scale_names": np.array(list(scales)), "scale_values": np.array(list(scales.values()))}
    for l, w in enumerate(weights["layers"]):
        for name, (wt, b) in w.items():
            a[f"{l}.{name}.w"], a[f"{l}.{name}.b"] = wt.astype(np.float32), b.astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **a)


def load(path):
    """-> (GPTConfig fields, weights, scales, tokens)."""
    z = np.load(path)
    fields = json.loads(str(z["fields"]))
    layers = [{n: (z[f"{l}.{n}.w"], z[f"{l}.{n}.b"]) for n in ("attn", "proj", "fc", "mlp")} for l in range(fields["n_layer"])]
    weights = {"wte": z["wte"], "wpe": z["wpe"], "head": (z["head_w"], z["head_b"]), "layers": layers}
    return fields, weights, dict(zip(z["scale_names"].tolist(), z["scale_values"].tolist())), z["tokens"]


def main():
    from tokenizers import Tokenizer
    c = config.load()
    fields, weights = load_folded(c.model_path)
    tok = Tokenizer.from_file(str(c.model_path / "tokenizer.json"))
    tokens = np.array(tok.encode(c.calib_path.read_text(encoding="utf-8")).ids)
    if len(tokens) < c.calib_tokens:
        sys.exit(f"{c.calib_path}: {len(tokens)} tokens, need {c.calib_tokens}")
    cfg = model.GPTConfig(seq=c.seq, **fields)
    windows = tokens[:c.calib_tokens // c.seq * c.seq].reshape(-1, c.seq)
    scales = calibrate(cfg, weights, windows)
    save(c.calibrated, fields, weights, scales, tokens[:c.calib_tokens])
    print(f"wrote {c.calibrated}: {len(scales)} scales from {windows.size} tokens")


if __name__ == "__main__":
    main()
