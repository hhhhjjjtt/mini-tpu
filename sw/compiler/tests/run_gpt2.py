"""GPT-2 small prefill: compile, run in the simulator, compare with the reference and the float model.

Needs `make` first; takes its model and sequence length from config.toml.
usage: run_gpt2.py [--windows N] [--no-check] [--prompt TEXT --tokens N]
--prompt needs transformers (in the repository's .venv).
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "sw" / "compiler"))

from mini_tpu import calibrate, codegen, config, model
from mini_tpu.isa import C
from mini_tpu.sim import Machine

MB = 1 << 20
PROG, R0, R1, R2, R3, KV = 0x1000, 32 * MB, 192 * MB, 208 * MB, 212 * MB, 240 * MB


def nll(logits, ids):
    lg = logits[:-1].astype(np.float64)
    lg -= lg.max(-1, keepdims=True)
    lp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
    return -lp[np.arange(len(ids) - 1), ids[1:]].sum()


def generate(q, c, m, prompt, n):
    """Greedy decoding, the ARM's loop: embed, run the program, take the argmax at the last real position.

    Rows past the prompt are padding; causal attention keeps them out of the real rows.
    """
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    ids = tok(prompt)["input_ids"]
    cfg = q.cfg
    for _ in range(n):
        padded = np.array(ids + [0] * (cfg.seq - len(ids)))
        status = codegen.run(m, c, q.embed(padded), (PROG, R0, R1, R2, R3, KV))
        assert status == C.ST_DONE, m.message
        out = np.frombuffer(m.read_ddr(R3, c.output_bytes), dtype="<i4").reshape(cfg.seq, cfg.vocab_pad)
        ids.append(int(q.logits(out[len(ids) - 1:len(ids)].astype(np.float64)).argmax()))
        print(repr(tok.decode(ids)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--no-check", action="store_true", help="skip the race checker")
    ap.add_argument("--prompt")
    ap.add_argument("--tokens", type=int, default=12)
    a = ap.parse_args()
    conf = config.load()
    S = conf.seq

    t = time.time()
    fields, weights, scales, test = calibrate.load(conf.calibrated)
    cfg = model.GPTConfig(seq=S, **fields)
    q = model.quantize(cfg, weights, scales)
    print(f"loaded and quantized in {time.time() - t:.0f} s", flush=True)
    t = time.time()
    c = codegen.compile_gpt(q, codegen.Target.from_hardware(conf.hardware))
    print(f"compiled in {time.time() - t:.0f} s: {c.descriptors} descriptors, program {len(c.program) / MB:.1f} MB, "
          f"weights {len(c.weights) / MB:.1f} MB, params {len(c.params) / MB:.1f} MB, vmem {c.vmem_used / 1024:.0f} KB", flush=True)

    m = Machine(ddr_bytes=KV + c.kv_bytes + MB, check=not a.no_check)
    if a.prompt:
        return generate(q, c, m, a.prompt, a.tokens)
    totals = {"q": 0.0, "f": 0.0, "agree": 0, "n": 0}
    bad = 0
    for w in range(a.windows):
        ids = test[w * S:(w + 1) * S].astype(np.int64)
        x = q.embed(ids)
        t = time.time()
        status = codegen.run(m, c, x, (PROG, R0, R1, R2, R3, KV))
        ts = time.time() - t
        if status != C.ST_DONE:
            sys.exit(f"status {status:#x}: {m.message}")
        out = np.frombuffer(m.read_ddr(R3, c.output_bytes), dtype="<i4").reshape(S, cfg.vocab_pad)
        t = time.time()
        ref = model.forward(q, x)
        tr = time.time() - t
        diff = int(np.count_nonzero(out != ref))
        bad += diff
        fl = model.float_forward(cfg, weights, ids)
        lq = q.logits(out.astype(np.float64))
        totals["q"] += nll(lq, ids)
        totals["f"] += nll(fl, ids)
        totals["agree"] += int((lq.argmax(-1) == fl.argmax(-1))[:-1].sum())
        totals["n"] += S - 1
        k = m.checker
        extra = f", races {k.races}, rules {k.rule_violations}, uninit {k.count.get('UNINIT', 0)}" if k else ""
        print(f"window {w}: sim {ts:.1f} s, reference {tr:.1f} s, {diff} of {out.size} logits differ{extra}", flush=True)
    n = totals["n"]
    print(f"{a.windows} windows of {S}: integer ppl {math.exp(totals['q'] / n):.2f}, float ppl {math.exp(totals['f'] / n):.2f}, "
          f"top-1 match {totals['agree'] / n:.3f}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
