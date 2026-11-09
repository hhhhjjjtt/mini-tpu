"""The ARM's token loop on the simulator; sw/runtime/generate.c does the same on the KV260.

Prefill runs the prompt, padded to S, and fills the KV cache. Each later token runs the decode
program of the smallest bucket L > its position: the ARM writes the token's embedding and the
mask to region 2 and points region 6 at the token's KV cache row.
"""
from types import SimpleNamespace

import numpy as np

from . import codegen
from .isa import C
from .sim import Machine

ALIGN = 4096


def layout(model):
    """Offsets in the DMA buffer: prefill, decode programs by bucket, weights, params, input, output,
    KV cache. Returns ({name: offset}, total bytes)."""
    off, top = {}, 0
    p, dec = model.prefill, model.decode.values()
    parts = [("prefill", len(p.program))] + [(L, len(c.program)) for L, c in model.decode.items()] + [
        ("weights", len(p.weights)), ("params", len(p.params)),
        ("input", max(c.input_bytes for c in (p, *dec))), ("output", max(c.output_bytes for c in (p, *dec))),
        ("kv", model.kv_bytes)]
    for name, size in parts:
        off[name] = top
        top += -(-size // ALIGN) * ALIGN
    return off, top


def mask_row(ctx, pos):
    """i32 [ctx]: 0 up to pos, MASKED after it."""
    return np.where(np.arange(ctx) <= pos, 0, codegen.MASKED).astype("<i4")


def load(d):
    """A build output folder as the ARM sees it. Returns (codegen.Model, dims) where dims has
    seq, d_model, vocab_pad and embed(ids, pos), computed from embed.bin and model.txt."""
    info = {k: v for k, _, v in (line.partition(" ") for line in (d / "model.txt").read_text(encoding="utf-8").splitlines())}
    num = lambda k: int(info[k])
    D, ctx, vocab = num("d_model"), num("ctx"), num("vocab")
    w, p = (d / "weights.bin").read_bytes(), (d / "params.bin").read_bytes()
    prog = lambda name, i, o: codegen.Compiled((d / name).read_bytes(), w, p, num(i), num(o), 0, 0, num("kv_bytes"))
    model = codegen.Model(prog("program.bin", "prefill_input", "prefill_output"),
                          {int(L): prog(f"decode_{L}.bin", "decode_input", "decode_output") for L in info["buckets"].split()}, ctx)
    table = np.fromfile(d / "embed.bin", dtype="<f4")
    wte, wpe = table[:vocab * D].reshape(vocab, D), table[vocab * D:].reshape(ctx, D)
    s_embed = float(info["s_embed"])

    def embed(ids, pos=0):
        x = wte[ids].astype(np.float64) + wpe[pos:pos + len(ids)]
        return np.clip(np.round(x / s_embed), -32767, 32767).astype(np.int32)

    return model, SimpleNamespace(seq=num("seq"), d_model=D, vocab_pad=num("vocab_pad"), embed=embed)


class Session:
    """One compiled model in one Machine. dims: seq, d_model, vocab_pad and embed(ids, pos), such as
    a model.QuantModel's cfg and embed, or what load() returns."""

    def __init__(self, model, dims, base=0x10000, **machine_kw):
        self.model, self.dims = model, dims
        self.off, size = layout(model)
        self.base = base
        self.m = m = Machine(ddr_bytes=base + size, **machine_kw)
        p = model.prefill
        for r, name in ((codegen.R_WGT, "weights"), (codegen.R_PRM, "params"), (codegen.R_IN, "input"),
                        (codegen.R_OUT, "output"), (codegen.R_KV, "kv")):
            m.region[r] = self.addr(name)
        m.write_ddr(self.addr("prefill"), p.program)
        for L, c in model.decode.items():
            m.write_ddr(self.addr(L), c.program)
        m.write_ddr(self.addr("weights"), p.weights)
        m.write_ddr(self.addr("params"), p.params)
        self.runs = []                          # (program, descriptors, races, rule violations, uninitialized reads)

    @classmethod
    def of(cls, q, model, **kw):
        """For a model.QuantModel."""
        c = q.cfg
        return cls(model, SimpleNamespace(seq=c.seq, d_model=c.d_model, vocab_pad=c.vocab_pad, embed=q.embed), **kw)

    def addr(self, name):
        return self.base + self.off[name]

    def _run(self, name, out_bytes):
        m = self.m
        status = m.run(self.addr(name))
        if status != C.ST_DONE:
            raise RuntimeError(f"{name}: status {status:#x} {m.message}")
        k = m.checker
        self.runs.append((name, m.desc_count) + ((k.races, k.rule_violations, k.count.get("UNINIT", 0)) if k else (0, 0, 0)))
        return np.frombuffer(m.read_ddr(self.addr("output"), out_bytes), dtype="<i4").copy()

    def prefill_x(self, x):
        """x: i32 [S, D] embeddings. Returns i32 logits [S, vocab_pad]."""
        self.m.write_ddr(self.addr("input"), x.astype("<i4").tobytes())
        return self._run("prefill", self.model.prefill.output_bytes).reshape(self.dims.seq, self.dims.vocab_pad)

    def prefill(self, ids):
        """ids: the prompt, 1 ... S tokens, padded with token 0. Returns the i32 logits of its last token."""
        S, n = self.dims.seq, len(ids)
        assert 1 <= n <= S
        return self.prefill_x(self.dims.embed(np.concatenate([ids, np.zeros(S - n, dtype=np.int64)])))[n - 1]

    def step(self, token, pos):
        """token at position pos, after positions 0 ... pos - 1 are in the cache. Returns its i32 logits."""
        model = self.model
        L = min(L for L in model.decode if L > pos)
        x = self.dims.embed(np.array([token]), pos)[0].astype("<i4")
        self.m.write_ddr(self.addr("input"), x.tobytes() + mask_row(model.ctx, pos).tobytes())
        self.m.region[codegen.R_KV_ROW] = self.addr("kv") + pos * self.dims.d_model
        return self._run(L, model.decode[L].output_bytes)
