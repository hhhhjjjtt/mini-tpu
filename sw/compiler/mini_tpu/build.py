"""Stage 2: calibrated model + hardware config -> everything the TPU and the ARM need, in output/.

  program.bin      prefill: header and descriptors
  decode_<L>.bin   one decode program per context bucket L (codegen.buckets)
  weights.bin      region 0, shared by every program
  params.bin       region 1, shared
  embed.bin        for the ARM: wte then wpe (ctx rows), f32
  logit_scale.bin  for the ARM: f32 per vocabulary column
  model.txt        for the ARM: sizes, buckets, s_embed; one "key value ..." per line
  tokenizer.json   for the ARM's frontend: the model's tokenizer, copied
  manifest.json    sizes, shapes, scales, hardware
  golden_input.bin, golden_output.bin   prefill check: one input and its expected output, raw i32
  golden_ids.bin                        decode check: ctx token ids, i32
  golden_decode.bin                     decode check: i32 prompt length, count, positions[count],
                                        then the expected logits at each position, i32 [vocab_pad]

usage: python -m mini_tpu.build
"""
import json
import shutil

import numpy as np

from . import calibrate, codegen, config, model

DECODE_PROMPT = 8           # decode check: prompt length; later tokens go through decode one at a time


def check_positions(prompt, buckets, ctx):
    """Positions whose logits the decode check compares: the prompt's last, the first decode
    steps, and both sides of every bucket boundary."""
    p = {prompt - 1, prompt, prompt + 1, ctx - 1} | {x for L in buckets for x in (L - 1, L)}
    return sorted(x for x in p if prompt - 1 <= x < ctx)


def main():
    c = config.load()
    fields, weights, scales, tokens = calibrate.load(c.calibrated)
    cfg = model.GPTConfig(seq=c.seq, **fields)
    q = model.quantize(cfg, weights, scales)
    out = codegen.compile_model(q, codegen.Target.from_hardware(c.hardware))
    pre, ctx = out.prefill, out.ctx

    d = c.output
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("decode_*.bin"):
        f.unlink()
    (d / "program.bin").write_bytes(pre.program)
    for L, dec in out.decode.items():
        (d / f"decode_{L}.bin").write_bytes(dec.program)
    (d / "weights.bin").write_bytes(pre.weights)
    (d / "params.bin").write_bytes(pre.params)
    wpe = np.asarray(q.wpe[:ctx], dtype="<f4")
    (d / "embed.bin").write_bytes(np.asarray(q.wte, dtype="<f4").tobytes() + wpe.tobytes())
    scale = q.head.s_in * q.head.s_w[:cfg.vocab]
    (d / "logit_scale.bin").write_bytes(scale.astype("<f4").tobytes())

    ids = tokens[:ctx].astype(np.int64)
    x = q.embed(ids[:c.seq])
    (d / "golden_input.bin").write_bytes(x.astype("<i4").tobytes())
    (d / "golden_output.bin").write_bytes(model.forward(q, x).astype("<i4").tobytes())
    pos = check_positions(DECODE_PROMPT, list(out.decode), ctx)
    rows = model.forward(q, q.embed(ids), rows=pos)
    (d / "golden_ids.bin").write_bytes(ids.astype("<i4").tobytes())
    (d / "golden_decode.bin").write_bytes(np.array([DECODE_PROMPT, len(pos), *pos], dtype="<i4").tobytes() + rows.astype("<i4").tobytes())

    dec = next(iter(out.decode.values()))
    info = {"d_model": cfg.d_model, "n_layer": cfg.n_layer, "vocab": cfg.vocab, "vocab_pad": cfg.vocab_pad, "seq": c.seq, "ctx": ctx,
            "s_embed": repr(float(q.s_embed)), "kv_bytes": out.kv_bytes, "prefill_input": pre.input_bytes,
            "prefill_output": pre.output_bytes, "decode_input": dec.input_bytes, "decode_output": dec.output_bytes,
            "buckets": " ".join(map(str, out.decode))}
    (d / "model.txt").write_text("".join(f"{k} {v}\n" for k, v in info.items()), encoding="utf-8")
    shutil.copy(c.model_path / "tokenizer.json", d / "tokenizer.json")

    manifest = {
        "model": c.model_name, "seq": c.seq, "ctx": ctx, "d_model": cfg.d_model, "vocab": cfg.vocab, "vocab_pad": cfg.vocab_pad,
        "s_embed": q.s_embed, "descriptors": pre.descriptors, "vmem_used": pre.vmem_used,
        "decode": {str(L): {"descriptors": x.descriptors, "vmem_used": x.vmem_used, "bytes": len(x.program)} for L, x in out.decode.items()},
        "config_hash": f"0x{c.hardware.hash:016X}", "arch_word": f"0x{c.hardware.arch_word:08X}",
        "hardware": {k: getattr(c.hardware, k) for k in ("name", "array_n", "vector_lanes", "vmem_kb", "max_k", "max_len", "fifo_depth")},
        "regions": {"0": "weights.bin", "1": "params.bin",
                    "2": {"prefill": "i32 [seq, d_model]", "decode": "i32 [d_model], then mask i32 [ctx]"},
                    "3": {"prefill": "i32 [seq, vocab_pad]", "decode": "i32 [vocab_pad]"},
                    "5": {"kv_bytes": out.kv_bytes, "layout": "per layer: K then V, i8 [ctx, d_model]"},
                    "6": "region 5 + position * d_model"},
        "bytes": {"program": len(pre.program), "weights": len(pre.weights), "params": len(pre.params)},
        "embed": {"wte_rows": cfg.vocab, "wpe_rows": ctx, "cols": cfg.d_model},
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {d}: prefill {pre.descriptors} descriptors, decode {min(x.descriptors for x in out.decode.values())}-"
          f"{max(x.descriptors for x in out.decode.values())} for buckets {list(out.decode)}, "
          f"{len(pre.weights) >> 20} MB weights, {max(x.vmem_used for x in (pre, *out.decode.values())) >> 10} KB vmem")


if __name__ == "__main__":
    main()
