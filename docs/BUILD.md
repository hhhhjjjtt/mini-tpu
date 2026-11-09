# Build

`config.toml` in the repository root describes the hardware, the model and the build.

```bash
make model        # once: download GPT-2 and calibration text into sw/compiler/model/
make              # compile into sw/compiler/output/<hardware>_<model>_s<seq>/
make check        # run that output in the simulator against its golden data: prefill, then 2 decode steps
make test         # fast checks, about 15 s: simulator, tiny GPT prefill and decode, the board's tokenizer
make sim          # RTL regression
make bitstream    # builds the firmware and prints the Vivado command to run
make dist         # one folder to copy to the KV260: see sw/runtime/README.md
```

## Stages

| Stage | Input | Output | Reruns when |
|---|---|---|---|
| Config | `config.toml` | `hw/interface/mini_tpu_config.h`, `hw/rtl/common/mini_tpu_config.svh` | Every `make`; files change only if `[hardware]` does |
| Calibrate (minutes) | Model, calibration text | `output/calibrated_<model>_s<seq>.npz` | `[model]` or `[build]` changes |
| Compile (seconds) | Calibrated model, `[hardware]` | `output/<hardware>_<model>_s<seq>/` | Anything above changes |

The calibrated file is the seam for other frontends: anything that writes it can use the compiler. Stage 1 needs `tokenizers`; `make` uses the repository's `.venv` (see the [README](../README.md)).

## Output

| File | Goes to | Contents |
|---|---|---|
| `program.bin` | DDR, `PROG_BASE` | Prefill: header and descriptors |
| `decode_<L>.bin` | DDR, `PROG_BASE` | Decode, one program per context bucket L = 16, 32, … 1024 |
| `weights.bin` | DDR, region 0 | Tiled i8 weights, shared by all programs |
| `params.bin` | DDR, region 1 | Requant tables, bias and multiplier rows, GELU tables |
| `embed.bin` | ARM | `wte` then `wpe` (1024 rows), f32 |
| `logit_scale.bin` | ARM | f32 per vocabulary column |
| `model.txt` | ARM | Sizes, buckets, `s_embed`: one `key value` per line |
| `tokenizer.json` | ARM | The model's tokenizer, for the text frontend |
| `manifest.json` | People | Sizes, layouts, scales, hardware, config hash |
| `golden_input.bin`, `golden_output.bin` | Testing | Prefill: one input and its expected output, raw i32 |
| `golden_ids.bin`, `golden_decode.bin` | Testing | Decode: 1024 token ids; expected logits at the positions around each bucket boundary |

The embedding of token `id` at position `pos` is `round((wte[id] + wpe[pos]) / s_embed)`, clamped to ±32767, as i32. Multiply logits by `logit_scale.bin`.

| Region | Prefill | Decode |
|---|---|---|
| 2, input | i32 `[seq, d_model]` | i32 `[d_model]`, then the mask, i32 `[1024]`: 0 up to this position, −2³¹ after |
| 3, output | i32 `[seq, vocab_pad]` | i32 `[vocab_pad]` |
| 5, KV cache | Writes rows 0 … seq − 1 | Reads rows 0 … L − 1 |
| 6, KV row | — | Region 5 + position × `d_model`: this token's row |

The KV cache holds, per layer, K then V as i8 `[1024, d_model]`; 18 MB for GPT-2. Decode at position `p` runs the smallest bucket `L > p`. Positions after `p` in that window hold stale data, which the mask blocks exactly as `CAUSAL` would.

## Hardware parameters

Generated headers are checked in so `hw/` builds without Python; `make` keeps them current. A program's header carries the hardware's arch word and a hash of `[hardware]` in `compiler_hash`. `make check`, `run_prefill` and `generate` on the board compare that hash.

| Parameter | Legal values | Effect on the RTL |
|---|---|---|
| `array_n` | 8, 16, 32, 64 | Array size. 8 keeps 16 x 16 tiles and runs each in 4 passes, so programs do not change |
| `vector_lanes` | 4, 8, 16, 32, 64 | Vector lanes. Below 4 an i32 block would not fill a vmem word |
| `vmem_kb` | 256 ... 2048, power of two | 256 KB banks |
| `max_k` | 16 ... 1024, power of two | Weight and staging buffer depth |
| `max_len` | Multiple of 16 and `vector_lanes`, up to 1024 | Longest softmax and LayerNorm row |
| `fifo_depth` | 2 ... 64 | Unit command FIFOs |

vmem words and DMA stay 128 bits, so a wider array reads activations and loads weights at a lower rate per column: `array_n = 32` takes two cycles per k. xck26 has 1,248 DSPs; a 32 x 32 array uses 1,024 and each lane about 17, so `array_n = 32` needs `vector_lanes` of 8 or fewer.

The RTL takes all of these, but only the defaults have been simulated or built. The compiler and simulator handle `array_n` of 8 or 16 and `vector_lanes = 16`: `make` stops with an error at the compile stage otherwise. The testbenches have the same limits.
