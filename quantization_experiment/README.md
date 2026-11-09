# GPT-2 int8 format study

Finds which number formats the mini-tpu needs to run GPT-2 small. WikiText-2 test, calibrated on the train split. The short version, with the compiler's results, is [docs/QUANTIZATION.md](../docs/QUANTIZATION.md).

## Result

GPT-2 works in int8 on this hardware's formats, as long as the residual stream stays wide.

| S | fp32 ppl | Fake quant ppl | Increase | Top-1 match with fp32 |
|---|---|---|---|---|
| 256 | 44.63 | 48.54 | +8.8% | 74.5% |
| 1024 | 29.95 | 34.76 | +16.1% | 74.3% |

Final = i8 weights (per channel), i8 GEMM inputs (static, MSE clip), i8 Q/K/V, i32 scores, 255-level `P`, i8-input GELU LUT, 16-bit residual (full range), LayerNorm γ/β folded.

With the whole model in integer arithmetic (`int_model.py`), including LayerNorm ([docs/VEC_LAYERNORM.md](../docs/VEC_LAYERNORM.md)):

| S | `P` levels | fp32 ppl | Integer ppl | Top-1 match with fp32 |
|---|---|---|---|---|
| 256 | 127 | 44.63 | 50.99 | 73.6% |
| 1024 | 255 | 29.95 | 34.61 | 74.4% |

The GPU integer functions are proven bit-exact against `sw/compiler/mini_tpu/golden.py`.

**Calibration noise:** the MSE clip uses a random sample of values. Different samples moved S=256 between 49.10 and 50.99, and S=1024 between 34.58 and 34.63. Calibration is seeded now; the fake-quant numbers above came from unseeded runs, so compare within one run only.

## Format decisions

| Item | Decision | Evidence (ppl, S=1024 unless noted) |
|---|---|---|
| Residual | 16 bits of precision, stored as I32, full range | i8: over 8,000. 16-bit: same as 24-bit. 14-bit: starts to degrade. Clipping position 0 (about 3,000 vs about 60 elsewhere) breaks it |
| Residual add | `ACC_I32` with a bias row, `VEC_ELTWISE MUL` by a row of per-column multipliers, then `ADD` | `VEC_ELTWISE` has one multiplier per tensor, but per-channel weights give each accumulator column its own scale. One shift shared per row costs nothing measurable. No RTL change; a per-lane i32 epilogue mode would save one pass |
| LayerNorm | Fold γ/β into the next GEMM; no γ/β in hardware | Not folded, with i8 activations: 514 vs 34 |
| Attention scores | Keep I32 into softmax | i8 scores: 37.11 vs 34.26 |
| `P` at S ≤ 256 | 127 levels, zp 0, no correction | 48.90 vs 48.54 at S=256 |
| `P` at S = 1024 | 255 levels, needs `P·V` zero-point correction | 38.29 vs 34.76 |
| `P` clipping | Do not clip | Clip 0.5: 46.06 vs 37.65 unclipped |
| GELU | i8 → i8 LUT is enough | Exact input gains only 0.54 |
| Q/K/V | i8 | Wide gains only 0.10 |

## Where the remaining loss is

Keeping one item wide, S=1024 (final = 34.76):

| Kept wide | ppl |
|---|---|
| `ln1` output | 33.17 |
| GELU output | 33.23 |
| LM head weights | 33.75 |
| Anything else | 34.2 or worse |

SmoothQuant is inconsistent: alpha 0.5 helps at S=256 (47.77) and does nothing at S=1024 (34.73); alpha 0.75 helps at S=1024 (33.68) and hurts at S=256 (50.28). It would need a per-column multiply in LayerNorm, so it is not adopted.

## Limits

- Scales depend on the calibration sample (above). The compiler should calibrate on all values, not a sample.
- One dataset, one metric family.
- Greedy text is visibly weaker than fp32. Quantized output fell into repetition sooner in the 6 samples in `results/sweep3_S*.json` (fp32 also loops on one prompt). Sampling with temperature is untested.

## Run

From this folder, with the repository's `.venv` (see the [README](../README.md)):

```
../.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
../.venv/Scripts/python -m pip install -r requirements.txt
../.venv/Scripts/python sanity.py      # forward pass matches Hugging Face
../.venv/Scripts/python sweep3.py      # final fake-quant numbers; sweep.py and sweep2.py are the earlier rounds
../.venv/Scripts/python int_model.py   # integer arithmetic
```

`sweep.py` rows `resid 16` and `resid 12` used the MSE clip, which cuts position 0; those rows are misleading. `sweep2.py` sensitivity rows sit on the clipped `P`; use `sweep3.py` instead.
