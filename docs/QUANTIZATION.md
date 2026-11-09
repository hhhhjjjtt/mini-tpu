# Quantization

How GPT-2 small is turned into integers for this hardware, and what it costs. The study behind these choices, with every experiment, is in [quantization_experiment/](../quantization_experiment/README.md); the code that applies them is `sw/compiler/mini_tpu/model.py`.

## Formats

| Tensor | Format | Why |
|---|---|---|
| Weights | i8, one scale per output column | The matrix unit's input type |
| GEMM inputs (LayerNorm, GELU, attention outputs) | i8, one static scale per tensor, clip chosen by MSE on the calibration set | |
| Q, K, V | i8 | Wider gains only 0.10 ppl |
| Attention scores | i32 into softmax | i8 scores: 37.11 vs 34.26 ppl at S = 1024 |
| Softmax output `P` | 127 levels, zero point 0 | Enough up to S = 256 (48.90 vs 48.54 ppl). S = 1024 needs 255 levels and a zero-point correction in `P·V` |
| Residual stream | i32 holding 16 bits of precision, full range | i8: over 8,000 ppl. Position 0 is about 50 times larger than the rest, so it must not be clipped |
| LayerNorm γ, β | Folded into the next GEMM | No γ/β in hardware; not folding, with i8 activations: 514 ppl |
| GELU | i8 → i8 lookup table | An exact input gains only 0.54 ppl |

Scales are per tensor and fixed after calibration on 16,384 tokens of WikiText-2 train. The residual add uses `ACC_I32` with a bias row, then `VEC_ELTWISE MUL` by per-column multipliers and `ADD`, because per-column weight scales give each accumulator column its own scale.

## Cost

| Model | S | Float ppl | Integer ppl | Top-1 agreement with float |
|---|---|---|---|---|
| Study, integer model | 256 | 44.63 | 50.99 | 73.6 % |
| Study, integer model, 255-level `P` | 1024 | 29.95 | 34.61 | 74.4 % |
| Compiler output, as run on the KV260 | 128 | 53.26 | 63.25 | 71.9 % |

WikiText-2 test. Shorter windows give higher perplexity because each token sees less context.

Most of the remaining loss is in three tensors: keeping the `ln1` output, the GELU output or the LM head weights wide each gains 1 to 1.6 ppl at S = 1024.

## Limits

- Calibrated at 128 tokens and measured at 128, 256 and 1024. Decode reaches 1024 positions with 127-level `P`, which the study did not measure at that length.
- One dataset. Greedy decoding repeats itself sooner than float; the board samples with temperature 0.8 and top-k 40 instead.
