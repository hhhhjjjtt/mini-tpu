# VEC_LAYERNORM design notes

Integer row LayerNorm without γ/β, which the compiler folds into the next GEMM. Input is the I32 residual, output is i8. The normative definition and descriptor are in [INTERFACE_SPEC.md §5.7.1](INTERFACE_SPEC.md). Reference: `layernorm` in [golden.py](../sw/compiler/mini_tpu/golden.py).

## Algorithm

Per row of `n` values:

```
x      = clamp(in, -(2^20 - 1), 2^20 - 1)
pass 1 S1 = sum(x), S2 = sum(x * x)                     exact
       D  = n * S2 - S1 * S1 + eps                      n^2 * (var + eps), exact
       mean = apply_scale(S1, inv_mult, inv_shift - 8)  Q8 mean
row    c  = pairs of bits until D << 2c >= 2^62         normalize, no barrel shifter
       q  = floor(sqrt((D << 2c) >> 32))                16 steps, in [2^15, 2^16)
       R  = floor(2^46 / (2 * q))                       softmax divider, 47 steps
pass 2 u   = apply_scale((x << 8) - mean, R, 45 - c)    x_hat / n in Q24
       out = clamp(apply_scale(u, c_mult, c_shift) + out_zp, -128, 127)
```

Compile-time values come from `layernorm_params` in golden.py.

## Hardware

| Reused | New |
|---|---|
| Row sequencing and read engine of `VEC_SOFTMAX` | Accumulators for S1, S2; adder trees over the lanes |
| Lane multipliers (x · x in pass 1, S1², mean, pass 2 scale) | n · S2 by shift-add |
| Output stage: `out_mult`, `out_shift`, clamp | Normalize loop |
| Reciprocal divider | 16-step bit-serial square root |

Measured on the RTL: 1,558 cycles for a 768-element i32 row (32.5 per 16-element block), about 100 ms per S=256 forward pass.

Bounds: |x| < 2²⁰ and n ≤ 1024 keep every value in 64 bits. The largest |x| seen on GPT-2 is 72,025.

## Results

GPT-2 small, whole model in integer arithmetic, WikiText-2 test. Float and integer LayerNorm share the same calibration.

| S | `P` levels | Float LayerNorm ppl | Integer LayerNorm ppl |
|---|---|---|---|
| 256 | 127 | 50.99 | 50.99 |
| 1024 | 255 | 34.63 | 34.61 |

On 19.7 M outputs from real residuals, 0.020% differ from float LayerNorm, each by one i8 step. Without the 8 fractional bits on the mean, 3.7% differ.

The GPU port in `quantization_experiment/int_model.py` matches golden.py on 2.4 M values, including constant rows.
