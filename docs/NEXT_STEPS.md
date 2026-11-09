# mini-tpu next steps

Target: GPT-2 on KV260. The ARM runs the token loop (embedding, sampling, per-token region bases). iverilog is for lightweight tests only; the full model runs in the Python simulator and on the board.

1. **Migrate (done):** `rtl`, `sim`, `fw`, `interface`, `fpga` are in `hw/`, renamed to mini_tpu. Regression matches the baseline on both cores. Git is deferred.
2. **Numerics study (done):** results and format decisions are in [QUANTIZATION.md](QUANTIZATION.md).
3. **Python simulator and encoder (done):** in [sw/compiler/](../sw/compiler/README.md). Matches the KV260 test suite and the RTL bit for bit.
4. **Spec and RTL (done):** `CAUSAL`, `VEC_LAYERNORM` and `VEC_LUT` are in [INTERFACE_SPEC.md](INTERFACE_SPEC.md) and the RTL, bit-exact against the simulator. KV260 bitstream built: timing met (WNS +0.45 ns), CLB 85 %. Bit-exact on the board (step 8).
5. **Compiler (done):** [sw/compiler/](../sw/compiler/README.md) turns GPT-2 into a program and weights. The tiny GPT matches on the RTL, GPT-2 prefill (S = 128) matches its reference in the Python simulator. Bit-exact in both.
6. **Config and build (done):** `config.toml` drives the RTL, firmware and compiler; `make` writes the program and data to `sw/compiler/output/`. See [BUILD.md](BUILD.md).
7. **Parameterize the RTL (done):** `array_n` and `vector_lanes`, with their adapters. The default config is cycle-identical. Legal values are in [BUILD.md](BUILD.md). Other configs have not been compiled or simulated.
8. **Board (done):** [sw/runtime/](../sw/runtime/README.md) on the KV260: bring-up 12 / 12, GPT-2 prefill (S = 128) bit-exact against the simulator in 1,034 ms. The vector unit is the bottleneck (100 % busy).
9. **Decode:** KV cache in DDR, decode programs for 7 context buckets, `generate` on the ARM, and a text frontend ([sw/runtime/](../sw/runtime/README.md)). Bit-exact against the reference in the simulator, on the RTL (tiny GPT) and on the KV260 (`make decode`, positions 7 to 1023). 5.1 tokens/s up to 128 positions, 3.8 at 1024; `make chat` works. Decode is bound by its ~19,700 one-tile GEMMs per token. Next: weights as the streamed operand (a few large GEMMs per layer), about 10–12 tokens/s if weight DMA becomes the limit.
10. **Stretch:** extend the compiler, simulator and testbenches past 16, then simulate and build other hardware configs.
