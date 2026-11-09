# mini-tpu compiler tools

Compiles a GPT into a program and DDR images, and runs and checks programs. NumPy only, except calibration. `make` in the repository root drives it: see [docs/BUILD.md](../../docs/BUILD.md).

| File | Purpose |
|---|---|
| `mini_tpu/config.py` | Reads and validates `config.toml` |
| `mini_tpu/calibrate.py` | Stage 1: Hugging Face files and text to folded weights and scales |
| `mini_tpu/build.py`, `check.py` | Stage 2: writes `output/`; runs it in the simulator against its golden data |
| `mini_tpu/model.py` | Quantization to integers and multipliers; the reference forward pass a program must match; float model |
| `mini_tpu/codegen.py` | The compiler: `Target` (hardware limits), `Emitter` (derives waits from vmem use), `compile_model` (prefill and the decode programs, sharing weights), `compile_gpt` (prefill alone) |
| `mini_tpu/session.py` | The ARM's token loop on the simulator: prefill, then one decode program per token. `load` reads an output folder as the ARM does |
| `mini_tpu/isa.py` | Descriptor encoders and decoder, `build_program` (header, chunks). Constants are parsed from `hw/interface/*.h` |
| `mini_tpu/golden.py` | Bit-exact arithmetic: requant, eltwise, softmax, LayerNorm, LUT, GEMM |
| `mini_tpu/sim.py` | `Machine`: runs a program blob from DDR, following the firmware and issue engine. No timing |
| `mini_tpu/hazards.py` | Race checker: reports unordered RAW / WAR / WAW in vmem, weight halves and DDR, wait-rule violations, uninitialized reads |

```python
from mini_tpu import isa
from mini_tpu.sim import Machine

m = Machine(ddr_bytes=1 << 20)
m.region[:4] = [0x10000, 0x20000, 0x30000, 0x40000]
m.write_ddr(0x1000, isa.build_program(descs))
status = m.run(0x1000)                     # C.ST_DONE, or C.ST_ERROR | code << 8
m.checker.races, m.checker.findings
```

To compile a model from Python:

```python
from mini_tpu import codegen, model

q = model.quantize(cfg, weights, scales)   # folded float weights, one scale per tensor
c = codegen.compile_gpt(q)                 # program, weights (R0), params (R1)
status = codegen.run(m, c, q.embed(token_ids), (prog, r0, r1, r2, r3))
```

A hang (for example a GEMM without its epilogue) ends as `ERR_WATCHDOG`, with the state in `m.message`. `Machine(order_seed=n)` randomizes the execution order.

## Tests

Run from this folder.

| Command | Time | Checks |
|---|---|---|
| `PYTHONPATH=. python -m unittest tests.test_sim` | 1 s | The race checker finds real races and stays quiet on safe programs; errors; chunking |
| `PYTHONPATH=. python -m unittest tests.test_decode` | 4 s | Tiny GPT: prompt through prefill, then decode one token at a time up to the full context, every bucket; bit-exact against the reference, race-free, any execution order |
| `python tests/test_c_replay.py` | 6 min | `sw/runtime` with the simulator in place of the board (needs host `gcc`): the bring-up test, `run_prefill`, and `generate` (decode check, one request) on the compiled model. Python encoders must reproduce every C descriptor |
| `python tests/rtl_diff.py --seed N` | 6 min | Random programs on the simulator and the RTL (`make sim_blob`), all of DDR compared. Needs iverilog and the RISC-V gcc |
| `PYTHONPATH=. python tests/mutation_check.py --seed N` | 30 s | After `rtl_diff.py` with the same seed: injected simulator bugs must change the result |
| `python tests/bench_sim.py` | 2 s | Simulator speed |
| `PYTHONPATH=. python -m unittest tests.test_tiny_gpt` | 1 s | Small random GPTs, compiled: bit-exact against the reference, race-free, any execution order |
| `python tests/rtl_gpt.py` | 5 min | The tiny GPT on the RTL: all of DDR against the simulator, logits against the reference |
| `python tests/rtl_decode.py` | 10 min | The tiny GPT decoding on the RTL: 2 prefills and 5 decode steps across 3 buckets; all of DDR against the simulator, logits against the reference |
| `python tests/run_gpt2.py` | 2 min | GPT-2 small on real text, after `make`. `--prompt` generates text |

## Results (2026-11-05)

- `test_c_replay`: 12 / 12 tests, 42,552 values bit-exact, for each of 3 seeds. The same count as the KV260. 364 descriptors re-encoded byte for byte.
- `rtl_diff`: seeds 1 and 2, 154 programs: 0 of 1 MB DDR bytes differ, statuses and descriptor counts equal.
- `mutation_check`: all 15 injected bugs detected.
- `bench_sim`, c_fc layer at S=256: 0.11 ms per descriptor, 0.22 ms with the race checker.

## GPT-2 (2026-11-06)

| Run | Result |
|---|---|
| Tiny GPT (2 layers, width 64, S=32), RTL | 148,950 cycles, 390 descriptors. 0 DDR bytes differ from the simulator, 0 of 8,192 logits differ from the reference |
| GPT-2 small, S=128, simulator | 68,711 descriptors, 4.2 MB program, 118 MB weights, 1,791 KB vmem. 4 windows of WikiText-2: 0 of 6.4 M logits differ per window, no races |
| Quality on those 512 tokens | Integer ppl 65.54, float ppl 60.70, top-1 match 75.2% |

Compile time is 8 s; a pass takes about 21 s in the simulator with the race checker.

## Decode (2026-11-08)

| Run | Result |
|---|---|
| Tiny GPT, context 64, simulator (`test_decode`) | Prompt of 5, then decode at positions 5 … 63 through buckets 16, 32, 64: every logit matches the reference; no races, rule violations or uninitialized reads; also in random execution orders and with P·V split over two weight halves |
| Tiny GPT, RTL (`rtl_decode`) | 2 prefills (149 k cycles) and 5 decode steps (40.5 k to 47.1 k cycles): 0 DDR bytes differ from the simulator, 0 logits from the reference |
| GPT-2, simulator (`make check`) | Prompt of 8, decode at positions 8 and 9: 0 of 150,816 logits differ; no races |
| GPT-2, C replay (`generate -c`) | The same through `generate.c`: 0 logits differ. A greedy request returns the reference's most likely token |

GPT-2: prefill 68,735 descriptors; decode 59,554 (bucket 16) to 88,498 (bucket 1024), 3.7 to 5.4 MB each; vmem up to 1,970 KB. Weights are not duplicated: all programs read one `weights.bin`. Compiling all eight programs takes 34 s.

Compiler limits for now: prefill S ≤ 128 (vmem; no tiling yet), context ≤ 1024, 127-level `P`, fully unrolled. Prefill does not skip the causal upper triangle and computes the LM head for every row. Decode computes 16 rows to use 1 and issues 60 k to 88 k descriptors per token.

## Not covered

- Timing: `Machine` has no cycle model.
- `BREAK`: resumes at once, through the optional `on_break` hook.
- CORE_RESET, ABORT and AXI errors. Unit error reports are not cycle-accurate: the failing descriptor in `ERR_DESC` can differ from hardware.
