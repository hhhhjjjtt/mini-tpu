# mini-tpu runtime (KV260)

Runs on the KV260's ARM cores against `mini_tpu.bit` ([INTERFACE_SPEC §7.6](../../docs/INTERFACE_SPEC.md)). Migrated from the FlexNPU v2 runtime.

| Path | Contents |
|---|---|
| `libmtpu/mtpu.h`, `mtpu.c` | Device access: map the control window and the u-dma-buf DMA buffer, boot the firmware, run a program, read perf counters and error info |
| `libmtpu/mtpu_desc.h`, `mtpu_desc.c` | Program header and descriptor builders, used by the bring-up test |
| `tests/mtpu_test.c` | Bring-up test: 12 small programs, checked bit-exact against reference math |
| `run_prefill.c` | Runs the compiled prefill and compares its output with the simulator's, bit for bit |
| `generate.c` | Text generation: prefill, then one decode program per token with the KV cache in DDR. Serves token-id requests on stdin; `-c` checks decode bit for bit |
| `complete.py`, `gpt2_tokenizer.py` | Text frontend: type the start of a line, GPT-2 finishes it. Plain Python 3, no packages needed |
| `tests/test_frontend.py` | Workstation: the tokenizer against Hugging Face `tokenizers`, and `complete.py`'s protocol |
| `Makefile` | Build and board targets |

Shared constants come from [`../../hw/interface/`](../../hw/interface/). The firmware image is compiled in from `hw/fw/build/fw_image.h`.

## Run on the board

On the workstation, build everything and collect it (about 360 MB):

```bash
make dist                       # repository root
scp -r sw/runtime/build/dist/mini-tpu ubuntu@<board>:/home/ubuntu/
```

Build the bitstream first if `make dist` warns that it is old. On the board:

```bash
cd /home/ubuntu/mini-tpu
sudo xmutil unloadapp           # only if an xmutil app is loaded
make load                       # mini_tpu.bit -> fpga_manager; u-dma-buf at 256 MB
make                            # native gcc
make run                        # bring-up test: expect 12 passed, 42552 values, 0 mismatches
make prefill                    # GPT-2, S = 128: expect PASSED, 0 of 6434816 outputs differ
make decode                     # 8-token prompt, then decode to position 1023: expect PASSED, 0 logits differ
make chat                       # type a line; GPT-2 finishes it
```

Needs `sudo`, a native `gcc`, and the u-dma-buf module at `/home/ubuntu/udmabuf/u-dma-buf.ko` (override with `UDMABUF_KO=...`). `make load` is needed once per boot, and it replaces a smaller u-dma-buf.

`make prefill ARGS="-n 3"` repeats the run. `ARGS="-o out.bin"` saves the output for comparison on the workstation.

`make decode` runs 1016 decode steps (about 2 minutes) and prints the time and cycles per token for each bucket. `ARGS="-m 200"` stops at position 200.

## Generating text

```bash
make chat                                           # one line at a time; empty line quits
make chat ARGS='"The meaning of life is"'           # one prompt
make chat ARGS="--temperature 0 --stop line"        # always the most likely token; stop at a newline
```

`complete.py` stops at the first sentence end (`.`, `!` or `?`) or newline, after `--tokens` new tokens (50), at `<|endoftext|>`, or when the context of 1024 tokens is full. It samples from the 40 most likely tokens at temperature 0.8 (`--top-k`, `--temperature`); always taking the most likely token tends to repeat itself.

It starts `build/generate` through `sudo` and talks to it in token ids, one request per line (protocol in `generate.c`). Prompts up to 128 tokens take one prefill; longer ones finish through decode, one token at a time. Each new token runs the decode program for its position's bucket (16, 32, … 1024 positions).

## Tests

| Test | Checks |
|---|---|
| `id` | `HW_ID` = `0x5550544D` ("MTPU"), `HW_ARCH` matches the config |
| `boot` | Firmware written to I-mem and read back, core released, `FW_VERSION` and `FW_ARCH` correct, heartbeat advancing |
| `halt` | One-descriptor program: descriptor fetch from DDR, STATUS, perf counters |
| `dma` | Strided LOAD → vmem → STORE round trip; 64 KB LOAD and STORE bandwidth |
| `gemm` | LOAD ×3 → GEMM → requant → STORE |
| `loop` | Two chunks, `LOOP` ×2, weight ping-pong, waits across all four units |
| `attn` | `LOAD_WGT_VM` transposed and straight, Q·Kᵀ → requant → S·V |
| `smax` | `VEC_SOFTMAX` alone, i8 and i32 with mask |
| `head` | Attention head on-chip, every step checked |
| `tput` | GEMM throughput: MAC/cycle against the 256 peak |
| `rate` | Dispatch rate in cycles per descriptor |
| `err` | Bad opcode → `ERR_BAD_OPCODE`, then `CORE_RESET` recovery |

`id` and `boot` always run first. `CAUSAL`, `VEC_LAYERNORM` and `VEC_LUT` have no bring-up test: `make prefill` is their first run on the board.

## Reading failures

| Symptom | Likely cause |
|---|---|
| `make load`: state other than `operating` | Wrong file name, or an `xmutil` app holds the PL |
| `make load`: no `udmabuf0` | CMA smaller than 256 MB. The target prints `/proc/meminfo`'s CMA lines; see `dmesg`. The model needs about 196 MB: try `UDMABUF_SIZE=210000000` |
| `id` hangs or reads garbage | PL not loaded |
| `boot`: I-mem readback mismatch | AXI-Lite writes not landing; check `address_map.txt` |
| `halt` times out, heartbeat advancing | Firmware cannot fetch the program: DMA memory not reachable over HPC0, or a cache problem |
| Data mismatches in `dma` or `gemm` | Cache maintenance or DMA addressing |
| `run_prefill`: `program built for config` | `model/` and the runtime come from different configs: `make dist` again |
| `run_prefill`: `wrong bitstream?` | `HW_ARCH` differs from the program: rebuild the bitstream |
| `STATUS ... ERR_*` | Printed with the failing descriptor, unit and info words; codes in INTERFACE_SPEC §11 |
| Outputs differ in `run_prefill` | Compare with `-o out.bin`. If `make run` passes, suspect the three new vector ops first |
| `make decode`: logits differ | Prefill right but decode wrong: the per-token cache syncs (input, output) or region 6. Wrong only from a bucket on: that bucket's program |

`run_prefill` and `generate` run on the workstation too, with the simulator in place of the board: `python sw/compiler/tests/test_c_replay.py`.

## Results on the KV260

`mini_tpu.bit`, default config (`HW_ARCH 0x08000f10`), 2026-11-08:

| Run | Result |
|---|---|
| `make run` | 12 passed, 42,552 values, 0 mismatches |
| `dma` | 64 KB LOAD 1,581 MB/s, STORE 1,583 MB/s |
| `tput` | 51,009 cycles, 164.5 MAC/cycle overall, 213.5 while MATRIX busy (83 % of peak) |
| `rate` | 144.3 cycles/descriptor |
| `make prefill` (GPT-2, S = 128) | **PASSED, 0 of 6,434,816 outputs differ**. 103.4 M cycles, 1,034 ms, 68,980 descriptors |

Prefill unit activity: VECTOR busy 100 %, MATRIX busy 78.9 % (waiting 44.1 %), LOAD busy 8.7 %, STORE busy 2.2 %. The vector unit is the bottleneck.

`make decode` (8-token prompt, then decode to position 1023): **PASSED, 16 positions checked, 0 logits differ**, every bucket. Bitstream of 2026-11-09 (the matrix unit overlaps one-tile GEMMs: two pending GEMMs, three weight halves; `HW_ARCH 0x08001f10`), against the first one of 2026-11-08:

| Bucket | Positions | ms per token | M cycles | Tokens/s | Before: ms, tokens/s |
|---|---|---|---|---|---|
| 16 – 128 | 8 … 127 | 123 – 130 | 12.3 – 13.0 | **7.7 – 8.1** | 192 – 200, 5.0 – 5.2 |
| 256 | 128 … 255 | 139 | 13.9 | 7.2 | 209, 4.8 |
| 512 | 256 … 511 | 158 | 15.7 | 6.3 | 227, 4.4 |
| 1024 | 512 … 1023 | 197 | 19.6 | 5.1 | 266, 3.8 |

Prefill (S = 128): 1,010 ms, 100.5 M cycles (was 1,034 ms).

Unit activity, buckets 16 – 64: MATRIX busy 98.7 %; VECTOR busy 99.8 % (each GEMM's epilogue runs with it); LOAD busy 72 % (was 46 %): the weight DMA is now close to its limit; STORE 0.1 %. About 19,700 one-tile GEMMs per token at about 630 cycles each (was about 970). Longer contexts add attention work: LOAD busy falls to 52 % at 1024.

Before, the GEMMs cost about twice their weight-streaming time: each waited for the previous GEMM's rows to reach VECTOR, and weight loads and activation fills serialized over two weight halves (`hw/sim/README.md`, "Decode").

`make chat`: "I work in a lab called HPLP" → "I work in a lab called HPLP, which is funded by the US Department of Health and Human Services.", 14 tokens, prompt 1,010 ms, 124 ms per token (8.1 tokens/s). The first bitstream gave 192 ms per token.

