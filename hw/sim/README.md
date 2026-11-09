# mini-tpu simulation

Self-checking iverilog testbenches for the RTL in [`../rtl/`](../rtl/) and the top level in [`../fpga/top/`](../fpga/top/). Sources come from [`../rtl/files.mk`](../rtl/files.mk).

```bash
cd hw/sim
make sim_matrix                    # matrix unit against a reference GEMM
make sim_vector                    # vmem + matrix + vector end to end
make sim_system                    # mini_tpu_top + firmware + DDR model (builds ../fw first)
make sim_blob BLOB=<dir>           # run compiled programs, dump DDR and unit counters (tb/tb_blob.sv); PLUSARGS=+nostall turns off the DDR model's random stalls
make sim_matrix_timing             # per-tile cycle breakdown of the matrix unit (timing only)
make sim_all                       # matrix, vector and system TBs
```

Outputs go to `build/`; the compiler's RTL tests use `build/blob/`.

| File | Tests |
|---|---|
| `tb/tb_matrix_unit.sv` | Front end + matrix unit: k = 16 / 48 / 64 / 512, 1–5 M-tiles, padded row pitch, both weight halves, back-to-back commands, a wait condition held until another unit's DONE advances, `LOAD_WGT_VM` straight (while the LOAD_WGT port writes the other half; contention asserted) and transposed, each followed by checked GEMMs, a bad-field rejection. Random vmem grant stalls, response latency and output backpressure |
| `tb/tb_vector_path.sv` | vmem → matrix unit → stream → vector unit → vmem, 1,984 values bit-exact against golden models of INTERFACE_SPEC §9, plus `VEC_SOFTMAX`: 1,632 outputs bit-exact against the §5.7 reference model and compared with double-precision softmax (details below) |
| `tb/tb_system.sv` | System test of `mini_tpu_top`: controller + firmware, four front ends, LOAD, matrix, vector, STORE, vmem (programs below) |
| `tb/tb_blob.sv` | Runs a compiled program on `mini_tpu_top` and dumps DDR (not in `sim_all`; `sw/compiler/tests/rtl_diff.py` uses it) |
| `tb/axi_ddr_model.sv` | AXI4 DDR model used by `tb_system` and `tb_blob` |
| `tb/tb_matrix_timing.sv` | Timing only (not in `sim_all`): splits every M-tile of the matrix unit into lead, reads, read bubbles, drain, writeback, finish and handoff, plus per-GEMM first-tile fill and output tail, with ideal vmem and consumer ([below](#matrix-unit-per-tile-timing)) |

## tb_vector_path

GEMM + `REQUANT_I8` (per-channel, `DOUBLE_ROUND`), a K = 1536 split into 3 GEMMs (`STORE_I32` → in-place `ACC_I32` → `ACC_REQUANT_I8`, with a weight reload overlapping the middle GEMM), `PER_TENSOR` + ReLU clamp, six `VEC_ELTWISE` variants (ADD / MUL-ROW / SUB-SCALAR-i32 / MAX / MIN-mixed-types / PASS_A), and `ERR_STREAM_MISMATCH`. Random traffic on the unused LOAD (bank 7) and STORE (bank 0) ports forces bank contention.

Softmax (T6): i8 input len 64; i32 input with a mask (including −10000 blocked positions) and padded pitches; a uniform row and a single dominant element (largest reciprocal); len 1024; flat rows where every output is mid-range; a BF16 `out_type` rejected with `ERR_BAD_FIELD`. Against double precision, per run: 357–386 outputs above −128, 1–4 differ, each by one i8 step (the run fails if any differs by more).

## tb_system

The ARM side (AXI-Lite tasks) loads the real firmware (`../fw/build/fw.hex`) through the I-mem window, places programs in DDR, rings the doorbell and polls `STATUS`. DDR is `tb/axi_ddr_model.sv`: three AXI4 ports (fetch, LOAD, STORE), random latency (8–40 cycles) and stalls, an SLVERR window, and protocol checks (valid held with a stable payload until ready, 4 KB rule, WLAST) — the run fails on any violation.

| Program | Covers |
|---|---|
| P1 | LOAD ×3 → GEMM → `VEC_EPILOGUE` → STORE → HALT; output checked in DDR |
| P2 | Two chunks with `NEXT_CHUNK` prefetch, `LOOP` ×2 over a body unrolled for weight ping-pong, region strides, RAW / WAR waits across all four units, `FENCE`; 2,048 outputs checked |
| P3 | Dispatch rate: `LOOP` ×64 over four tiny ops; prints cycles per descriptor |
| P4 | `BREAK` / `RESUME`, `IRQ_STATUS`, PL→PS IRQ, trace record |
| P5 | Errors, each followed by `CORE_RESET` recovery: bad magic, bad opcode, unset region, unit error (`VEC_SOFTMAX` with `rows` = 0 → `ERR_BAD_FIELD`), watchdog (GEMM with no consumer), `ABORT` at a `FENCE`, bad `ddr_region` (issue-engine stop), `ABORT` while the issue engine is blocked on a full FIFO |
| P6 | DMA: strided round trip (rows crossing 4 KB, gaps between destination rows untouched), 64 KB bandwidth, LOAD / LOAD_WGT / STORE field errors, SLVERR on read and write, `CORE_RESET` in the middle of a 64 KB LOAD and a 64 KB STORE followed by a clean round trip |
| P7 | One attention head on-chip: `LOAD_WGT_VM` transposed (Kᵀ) → `GEMM` Q·Kᵀ → requant to i8 scores in vmem → `LOAD_WGT_VM` straight (V) → `GEMM` S·V waiting on VECTOR → i32 → STORE; 512 outputs checked |
| P8 | GEMM throughput: A 256 × 512 · W 512 × 64 as 4 weight tiles (next tile prefetched into the other half), same program as `sw/runtime/tests/mtpu_test.c` `tput`; sampled rows checked. Simulation with cross-GEMM prefill: 53.1 k cycles, 158 MAC/cycle overall, 213.5 MAC/cycle while MATRIX is busy (83 % of the 256 peak); before prefill 54.7 k / 153 / 205 |
| P9 | Attention head with softmax, same program as `sw/runtime/tests/mtpu_test.c` `head`: 2 × (`LOAD_WGT_VM` Kᵀ + GEMM + requant) → `VEC_SOFTMAX` → `LOAD_WGT_VM` V → GEMM P·V waiting on VECTOR → i32 → STORE; 512 outputs checked against the §5.7 model |
| P1 | Rerun after all resets |

Measured issue rate:

| | Firmware only (before) | With issue engine |
|---|---|---|
| Push-to-push, median | 550 cycles | **7 cycles** |
| P2 (K = 128, 4 weight tiles, 2 M-tiles), `RUN` → `DONE` | 13.9 k cycles; `UNIT_WAIT` = 0 (dispatch-bound) | **5.4 k cycles** with testbench DMA stand-ins; 5.6 k with the RTL DMAs and DDR latency; 5.9 k with the pipelined vector unit |
| P3 (`LOOP` ×64, 4 tiny ops per body) | 628 cycles / descriptor | 141 cycles / descriptor: each `LOOP` iteration still costs ≈ 500 cycles of firmware |

DMA bandwidth (P6): LOAD and STORE each 12.8 B/cycle, the full beat rate the DDR model's 20 % random stalls allow.

## Matrix unit per-tile timing

`make sim_matrix_timing` runs `tb_matrix_timing` (scenarios S1 one GEMM, S2 two back-to-back GEMMs, S3 short k, S4 six one-tile GEMMs as decode issues them) with ideal surroundings (vmem grants every request, data one cycle later; VECTOR always ready), so every lost cycle belongs to the matrix unit or its systolic array. It explains why `tput` on the KV260 reaches 80 % of peak while the matrix unit is busy.

**Cycles:**

| Where | Per M-tile, k = 512 | Per M-tile, k = 64 | Owner |
|---|---|---|---|
| Useful reads (one k-step per cycle) | 512 | 64 | — |
| Read bubbles: the array idles 1 cycle in `S_WAIT_TILE` before every 16-deep k-tile | 31 | 3 | `systolic_array.sv` |
| Drain: `DRAIN_LEN` = (16 − 1) + (16 − 1) + 4 | 34 | 34 | `systolic_array.sv` |
| Writeback of 16 result rows | 16 | 16 | `systolic_array.sv` |
| Lead (start → first read), finish, handoff to the next tile | 5 | 5 | adapter + array |
| **Tile total** | **597 (85.8 % useful)** | **121 (52.9 %)** | |
| Per GEMM: first-tile fill (staging must hold all k words before the core starts) — **hidden for back-to-back GEMMs by cross-GEMM prefill** | 546 | 84 | adapter |
| Per GEMM: tail (stream the last tile's 16 rows) | 17 | 17 | adapter |

With cross-GEMM prefill (INTERFACE_SPEC §10.5) the matrix unit accepts the next `GEMM` as soon as its filler is done with the current one and fills the next command's first tiles while the current GEMM's last tiles compute, so only the first GEMM of a run pays the fill. S2 (two back-to-back GEMMs): 5,354 cycles instead of 5,902; what remains at the switch is 17 cycles (the next GEMM's first tile starts after the previous GEMM's last tile has streamed).

**This accounts for `tput`** (4 GEMMs × 16 M-tiles, k = 512): before prefill, 64 × 597 + 4 × (546 + 17) + handoffs ≈ 40.5 k of the 40,932 busy cycles measured on the board, overhead split roughly read bubbles 24 %, drain 27 %, writeback 13 %, per-GEMM fill 27 %, the rest 9 %. With prefill only the first fill remains: simulated MATRIX busy 39,297 cycles (213.5 MAC/cycle, 83 % of peak).

**Decode** (one M-tile per GEMM). Measured on the whole system by `sw/compiler/tests/rtl_gemm_timing.py`: the steady-state interval between GEMM completions, DDR model without random stalls (2026-11-09).

| k = 512, one M-tile | Before | After | |
|---|---|---|---|
| `LOAD_WGT` before every GEMM, as decode does | ~940 | **626** | |
| Weights already loaded | ~780 | **598** | One tile's compute: the core never waits |
| k = 256, `LOAD_WGT` before every GEMM | ~550 | **342** | |

What changed in `matrix_unit.sv`:

- **The core no longer waits for the hand-over to VECTOR.** A GEMM completes only when VECTOR has taken its last row, and the epilogue takes one row at a time (about 11 to 16 cycles: take, a 6-cycle calculation, the write). The core used to start the next GEMM only then. Now it starts the next GEMM's tiles as soon as it has finished the current GEMM's, while the streamer is still handing over rows.
- **Two pending GEMMs instead of one.** With one, the next GEMM was accepted only when the GEMM two back had completed, so its fill (k + 34 cycles) ran late and the core waited about 146 cycles every other GEMM. With two, the filler, the core and the streamer each work on a different GEMM.
- **Three weight halves.** With two, a `LOAD_WGT` waited for the GEMM before the current one and the next GEMM waited for that `LOAD_WGT`, so weight loads and fills serialized (about 160 cycles per GEMM).

Within a GEMM of several M-tiles the hand-over to VECTOR overlaps the next tile, so prefill paid it once per GEMM and gains little: P8 (`tput`) went from 213.5 to 215.7 MAC/cycle while MATRIX is busy.

What can be changed:

| Loss | Share of `tput` overhead | Fix | Cost |
|---|---|---|---|
| Per-GEMM first-tile fill | ~27 % | **Done:** cross-GEMM prefill in `matrix_unit.sv` (next `GEMM` accepted early, its first tiles filled during the current GEMM's last tiles) | Adapter logic only |
| Drain + writeback (50 / tile) | ~40 % | (a) two arrays alternating M-tiles, one draining while the other streams; (b) change `systolic_array.sv` to overlap the drain with the next tile | (a) doubles matrix area (+256 DSP, ~6.5 k LUT); (b) RTL work in the array |
| Read bubble every 16 k | ~24 % | Remove the idle cycle in `systolic_array.sv`, or a larger `K_TILE`: 32 halves it | GEMM `k` must then be a multiple of `K_TILE` |

