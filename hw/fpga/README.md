# mini-tpu FPGA

| Path | Contents |
|---|---|
| `top/mini_tpu_top.v` | Structural top level (Verilog-2001, so it can be a block-design module reference): AXI-Lite slave, fetch / LOAD AXI4 read masters, STORE AXI4 write master, IRQ |
| `constr/mini_tpu_ooc.xdc` | Out-of-context constraints: 10 ns clock, 2 ns I/O budget |
| `syn/run_vivado.sh`, `syn/syn_ooc.tcl` | Out-of-context synthesis / place + route |
| `build_kv260.tcl` | **KV260 build**: Vivado project, block design (Zynq PS + `mini_tpu_top`), bitstream, XSA |

RTL sources come from [`../rtl/files.mk`](../rtl/files.mk); `tb_system` in [`../sim/`](../sim/) simulates `mini_tpu_top`.

## KV260 bitstream

Either way, it builds everything in `build/kv260/`:

- **Vivado GUI:** Tools → Run Tcl Script… → `hw/fpga/build_kv260.tcl`. The project stays open afterwards (Open Block Design to inspect it).
- **Terminal** (VS Code, from `hw/`):

  ```powershell
  # PowerShell
  & D:\Xilinx\Vivado\2024.1\bin\vivado.bat -mode batch -nojournal -log fpga/build/kv260.log -source fpga/build_kv260.tcl
  ```
  ```bash
  # Git Bash
  /d/Xilinx/Vivado/2024.1/bin/vivado.bat -mode batch -nojournal -log fpga/build/kv260.log -source fpga/build_kv260.tcl
  ```

  Optional `-tclargs <jobs>`. In the GUI, set `mini_tpu_jobs` in the Tcl console before running.

Build the firmware first (`make -C fw`) so `fw.hex` is copied next to the bitstream; the bitstream does not contain it — the runtime loads it into I-mem.

| Output (`build/kv260/`) | Use |
|---|---|
| `out/mini_tpu.bit` | Bitstream. Same loading flow as v1: copy to `/lib/firmware/` and write its name to `/sys/class/fpga_manager/fpga0/firmware` |
| `out/mini_tpu.bit.bin` | bootgen-processed bitstream, for flows that want the `.bin` |
| `out/mini_tpu.xsa` | Hardware platform (PetaLinux / Vitis / device tree generation) |
| `out/fw.hex` | Scalar-unit firmware image |
| `out/address_map.txt`, `out/timing_summary.rpt`, `out/utilization.rpt`, `out/build_info.txt` | Address map, reports, build settings and timing |
| `project/mini_tpu_kv260.xpr` | The Vivado project |

Block design: the PS (KV260 board preset, `kv260_som:part0:1.4`) `M_AXI_HPM0_FPD` reaches `mini_tpu_top`'s AXI-Lite window at **`0xA000_0000` (64 KB)** through a SmartConnect; the fetch / LOAD / STORE masters reach DDR through a second SmartConnect into `S_AXI_HPC0_FPD` (non-coherent with AXI cache `0000`, like the v1 DMA flow with udmabuf; DDR low 2 GB at 0, high at `0x8_0000_0000`). `pl_clk0` (≈ 100 MHz) clocks everything, `proc_sys_reset` derives the reset from `pl_resetn0`, and `irq` drives `pl_ps_irq0[0]`.

## Out-of-context run

```bash
# from hw/
fpga/syn/run_vivado.sh synth            # synthesis only
fpga/syn/run_vivado.sh impl             # + place + route
```

Runs `mini_tpu_top` alone on `xck26-sfvc784-2LV-c` at 100 MHz (no PS, no block design). Vivado runs inside `build/<stage>/`, so reports (`util_*.rpt`, `timing_*.rpt`), checkpoints (`post_*.dcp`) and Vivado's scratch files stay there. Uses `vivado` from `PATH`, else `D:/Xilinx/Vivado/2024.1`; override with `VIVADO=...`. It is a script, not a make target: msys make here gives child processes a stripped environment and `vivado.bat` cannot start.

Result (2026-11-06, `systolic_streaming`, with `CAUSAL`, `VEC_LAYERNORM`, `VEC_LUT`): **timing met, WNS +0.35 ns, WHS +0.04 ns.** LUT 61.0 k (52.1 %), FF 45.8 k (19.6 %), DSP 524 (42.0 %), BRAM 45.5 tiles (31.6 %), URAM 64 (100 %), CLB 84.5 %.

Full KV260 build (2026-11-06, parameterized RTL at the default config): WNS +0.95 ns, WHS +0.01 ns, LUT 66.2 k (56.5 %), FF 53.1 k (22.7 %), DSP 524 (42.0 %), BRAM 45.5 (31.6 %), URAM 64 (100 %), CLB 84.3 %. Before parameterizing: WNS +0.55 ns, LUT 65.9 k, CLB 83.6 %.

Earlier result (2026-11-01, `systolic_streaming`, complete RTL incl. `VEC_SOFTMAX`): **timing met, WNS +0.64 ns, WHS +0.03 ns.** LUT 55.6 k (47.5 %), FF 43.3 k (18.5 %), DSP 524 (42.0 %), BRAM 37.5 (26.0 %), URAM 64 (100 %), CLB slices 73 %. (Before softmax: WNS +0.55 ns, LUT 49.2 k.) Breakdown in [ARCHITECTURE.md §8](../../docs/ARCHITECTURE.md). The first softmax route missed by 0.1 ns on a 16-way block max written as a chain; a balanced tree fixed it.

Fixes made to get there:

| Problem in the first synthesis | Fix |
|---|---|
| Vector unit: whole requant / eltwise chain in one cycle (78 logic levels, WNS −12.6 ns) | Feed-forward pipeline in `vector_unit.sv`: 5 stages for the epilogue, 10 for eltwise; MUL reuses the `apply_scale` pipeline. Still bit-exact; P2 takes 5.9 k instead of 5.6 k cycles |
| Unit command FIFOs and DMA FIFOs built from flip-flops (storage written inside the async-reset block) | Storage moved to a reset-free `always` → LUTRAM |
| Controller descriptor buffer (2 K × 128) in LUTRAM | Single read address / enable → BRAM |

## On the board

The runtime, the bring-up test and the load target are in [`../../sw/runtime/`](../../sw/runtime/README.md). `make dist` in the repository root collects them with `out/mini_tpu.bit`, the firmware and the compiled model.
