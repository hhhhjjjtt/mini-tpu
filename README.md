# mini-tpu

A small int8 TPU for the AMD Kria KV260, and a Python compiler that runs GPT-2 small on it. Type the start of a line on the board; the TPU finishes it.

The hardware is a 16 × 16 systolic array, a 16-lane vector unit, 2 MB of on-chip memory and a RISC-V controller that runs a program of 64-byte descriptors from DDR. The compiler quantizes GPT-2 to integers, tiles it for the hardware and emits the programs and DDR images; the board's ARM cores run the token loop. Every result is checked bit for bit against an integer reference model, in a Python simulator, on the RTL and on the board.

## Results on the KV260 (100 MHz)

| | Result |
|---|---|
| Prompt of 128 tokens (prefill) | 1.04 s, bit-exact against the simulator (6.4 M logits) |
| New tokens (decode, KV cache in DDR) | 5.2 tokens/s up to 128 positions, 3.8 at 1024; bit-exact at every context size |
| Quality (int8 vs float GPT-2, WikiText-2) | Perplexity 63.3 vs 53.3 at 128 tokens; top-1 agreement 72 % |
| FPGA | Timing met (WNS +0.95 ns); 66 k LUT, 524 DSP, 64 URAM; 84 % of CLBs |

Details and what limits each number: [sw/runtime/README.md](sw/runtime/README.md), [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md).

## Layout

| Path | Contents |
|---|---|
| [`config.toml`](config.toml) | The one configuration: hardware parameters, model, build |
| [`hw/`](hw/rtl/README.md) | RTL ([`rtl/`](hw/rtl/README.md)), testbenches ([`sim/`](hw/sim/README.md)), RISC-V firmware ([`fw/`](hw/fw/README.md)), Vivado flow ([`fpga/`](hw/fpga/README.md)), headers shared by firmware, runtime and compiler ([`interface/`](hw/interface/)) |
| [`sw/compiler/`](sw/compiler/README.md) | Calibration, quantization, compiler, bit-exact simulator, race checker, tests |
| [`sw/runtime/`](sw/runtime/README.md) | Runs on the KV260: device library, bring-up test, `run_prefill`, `generate`, text frontend |
| [`docs/`](docs/) | [Build](docs/BUILD.md), [interface spec](docs/INTERFACE_SPEC.md) (the ISA), [architecture](docs/ARCHITECTURE.md), [quantization](docs/QUANTIZATION.md), [next steps](docs/NEXT_STEPS.md) |
| [`quantization_experiment/`](quantization_experiment/README.md) | The study that chose the number formats (PyTorch) |
| [`tools/`](tools/) | `gen_config.py`: `config.toml` → RTL and C headers |

## Setup (workstation)

Python 3.11 or newer, GNU make (MSYS2 on Windows), Icarus Verilog for RTL simulation, `riscv-none-elf-gcc` for the firmware, and Vivado 2024.1 for the bitstream.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt      # .venv/bin/python on Linux
make model                                                   # once: GPT-2 and calibration text, about 0.6 GB
```

`make` uses `.venv` when it exists. The quantization study also needs PyTorch: see its README.

## Build and run

```bash
make              # calibrate (once) and compile GPT-2 into sw/compiler/output/
make check        # run the output in the simulator against its golden data
make test         # fast checks, about 15 s
make sim          # RTL regression
make bitstream    # firmware, then prints the Vivado command
make dist         # one folder for the board: sw/runtime/build/dist/mini-tpu/
```

On the KV260 (Ubuntu, with the u-dma-buf module), after copying that folder:

```bash
make load         # bitstream and a 256 MB DMA buffer, once per boot
make              # build the runtime
make run          # bring-up test
make prefill      # GPT-2 prefill against the simulator
make decode       # decode at every context size against the reference
make chat         # type a line; GPT-2 finishes it
```

See [docs/BUILD.md](docs/BUILD.md) for the build stages and outputs, and [sw/runtime/README.md](sw/runtime/README.md) for the board.

## License

[Apache License 2.0](LICENSE).
