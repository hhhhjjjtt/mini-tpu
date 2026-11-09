# mini-tpu controller firmware

Program sequencer for the PicoRV32 in the scalar unit ([../rtl/scalar_unit/scalar_unit.sv](../rtl/scalar_unit/scalar_unit.sv)). Implements the program semantics of [INTERFACE_SPEC.md](../../docs/INTERFACE_SPEC.md) §3–§6: header validation, chunk fetch with prefetch, `LOOP` / `NEXT_CHUNK` / `BREAK` / `FENCE` / `HALT`, and error reporting (§11). Runs of unit descriptors are handed to the hardware issue engine (§8.7), which builds and pushes the commands; firmware keeps the engine's `LOOP_ADD` offsets equal to the active loop strides.

| File | Contents |
|---|---|
| `start.S` | Reset entry: stack, zero `.bss`, `main` |
| `link.ld` | I-mem 16 KB at 0 (`.text`, `.rodata`); D-mem 8 KB at 0x1_0000 (`.bss`, stack). Initialized `.data` is rejected at link time |
| `mini_tpu_hw.h` | RISC-V side register map (§8). Program format, opcodes, STATUS values and error codes come from the shared headers in [`../interface/`](../interface/), which the ARM runtime uses too |
| `main.c` | Idle loop and program interpreter |

```bash
make          # build/fw.elf, build/fw.lst, build/fw.hex (4096 words = I-mem image),
              # build/fw_image.h (the used words as a C array; the ARM runtime compiles it in)
make size
```

Toolchain: `riscv-none-elf-gcc` (xPack 15.2), `-march=rv32i -mabi=ilp32 -O2`. Current size: about 3.4 KB code, 360 B `.bss`.

The ARM runtime (`mtpu_boot` in [`../../sw/runtime/libmtpu/`](../../sw/runtime/libmtpu/)) writes the image word by word into the I-mem window (`0x8000`) while `CORE_CTRL.CORE_RESET = 1`, then releases reset and polls `FW_VERSION` (`0x0001_0000`). Boot takes about 2 k cycles, mostly zeroing `.bss`.

