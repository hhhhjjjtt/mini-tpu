/* mini-tpu firmware memory map (INTERFACE_SPEC §8).
 * I-mem holds .text + .rodata and is loaded by the ARM through the I-mem
 * window; D-mem holds .bss + stack. Initialized writable data is not
 * supported (nothing would load it into D-mem). */
OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
    IMEM (rx)  : ORIGIN = 0x00000000, LENGTH = 16K
    DMEM (rw)  : ORIGIN = 0x00010000, LENGTH = 8K
}

SECTIONS
{
    .text : {
        KEEP(*(.text.start))
        *(.text .text.*)
    } > IMEM

    .rodata : {
        *(.rodata .rodata.* .srodata .srodata.*)
    } > IMEM

    .data : {
        *(.data .data.* .sdata .sdata.*)
    } > DMEM
    ASSERT(SIZEOF(.data) == 0, "firmware must not use initialized writable data")

    .bss (NOLOAD) : {
        . = ALIGN(4);
        _bss_start = .;
        *(.bss .bss.* .sbss .sbss.* COMMON)
        . = ALIGN(4);
        _bss_end = .;
    } > DMEM

    /* keep 2 KB for the stack */
    ASSERT(_bss_end <= 0x00011800, "D-mem overflow: less than 2 KB left for the stack")

    /DISCARD/ : { *(.comment) *(.riscv.attributes) *(.note*) *(.eh_frame*) }
}
