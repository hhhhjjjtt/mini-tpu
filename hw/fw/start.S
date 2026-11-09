// mini-tpu firmware entry. Reset vector 0x0000_0000 (INTERFACE_SPEC §8).
// Core IRQs are masked at reset and the firmware polls, so the IRQ vector
// at 0x10 is never taken.

    .section .text.start, "ax"
    .globl _start
_start:
    li   sp, 0x00012000             // top of D-mem

    // zero .bss (link.ld forbids initialized .data)
    la   a1, _bss_start
    la   a2, _bss_end
1:  bgeu a1, a2, 2f
    sw   zero, 0(a1)
    addi a1, a1, 4
    j    1b

2:  call main
3:  j    3b
