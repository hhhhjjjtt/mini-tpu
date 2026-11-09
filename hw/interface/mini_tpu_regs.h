// mini-tpu — ARM-side register map (INTERFACE_SPEC.md §7).
//
// Byte offsets into the 64 KB AXI-Lite window. The firmware's mailbox (§8.3)
// uses the same offsets from its own base address.
#ifndef MINI_TPU_REGS_H
#define MINI_TPU_REGS_H

#define MINI_TPU_CTRL_SIZE      0x10000u

// ---- Control and status (§7.1) ----
#define MINI_TPU_R_HW_ID        0x000u
#define MINI_TPU_R_HW_VERSION   0x004u
#define MINI_TPU_R_HW_ARCH      0x008u
#define MINI_TPU_R_CORE_CTRL    0x010u
#define MINI_TPU_R_DOORBELL     0x014u
#define MINI_TPU_R_STATUS       0x018u
#define MINI_TPU_R_IRQ_STATUS   0x01Cu
#define MINI_TPU_R_PROG_BASE_LO 0x020u
#define MINI_TPU_R_PROG_BASE_HI 0x024u
#define MINI_TPU_R_WATCHDOG     0x030u

#define MINI_TPU_HW_ID          0x5550544Du     // "MTPU"

// HW_ARCH / FW_ARCH / program-header arch word: [7:0] N, [15:8] units mask, [31:16] vmem KB
#define MINI_TPU_ARCH_WORD(n, units, vmem_kb) \
    ((unsigned)(n) | ((unsigned)(units) << 8) | ((unsigned)(vmem_kb) << 16))
#define MINI_TPU_ARCH_N(a)      ((a) & 0xFFu)
#define MINI_TPU_ARCH_UNITS(a)  (((a) >> 8) & 0xFFu)
#define MINI_TPU_ARCH_VMEM_KB(a)    ((a) >> 16)

#define MINI_TPU_CORE_RESET     (1u << 0)       // CORE_CTRL: 1 = core held in reset (reset value)
#define MINI_TPU_CORE_IRQ_EN    (1u << 1)

#define MINI_TPU_DB_RUN         1u              // DOORBELL values
#define MINI_TPU_DB_RESUME      2u
#define MINI_TPU_DB_ABORT       3u

#define MINI_TPU_ST_IDLE        0u              // STATUS[3:0]
#define MINI_TPU_ST_RUNNING     1u
#define MINI_TPU_ST_BREAK       2u
#define MINI_TPU_ST_DONE        3u
#define MINI_TPU_ST_ERROR       4u
#define MINI_TPU_ST_STATE(s)    ((s) & 0xFu)
#define MINI_TPU_ST_ERR_CODE(s) (((s) >> 8) & 0xFFu)
#define MINI_TPU_ST_CODE(s)     ((s) >> 16)     // break_id or error detail

#define MINI_TPU_IRQ_DONE       (1u << 0)       // IRQ_STATUS, write 1 to clear
#define MINI_TPU_IRQ_BREAK      (1u << 1)
#define MINI_TPU_IRQ_ERROR      (1u << 2)

// ---- Regions (§7.2) ----
#define MINI_TPU_R_REGION_LO(r) (0x100u + 8u * (r))
#define MINI_TPU_R_REGION_HI(r) (0x104u + 8u * (r))

// ---- Firmware and error info (§7.3) ----
#define MINI_TPU_R_FW_VERSION   0x200u
#define MINI_TPU_R_FW_ARCH      0x204u
#define MINI_TPU_R_FW_HEARTBEAT 0x208u
#define MINI_TPU_R_ERR_DESC     0x210u          // [15:0] descriptor in chunk, [31:16] chunk
#define MINI_TPU_R_ERR_UNIT     0x214u
#define MINI_TPU_R_ERR_INFO0    0x218u
#define MINI_TPU_R_ERR_INFO1    0x21Cu

#define MINI_TPU_FW_VERSION     0x00010000u     // firmware interface version 1.0 ([31:16] major)

// ---- Performance counters (§7.4), reset at RUN ----
#define MINI_TPU_R_CYCLES_LO    0x300u
#define MINI_TPU_R_CYCLES_HI    0x304u
#define MINI_TPU_R_UNIT_BUSY(u) (0x310u + 4u * (u))
#define MINI_TPU_R_UNIT_WAIT(u) (0x320u + 4u * (u))
#define MINI_TPU_R_UNIT_STARVED(u)  (0x330u + 4u * (u))
#define MINI_TPU_R_DESC_COUNT   0x340u
#define MINI_TPU_R_TRACE_COUNT  0x344u

// ---- Memory windows (§7.5) ----
#define MINI_TPU_W_TRACE        0x2000u         // 256 records x 16 B (§8.5)
#define MINI_TPU_TRACE_RECORDS  256u
#define MINI_TPU_W_DMEM         0x4000u         // 8 KB, debug
#define MINI_TPU_W_IMEM         0x8000u         // 16 KB, writable only in CORE_RESET
#define MINI_TPU_IMEM_WORDS     4096u

#endif
