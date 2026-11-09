// mini-tpu controller — RISC-V side register map (INTERFACE_SPEC.md §8).
// Program format, opcodes, STATUS values and error codes come from the shared
// interface headers (../interface/), which the ARM runtime uses as well.
#ifndef MINI_TPU_HW_H
#define MINI_TPU_HW_H

#include <stdint.h>

#include "mini_tpu_errors.h"
#include "mini_tpu_isa.h"
#include "mini_tpu_regs.h"

#define REG32(a) (*(volatile uint32_t *)(a))

// ---- Memories ----
#define DESC_BANK_BASE(b)   (0x00020000u + ((b) << 14))     // 16 KB each

// ---- Unit ports (§8.1) ----
#define UNIT_BASE(u)        (0x10000000u + ((u) << 8))
#define UNIT_CMD(u, w)      REG32(UNIT_BASE(u) + 4u * (w))
#define UNIT_PUSH(u)        REG32(UNIT_BASE(u) + 0x40)
#define UNIT_FREE(u)        REG32(UNIT_BASE(u) + 0x44)
#define UNIT_DONE(u)        REG32(UNIT_BASE(u) + 0x48)
#define UNIT_STATUS(u)      REG32(UNIT_BASE(u) + 0x4C)

// ---- Fetch DMA (§8.2) ----
#define FETCH_BASE          0x10002000u
#define FETCH_SRC_LO        REG32(FETCH_BASE + 0x00)
#define FETCH_SRC_HI        REG32(FETCH_BASE + 0x04)
#define FETCH_LEN           REG32(FETCH_BASE + 0x08)
#define FETCH_BANK          REG32(FETCH_BASE + 0x0C)
#define FETCH_START         REG32(FETCH_BASE + 0x10)
#define FETCH_STATUS        REG32(FETCH_BASE + 0x14)
#define FETCH_BUSY          (1u << 0)
#define FETCH_DONE          (1u << 1)
#define FETCH_AXI_ERR       (1u << 2)

// ---- Mailbox (§8.3) — same offsets as the ARM map (§7) ----
#define MB_BASE             0x10003000u
#define MB_HW_ARCH          REG32(MB_BASE + MINI_TPU_R_HW_ARCH)
#define MB_DOORBELL         REG32(MB_BASE + MINI_TPU_R_DOORBELL)  // read clears
#define MB_STATUS           REG32(MB_BASE + MINI_TPU_R_STATUS)
#define MB_PROG_BASE_LO     REG32(MB_BASE + MINI_TPU_R_PROG_BASE_LO)
#define MB_PROG_BASE_HI     REG32(MB_BASE + MINI_TPU_R_PROG_BASE_HI)
#define MB_REGION_LO(r)     REG32(MB_BASE + MINI_TPU_R_REGION_LO(r))
#define MB_REGION_HI(r)     REG32(MB_BASE + MINI_TPU_R_REGION_HI(r))
#define MB_FW_VERSION       REG32(MB_BASE + MINI_TPU_R_FW_VERSION)
#define MB_FW_ARCH          REG32(MB_BASE + MINI_TPU_R_FW_ARCH)
#define MB_FW_HEARTBEAT     REG32(MB_BASE + MINI_TPU_R_FW_HEARTBEAT)
#define MB_ERR_DESC         REG32(MB_BASE + MINI_TPU_R_ERR_DESC)
#define MB_ERR_UNIT         REG32(MB_BASE + MINI_TPU_R_ERR_UNIT)
#define MB_ERR_INFO0        REG32(MB_BASE + MINI_TPU_R_ERR_INFO0)
#define MB_ERR_INFO1        REG32(MB_BASE + MINI_TPU_R_ERR_INFO1)
#define MB_DESC_COUNT       REG32(MB_BASE + MINI_TPU_R_DESC_COUNT)
#define MB_UNIT_CTRL        REG32(MB_BASE + 0x400)          // bit 0: clear DONE counters
#define MB_UNIT_SUMMARY     REG32(MB_BASE + 0x404)

#define SUM_ERR_MASK        0x0000000Fu                     // [3:0]  unit error
#define SUM_WATCHDOG        (1u << 12)
#define SUM_DOORBELL        (1u << 16)

// ---- Timer / trace (§8.4) ----
#define TM_BASE             0x10004000u
#define TM_CYCLE_LO         REG32(TM_BASE + 0x00)
#define TM_TRACE_W(i)       REG32(TM_BASE + 0x10 + 4u * (i))
#define TM_TRACE_PUSH       REG32(TM_BASE + 0x20)

// ---- Issue engine (§8.7) ----
#define ISE_BASE            0x10005000u
#define ISE_CTRL            REG32(ISE_BASE + 0x00)          // bit 0 GO, bit 1 STOP
#define ISE_POS             REG32(ISE_BASE + 0x04)          // {bank, index}
#define ISE_COUNT           REG32(ISE_BASE + 0x08)
#define ISE_STATUS          REG32(ISE_BASE + 0x0C)
#define ISE_ISSUED(u)       REG32(ISE_BASE + 0x10 + 4u * (u))
#define ISE_LOOP_ADD(r)     REG32(ISE_BASE + 0x40 + 4u * (r))
#define ISE_GO              (1u << 0)
#define ISE_STOP            (1u << 1)
#define ISE_ST_BUSY         (1u << 0)
#define ISE_REASON(s)       (((s) >> 1) & 7u)
#define ISE_OP(s)           (((s) >> 8) & 0xFFu)
#define ISE_N(s)            (((s) >> 16) & 0x1FFu)
#define ISE_R_DONE          0u
#define ISE_R_CTRL          1u
#define ISE_R_REGION        2u
#define ISE_R_OVF           3u
#define ISE_R_STOP          4u

#endif
