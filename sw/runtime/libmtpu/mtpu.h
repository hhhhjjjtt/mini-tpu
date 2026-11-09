// libmtpu — ARM-side access to mini-tpu on the KV260 (INTERFACE_SPEC.md §7.6).
//
//   mtpu_open    map the AXI-Lite window (/dev/mem) and the DMA buffer (/dev/udmabuf0)
//   mtpu_boot    load the controller firmware into I-mem and start it
//   mtpu_run     sync the DMA buffer, ring the doorbell, poll STATUS, sync back
//
// The DMA buffer is cached on the ARM side and the accelerator's masters use the
// non-coherent HPC0 port, so mtpu_run syncs it through u-dma-buf around every run.
// Programs and data are placed in dev->buf by the caller (see mtpu_desc.h).
#ifndef MTPU_H
#define MTPU_H

#include <stdbool.h>
#include <stdint.h>

#include "mini_tpu_errors.h"
#include "mini_tpu_isa.h"
#include "mini_tpu_regs.h"

#define MTPU_CTRL_PHYS_DEFAULT  0xA0000000u     // KV260 block design (hw/fpga/build_kv260.tcl)
#define MTPU_UDMABUF_DEV        "/dev/udmabuf0"
#define MTPU_UDMABUF_SYSFS      "/sys/class/u-dma-buf/udmabuf0"
#define MTPU_RUN_FAILED         0xFFFFFFFFu     // mtpu_run: not started, sync failed or timed out

typedef struct {
    volatile uint8_t *ctrl;                     // AXI-Lite window
    uint64_t ctrl_phys;
    uint8_t *buf;                               // DMA buffer, CPU view
    uint64_t buf_phys, buf_size;
} mtpu_dev_t;

typedef struct {
    uint64_t cycles;                            // RUN -> DONE
    uint32_t busy[MINI_TPU_NUM_UNITS], wait[MINI_TPU_NUM_UNITS], starved[MINI_TPU_NUM_UNITS];
    uint32_t desc_count, trace_count;
} mtpu_perf_t;

typedef struct {
    uint32_t status, code;                      // STATUS and its error code (§11)
    uint32_t chunk, desc, unit, info0, info1;
    bool has_trace;
    uint32_t trace[4];                          // last trace record (§8.5)
} mtpu_error_t;

static inline uint32_t mtpu_rd(const mtpu_dev_t *d, uint32_t off)
{
    return *(volatile uint32_t *)(d->ctrl + off);
}
static inline void mtpu_wr(const mtpu_dev_t *d, uint32_t off, uint32_t v)
{
    *(volatile uint32_t *)(d->ctrl + off) = v;
}

// Maps both windows. Fails if the DMA buffer is smaller than min_buf_bytes.
int mtpu_open(mtpu_dev_t *d, uint64_t ctrl_phys, uint64_t min_buf_bytes);
void mtpu_close(mtpu_dev_t *d);

// Firmware image: one 32-bit word per line (hw/fw/build/fw.hex). Returns the word count or -1.
int mtpu_read_hex(const char *path, uint32_t *words, int max_words);
// CORE_RESET, write and verify I-mem, release, wait for FW_VERSION, check FW_ARCH == HW_ARCH.
int mtpu_boot(mtpu_dev_t *d, const uint32_t *image, int words);

void mtpu_set_region(mtpu_dev_t *d, unsigned region, uint64_t addr);
// u-dma-buf cache maintenance for [off, off + size) of the buffer.
int mtpu_sync(mtpu_dev_t *d, bool for_device, uint32_t off, uint32_t size);

// Runs the program at buffer offset prog_off and returns STATUS, or MTPU_RUN_FAILED.
// [0, sync_bytes) of the buffer is synced for the device before and for the CPU after.
uint32_t mtpu_run(mtpu_dev_t *d, uint32_t prog_off, uint32_t sync_bytes, uint64_t timeout_us,
                  uint64_t *wall_us);
// mtpu_run without cache maintenance: the caller syncs what it wrote and what it will read.
uint32_t mtpu_exec(mtpu_dev_t *d, uint32_t prog_off, uint64_t timeout_us, uint64_t *wall_us);

void mtpu_read_perf(const mtpu_dev_t *d, mtpu_perf_t *p);
void mtpu_read_error(const mtpu_dev_t *d, uint32_t status, mtpu_error_t *e);
const char *mtpu_err_name(uint32_t code);

uint64_t mtpu_now_us(void);

#endif
