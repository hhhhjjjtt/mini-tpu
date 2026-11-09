// libmtpu — device access (see mtpu.h).

#define _GNU_SOURCE
#include "mtpu.h"

#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

uint64_t mtpu_now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000u + (uint64_t)ts.tv_nsec / 1000u;
}

const char *mtpu_err_name(uint32_t code)
{
    switch (code) {
    case MINI_TPU_OK:                   return "OK";
    case MINI_TPU_ERR_BAD_MAGIC:        return "ERR_BAD_MAGIC";
    case MINI_TPU_ERR_VERSION:          return "ERR_VERSION";
    case MINI_TPU_ERR_ARCH:             return "ERR_ARCH";
    case MINI_TPU_ERR_REGION_UNSET:     return "ERR_REGION_UNSET";
    case MINI_TPU_ERR_BAD_OPCODE:       return "ERR_BAD_OPCODE";
    case MINI_TPU_ERR_BAD_FIELD:        return "ERR_BAD_FIELD";
    case MINI_TPU_ERR_LOOP_DEPTH:       return "ERR_LOOP_DEPTH";
    case MINI_TPU_ERR_LOOP_BODY:        return "ERR_LOOP_BODY";
    case MINI_TPU_ERR_CHUNK:            return "ERR_CHUNK";
    case MINI_TPU_ERR_COUNTER_OVERFLOW: return "ERR_COUNTER_OVERFLOW";
    case MINI_TPU_ERR_FETCH_AXI:        return "ERR_FETCH_AXI";
    case MINI_TPU_ERR_UNIT_AXI:         return "ERR_UNIT_AXI";
    case MINI_TPU_ERR_VMEM_RANGE:       return "ERR_VMEM_RANGE";
    case MINI_TPU_ERR_STREAM_MISMATCH:  return "ERR_STREAM_MISMATCH";
    case MINI_TPU_ERR_WATCHDOG:         return "ERR_WATCHDOG";
    case MINI_TPU_ERR_ABORTED:          return "ERR_ABORTED";
    case MINI_TPU_ERR_CORE_TRAP:        return "ERR_CORE_TRAP";
    default:                        return "?";
    }
}

// -----------------------------------------------------------------------------
// Mapping
// -----------------------------------------------------------------------------
static int read_sys(const char *name, bool hex, uint64_t *out)
{
    char path[256], buf[128];
    snprintf(path, sizeof path, "%s/%s", MTPU_UDMABUF_SYSFS, name);
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    ssize_t n = read(fd, buf, sizeof buf - 1);
    close(fd);
    if (n <= 0) return -1;
    buf[n] = 0;
    unsigned long long v;
    if (sscanf(buf, hex ? "%llx" : "%llu", &v) != 1) return -1;
    *out = v;
    return 0;
}

int mtpu_open(mtpu_dev_t *d, uint64_t ctrl_phys, uint64_t min_buf_bytes)
{
    memset(d, 0, sizeof *d);
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open(/dev/mem) (run with sudo)"); return -1; }
    void *p = mmap(NULL, MINI_TPU_CTRL_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)ctrl_phys);
    close(fd);
    if (p == MAP_FAILED) { perror("mmap(control window)"); return -1; }
    d->ctrl = (volatile uint8_t *)p;
    d->ctrl_phys = ctrl_phys;

    if (read_sys("phys_addr", true, &d->buf_phys) || read_sys("size", false, &d->buf_size)) {
        fprintf(stderr, "cannot read %s: is u-dma-buf loaded? (make load)\n", MTPU_UDMABUF_SYSFS);
        return -1;
    }
    if (d->buf_size < min_buf_bytes) {
        fprintf(stderr, "udmabuf0 is %" PRIu64 " bytes, need %" PRIu64 " (reload with UDMABUF_SIZE=%" PRIu64 ")\n",
                d->buf_size, min_buf_bytes, min_buf_bytes);
        return -1;
    }
    if (d->buf_phys == 0 || (d->buf_phys & 0xF) || d->buf_phys + d->buf_size > (1ull << 40)) {
        fprintf(stderr, "udmabuf0 phys 0x%" PRIx64 " unusable (must be nonzero, 16 B aligned, < 2^40)\n",
                d->buf_phys);
        return -1;
    }
    fd = open(MTPU_UDMABUF_DEV, O_RDWR);
    if (fd < 0) { perror("open(" MTPU_UDMABUF_DEV ")"); return -1; }
    p = mmap(NULL, (size_t)d->buf_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) { perror("mmap(" MTPU_UDMABUF_DEV ")"); return -1; }
    d->buf = (uint8_t *)p;
    return 0;
}

void mtpu_close(mtpu_dev_t *d)
{
    if (d->buf) munmap(d->buf, (size_t)d->buf_size);
    if (d->ctrl) munmap((void *)d->ctrl, MINI_TPU_CTRL_SIZE);
    memset(d, 0, sizeof *d);
}

// dir: 0 bidirectional (sysfs encoding: offset in the upper word, size|dir|1 below)
int mtpu_sync(mtpu_dev_t *d, bool for_device, uint32_t off, uint32_t size)
{
    (void)d;
    char path[256], msg[64];
    snprintf(path, sizeof path, "%s/%s", MTPU_UDMABUF_SYSFS, for_device ? "sync_for_device" : "sync_for_cpu");
    uint32_t lo = ((size + 15u) & ~15u) | 0x1u;
    int len = snprintf(msg, sizeof msg, "0x%08X%08X", off, lo);
    int fd = open(path, O_WRONLY);
    if (fd < 0) { perror(path); return -1; }
    ssize_t w = write(fd, msg, (size_t)len);
    close(fd);
    if (w != len) { perror(path); return -1; }
    return 0;
}

// -----------------------------------------------------------------------------
// Firmware
// -----------------------------------------------------------------------------
int mtpu_read_hex(const char *path, uint32_t *words, int max_words)
{
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return -1; }
    char line[64];
    int n = 0;
    while (fgets(line, sizeof line, f)) {
        unsigned v;
        if (sscanf(line, "%x", &v) != 1) continue;
        if (n == max_words) { fprintf(stderr, "%s: more than %d words\n", path, max_words); fclose(f); return -1; }
        words[n++] = v;
    }
    fclose(f);
    if (n == 0) { fprintf(stderr, "%s: no words\n", path); return -1; }
    return n;
}

int mtpu_boot(mtpu_dev_t *d, const uint32_t *image, int words)
{
    if (words < 1 || words > (int)MINI_TPU_IMEM_WORDS) {
        fprintf(stderr, "firmware image of %d words does not fit I-mem (%u)\n", words, MINI_TPU_IMEM_WORDS);
        return -1;
    }
    mtpu_wr(d, MINI_TPU_R_CORE_CTRL, MINI_TPU_CORE_RESET);
    for (int i = 0; i < (int)MINI_TPU_IMEM_WORDS; i++)
        mtpu_wr(d, MINI_TPU_W_IMEM + 4 * i, i < words ? image[i] : 0);
    for (int i = 0; i < words; i++) {
        uint32_t v = mtpu_rd(d, MINI_TPU_W_IMEM + 4 * i);
        if (v != image[i]) {
            fprintf(stderr, "I-mem readback mismatch at word %d: 0x%08x vs 0x%08x\n", i, v, image[i]);
            return -1;
        }
    }
    mtpu_wr(d, MINI_TPU_R_CORE_CTRL, 0);
    uint64_t t0 = mtpu_now_us();
    while (mtpu_rd(d, MINI_TPU_R_FW_VERSION) != MINI_TPU_FW_VERSION) {
        if (mtpu_now_us() - t0 > 100000) {
            fprintf(stderr, "firmware did not boot (FW_VERSION 0x%08x)\n", mtpu_rd(d, MINI_TPU_R_FW_VERSION));
            return -1;
        }
    }
    if (mtpu_rd(d, MINI_TPU_R_FW_ARCH) != mtpu_rd(d, MINI_TPU_R_HW_ARCH)) {
        fprintf(stderr, "FW_ARCH 0x%08x != HW_ARCH 0x%08x\n", mtpu_rd(d, MINI_TPU_R_FW_ARCH),
                mtpu_rd(d, MINI_TPU_R_HW_ARCH));
        return -1;
    }
    return 0;
}

// -----------------------------------------------------------------------------
// Running programs
// -----------------------------------------------------------------------------
void mtpu_set_region(mtpu_dev_t *d, unsigned region, uint64_t addr)
{
    mtpu_wr(d, MINI_TPU_R_REGION_LO(region), (uint32_t)addr);
    mtpu_wr(d, MINI_TPU_R_REGION_HI(region), (uint32_t)(addr >> 32));
}

uint32_t mtpu_run(mtpu_dev_t *d, uint32_t prog_off, uint32_t sync_bytes, uint64_t timeout_us,
                  uint64_t *wall_us)
{
    if (mtpu_sync(d, true, 0, sync_bytes)) return MTPU_RUN_FAILED;
    uint32_t st = mtpu_exec(d, prog_off, timeout_us, wall_us);
    if (st != MTPU_RUN_FAILED) mtpu_sync(d, false, 0, sync_bytes);
    return st;
}

uint32_t mtpu_exec(mtpu_dev_t *d, uint32_t prog_off, uint64_t timeout_us, uint64_t *wall_us)
{
    uint32_t st = MINI_TPU_ST_STATE(mtpu_rd(d, MINI_TPU_R_STATUS));
    if (st == MINI_TPU_ST_RUNNING || st == MINI_TPU_ST_BREAK) {
        fprintf(stderr, "    controller busy (STATUS 0x%08x); not starting\n", mtpu_rd(d, MINI_TPU_R_STATUS));
        return MTPU_RUN_FAILED;
    }
    uint64_t pa = d->buf_phys + prog_off;
    mtpu_wr(d, MINI_TPU_R_PROG_BASE_LO, (uint32_t)pa);
    mtpu_wr(d, MINI_TPU_R_PROG_BASE_HI, (uint32_t)(pa >> 32));
    mtpu_wr(d, MINI_TPU_R_IRQ_STATUS, MINI_TPU_IRQ_DONE | MINI_TPU_IRQ_BREAK | MINI_TPU_IRQ_ERROR);
    uint64_t t0 = mtpu_now_us();
    mtpu_wr(d, MINI_TPU_R_DOORBELL, MINI_TPU_DB_RUN);
    while (MINI_TPU_ST_STATE(st = mtpu_rd(d, MINI_TPU_R_STATUS)) == MINI_TPU_ST_RUNNING) {
        if (mtpu_now_us() - t0 > timeout_us) {
            fprintf(stderr, "    timeout: STATUS still RUNNING after %" PRIu64 " ms (heartbeat %u, desc %u)\n",
                    timeout_us / 1000, mtpu_rd(d, MINI_TPU_R_FW_HEARTBEAT), mtpu_rd(d, MINI_TPU_R_DESC_COUNT));
            return MTPU_RUN_FAILED;
        }
    }
    if (wall_us) *wall_us = mtpu_now_us() - t0;
    return st;
}

void mtpu_read_perf(const mtpu_dev_t *d, mtpu_perf_t *p)
{
    p->cycles = (uint64_t)mtpu_rd(d, MINI_TPU_R_CYCLES_HI) << 32 | mtpu_rd(d, MINI_TPU_R_CYCLES_LO);
    for (unsigned u = 0; u < MINI_TPU_NUM_UNITS; u++) {
        p->busy[u] = mtpu_rd(d, MINI_TPU_R_UNIT_BUSY(u));
        p->wait[u] = mtpu_rd(d, MINI_TPU_R_UNIT_WAIT(u));
        p->starved[u] = mtpu_rd(d, MINI_TPU_R_UNIT_STARVED(u));
    }
    p->desc_count = mtpu_rd(d, MINI_TPU_R_DESC_COUNT);
    p->trace_count = mtpu_rd(d, MINI_TPU_R_TRACE_COUNT);
}

void mtpu_read_error(const mtpu_dev_t *d, uint32_t status, mtpu_error_t *e)
{
    uint32_t ed = mtpu_rd(d, MINI_TPU_R_ERR_DESC);
    e->status = status;
    e->code = MINI_TPU_ST_ERR_CODE(status);
    e->chunk = ed >> 16;
    e->desc = ed & 0xFFFF;
    e->unit = mtpu_rd(d, MINI_TPU_R_ERR_UNIT);
    e->info0 = mtpu_rd(d, MINI_TPU_R_ERR_INFO0);
    e->info1 = mtpu_rd(d, MINI_TPU_R_ERR_INFO1);
    uint32_t n = mtpu_rd(d, MINI_TPU_R_TRACE_COUNT);
    e->has_trace = n != 0;
    if (n) {
        uint32_t base = MINI_TPU_W_TRACE + 16 * ((n - 1) % MINI_TPU_TRACE_RECORDS);
        for (int w = 0; w < 4; w++) e->trace[w] = mtpu_rd(d, base + 4 * (uint32_t)w);
    }
}
