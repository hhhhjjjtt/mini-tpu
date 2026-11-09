// Device layer of sw/runtime/libmtpu/mtpu.c with the board replaced by the Python
// simulator. sw/runtime/tests/mtpu_test.c, run_prefill.c and generate.c build against this
// unchanged: it fills the DMA buffer as on the KV260, and mtpu_run hands that buffer to replay.py.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "mtpu.h"

#define BUF_PHYS 0x40000000ull

static uint8_t ctrl[0x10000];
static uint64_t regions[MINI_TPU_NUM_REGIONS];

// the bring-up test sleeps between two heartbeat reads
int mtpu_stub_usleep(unsigned us)
{
    (void)us;
    *(uint32_t *)(ctrl + MINI_TPU_R_FW_HEARTBEAT) += 1;
    return 0;
}

int mtpu_open(mtpu_dev_t *d, uint64_t ctrl_phys, uint64_t min_buf_bytes)
{
    d->ctrl = ctrl;
    d->ctrl_phys = ctrl_phys;
    d->buf_size = min_buf_bytes;
    d->buf = calloc(1, min_buf_bytes);
    d->buf_phys = BUF_PHYS;
    *(uint32_t *)(ctrl + MINI_TPU_R_HW_ID) = MINI_TPU_HW_ID;
    *(uint32_t *)(ctrl + MINI_TPU_R_HW_ARCH) = MINI_TPU_ARCH_WORD(MINI_TPU_TILE_N, 0x0F, MINI_TPU_VMEM_KB);
    return d->buf ? 0 : -1;
}

void mtpu_close(mtpu_dev_t *d) { free(d->buf); }

int mtpu_read_hex(const char *path, uint32_t *words, int max_words)
{
    (void)path; (void)max_words;
    words[0] = 0;
    return 1;
}

int mtpu_boot(mtpu_dev_t *d, const uint32_t *image, int words)
{
    (void)d; (void)image; (void)words;
    *(uint32_t *)(ctrl + MINI_TPU_R_FW_VERSION) = 0x00010000u;
    return 0;
}

void mtpu_set_region(mtpu_dev_t *d, unsigned region, uint64_t addr)
{
    (void)d;
    regions[region] = addr;
}

int mtpu_sync(mtpu_dev_t *d, bool for_device, uint32_t off, uint32_t size)
{
    (void)d; (void)for_device; (void)off; (void)size;
    return 0;
}

uint32_t mtpu_exec(mtpu_dev_t *d, uint32_t prog_off, uint64_t timeout_us, uint64_t *wall_us)
{
    return mtpu_run(d, prog_off, (uint32_t)d->buf_size, timeout_us, wall_us);
}

uint32_t mtpu_run(mtpu_dev_t *d, uint32_t prog_off, uint32_t sync_bytes, uint64_t timeout_us, uint64_t *wall_us)
{
    (void)timeout_us;
    if (wall_us) *wall_us = 0;
    FILE *f = fopen("run.in", "wb");
    fwrite(d->buf, 1, sync_bytes, f);
    fclose(f);
    f = fopen("run.cfg", "w");
    fprintf(f, "%u %u %llu", prog_off, sync_bytes, (unsigned long long)d->buf_phys);
    for (unsigned r = 0; r < MINI_TPU_NUM_REGIONS; r++) fprintf(f, " %llu", (unsigned long long)regions[r]);
    fclose(f);
    if (system(getenv("REPLAY_CMD")) != 0) {
        fprintf(stderr, "replay.py failed\n");
        exit(3);
    }
    f = fopen("run.out", "rb");
    if (fread(d->buf, 1, sync_bytes, f) != sync_bytes) exit(3);
    fclose(f);
    unsigned status, desc_count, err_desc, err_unit, info0, info1;
    f = fopen("run.regs", "r");
    if (fscanf(f, "%u %u %u %u %u %u", &status, &desc_count, &err_desc, &err_unit, &info0, &info1) != 6) exit(3);
    fclose(f);
    *(uint32_t *)(ctrl + MINI_TPU_R_STATUS) = status;
    *(uint32_t *)(ctrl + MINI_TPU_R_DESC_COUNT) = desc_count;
    *(uint32_t *)(ctrl + MINI_TPU_R_ERR_DESC) = err_desc;
    *(uint32_t *)(ctrl + MINI_TPU_R_ERR_UNIT) = err_unit;
    *(uint32_t *)(ctrl + MINI_TPU_R_ERR_INFO0) = info0;
    *(uint32_t *)(ctrl + MINI_TPU_R_ERR_INFO1) = info1;
    return status;
}

void mtpu_read_perf(const mtpu_dev_t *d, mtpu_perf_t *p)
{
    (void)d;
    memset(p, 0, sizeof *p);                    // the simulator has no timing
    p->desc_count = *(uint32_t *)(ctrl + MINI_TPU_R_DESC_COUNT);
}

void mtpu_read_error(const mtpu_dev_t *d, uint32_t status, mtpu_error_t *e)
{
    (void)d;
    memset(e, 0, sizeof *e);
    e->status = status;
    e->code = MINI_TPU_ST_ERR_CODE(status);
    e->desc = *(uint32_t *)(ctrl + MINI_TPU_R_ERR_DESC) & 0xFFFFu;
    e->unit = *(uint32_t *)(ctrl + MINI_TPU_R_ERR_UNIT);
    e->info0 = *(uint32_t *)(ctrl + MINI_TPU_R_ERR_INFO0);
    e->info1 = *(uint32_t *)(ctrl + MINI_TPU_R_ERR_INFO1);
}

const char *mtpu_err_name(uint32_t code)
{
    (void)code;
    return "ERR";
}

uint64_t mtpu_now_us(void) { return (uint64_t)clock(); }
