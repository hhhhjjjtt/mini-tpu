// run_prefill — runs a compiled model on the KV260 and compares the result, bit for bit,
// with what the simulator computed.
//
//   run_prefill [-d dir] [-f fw.hex] [-n runs] [-o out.bin] [-t timeout_s]
//
// dir holds the compiler's output (sw/compiler/output/<name>/; `model` in a dist folder).
// The DMA buffer holds program, weights, params, input, output and the KV cache that prefill
// fills, in that order; region sizes are the file sizes and model.txt's kv_bytes. Exit code 0
// when every run is DONE with no output differing.

#define _GNU_SOURCE
#include <getopt.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "mtpu.h"
#include "mtpu_desc.h"

#ifdef __has_include
#if __has_include("fw_image.h")
#include "fw_image.h"
#define HAVE_FW_IMAGE 1
#endif
#endif

#define ALIGN           4096u
#define MAX_DIFFS_SHOWN 8

enum { PROGRAM, WEIGHTS, PARAMS, INPUT, OUTPUT, KV, NUM_PARTS };

static const char *const file_name[] = { "program.bin", "weights.bin", "params.bin", "golden_input.bin", "golden_output.bin", "model.txt" };
static const unsigned region_of[] = { 0, MINI_TPU_REGION_WEIGHTS, MINI_TPU_REGION_PARAMS, MINI_TPU_REGION_INPUT, MINI_TPU_REGION_OUTPUT,
                                      MINI_TPU_REGION_KV };

static uint32_t file_bytes[NUM_PARTS], offset[NUM_PARTS];

static void path_of(char *path, size_t n, const char *dir, const char *name)
{
    snprintf(path, n, "%s/%s", dir, name);
}

// "kv_bytes <n>" in model.txt
static long long kv_bytes(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return -1; }
    char line[512];
    long long v = -1;
    while (fgets(line, sizeof line, f))
        if (sscanf(line, "kv_bytes %lld", &v) == 1) break;
    fclose(f);
    if (v < 0) fprintf(stderr, "%s: no kv_bytes\n", path);
    return v;
}

static int read_file(const char *path, void *dst, uint32_t bytes)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    size_t got = fread(dst, 1, bytes, f);
    fclose(f);
    if (got != bytes) { fprintf(stderr, "%s: short read\n", path); return -1; }
    return 0;
}

static uint64_t compare(const mtpu_dev_t *d, const int32_t *expected)
{
    const int32_t *out = (const int32_t *)(d->buf + offset[OUTPUT]);
    uint32_t n = file_bytes[OUTPUT] / 4;
    uint64_t diffs = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (out[i] == expected[i]) continue;
        if (diffs++ < MAX_DIFFS_SHOWN)
            printf("    output %u: %d, expected %d\n", i, out[i], expected[i]);
    }
    return diffs;
}

static void print_perf(const mtpu_dev_t *d, uint64_t wall_us)
{
    static const char *const unit[] = { "LOAD", "MATRIX", "VECTOR", "STORE" };
    mtpu_perf_t p;
    mtpu_read_perf(d, &p);
    printf("    %" PRIu64 " cycles, %u descriptors, %.1f ms doorbell to DONE\n", p.cycles, p.desc_count, wall_us / 1000.0);
    for (unsigned u = 0; u < MINI_TPU_NUM_UNITS && p.cycles; u++)
        printf("    %-6s busy %5.1f %%, waiting %5.1f %%, starved %5.1f %%\n", unit[u], 100.0 * p.busy[u] / (double)p.cycles,
               100.0 * p.wait[u] / (double)p.cycles, 100.0 * p.starved[u] / (double)p.cycles);
}

static void print_error(const mtpu_dev_t *d, uint32_t status)
{
    mtpu_error_t e;
    mtpu_read_error(d, status, &e);
    printf("    STATUS 0x%08x %s: chunk %u, descriptor %u, unit %u, info 0x%08x 0x%08x\n", status, mtpu_err_name(e.code),
           e.chunk, e.desc, e.unit, e.info0, e.info1);
}

int main(int argc, char **argv)
{
    const char *dir = "model", *fw_hex = NULL, *out_path = NULL;
    int runs = 1, timeout_s = 120, c;
    while ((c = getopt(argc, argv, "d:f:n:o:t:")) != -1) {
        switch (c) {
        case 'd': dir = optarg; break;
        case 'f': fw_hex = optarg; break;
        case 'n': runs = atoi(optarg); break;
        case 'o': out_path = optarg; break;
        case 't': timeout_s = atoi(optarg); break;
        default:
            fprintf(stderr, "usage: %s [-d dir] [-f fw.hex] [-n runs] [-o out.bin] [-t timeout_s]\n", argv[0]);
            return 2;
        }
    }

    // ---- layout ----
    char path[512];
    uint64_t total = 0;
    for (int i = 0; i < NUM_PARTS; i++) {
        struct stat st;
        path_of(path, sizeof path, dir, file_name[i]);
        long long size = i == KV ? kv_bytes(path) : stat(path, &st) ? -1 : (long long)st.st_size;
        if (size < 0) { if (i != KV) perror(path); return 1; }
        file_bytes[i] = (uint32_t)size;
        offset[i] = (uint32_t)total;
        total += ((uint64_t)size + ALIGN - 1) / ALIGN * ALIGN;
    }

    // ---- firmware ----
    static uint32_t image[MINI_TPU_IMEM_WORDS];
    int words = -1;
    if (fw_hex) {
        words = mtpu_read_hex(fw_hex, image, MINI_TPU_IMEM_WORDS);
    } else {
#ifdef HAVE_FW_IMAGE
        memcpy(image, mini_tpu_fw_image, sizeof mini_tpu_fw_image);
        words = MINI_TPU_FW_IMAGE_WORDS;
#else
        fprintf(stderr, "no firmware built in (fw_image.h not found at build time): pass -f fw.hex\n");
#endif
    }
    if (words < 0) return 1;

    // ---- device ----
    mtpu_dev_t dev;
    if (mtpu_open(&dev, MTPU_CTRL_PHYS_DEFAULT, total)) return 1;
    uint32_t id = mtpu_rd(&dev, MINI_TPU_R_HW_ID);
    if (id != MINI_TPU_HW_ID) {
        fprintf(stderr, "HW_ID 0x%08x, expected 0x%08x: is mini_tpu.bit loaded? (make load)\n", id, MINI_TPU_HW_ID);
        return 1;
    }
    if (mtpu_boot(&dev, image, words)) return 1;

    // ---- data ----
    for (int i = 0; i < OUTPUT; i++) {
        path_of(path, sizeof path, dir, file_name[i]);
        if (read_file(path, dev.buf + offset[i], file_bytes[i])) return 1;
    }
    if (mtpu_check_header(dev.buf + offset[PROGRAM], file_name[PROGRAM], mtpu_rd(&dev, MINI_TPU_R_HW_ARCH), 0)) return 1;
    int32_t *expected = malloc(file_bytes[OUTPUT]);
    path_of(path, sizeof path, dir, file_name[OUTPUT]);
    if (!expected || read_file(path, expected, file_bytes[OUTPUT])) return 1;
    for (int i = WEIGHTS; i < NUM_PARTS; i++)
        mtpu_set_region(&dev, region_of[i], dev.buf_phys + offset[i]);
    printf("run_prefill: %s, %.1f MB in the DMA buffer at 0x%" PRIx64 "\n", dir, (double)total / (1 << 20), dev.buf_phys);

    // ---- run ----
    int failed = 0;
    for (int r = 0; r < runs; r++) {
        memset(dev.buf + offset[OUTPUT], 0xA5, file_bytes[OUTPUT]);     // stale data cannot pass
        uint64_t wall_us = 0;
        uint32_t status = mtpu_run(&dev, offset[PROGRAM], (uint32_t)total, (uint64_t)timeout_s * 1000000u, &wall_us);
        printf("run %d:\n", r);
        if (status == MTPU_RUN_FAILED) { failed = 1; break; }
        if (MINI_TPU_ST_STATE(status) != MINI_TPU_ST_DONE) {
            print_error(&dev, status);
            failed = 1;
            break;
        }
        print_perf(&dev, wall_us);
        uint64_t diffs = compare(&dev, expected);
        printf("    %" PRIu64 " of %u outputs differ\n", diffs, file_bytes[OUTPUT] / 4);
        if (diffs) failed = 1;
    }

    if (out_path) {
        FILE *f = fopen(out_path, "wb");
        if (!f) { perror(out_path); return 1; }
        fwrite(dev.buf + offset[OUTPUT], 1, file_bytes[OUTPUT], f);
        fclose(f);
    }
    printf("%s\n", failed ? "FAILED" : "PASSED");
    free(expected);
    mtpu_close(&dev);
    return failed;
}
