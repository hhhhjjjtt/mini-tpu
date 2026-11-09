// =============================================================================
// mtpu_test — mini-tpu bring-up test for the KV260 (ARM userspace)
// =============================================================================
//
// Runs on the board after mini_tpu.bit is loaded and u-dma-buf is inserted
// (see ../Makefile, ../README.md). Device access, firmware boot and the
// descriptor encoders are in libmtpu (../libmtpu/); this file builds the test
// programs (INTERFACE_SPEC §3-§5) in DMA memory, runs them and checks the
// results against reference math.
//
// The tests mirror hw/sim/tb/tb_system.sv so board and simulation run the same
// programs. All arithmetic checks are bit-exact.
//
//   sudo ./mtpu_test [-f fw.hex] [-s seed] [-t test[,test...]] [-v]
//   tests: id boot halt dma gemm loop attn smax head tput rate err   (default: all, in that order)
//
// The firmware is compiled in from fw_image.h (hw/fw/build/) when the build
// finds it; -f loads a fw.hex file instead.
// =============================================================================

#include <getopt.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "mtpu.h"
#include "mtpu_desc.h"

#ifdef __has_include
#if __has_include("fw_image.h")
#include "fw_image.h"
#define HAVE_FW_IMAGE 1
#endif
#endif

#define HW_ARCH_EXPECT  MINI_TPU_ARCH_WORD(MINI_TPU_TILE_N, MINI_TPU_UNITS_MASK, MINI_TPU_VMEM_KB)
#define N               ((int)MINI_TPU_WEIGHT_TILE_N)       // array width = lanes per weight tile
#define RUN_TIMEOUT_US  3000000u

// -----------------------------------------------------------------------------
// DMA buffer layout (offsets in /dev/udmabuf0; needs >= 8 MB)
// -----------------------------------------------------------------------------
#define OFF_PROG        0x000000u        // programs, 16 KB apart
#define OFF_R0          0x100000u        // weights
#define OFF_R1          0x180000u        // params (requant tables)
#define OFF_R2          0x200000u        // inputs
#define OFF_R3          0x400000u        // outputs
#define OFF_R4          0x600000u        // zero page
#define SYNC_SPAN       0x700000u
#define NEED_BYTES      0x800000u

#define PB_HALT         (OFF_PROG + 0x0000)
#define PB_GEMM         (OFF_PROG + 0x4000)
#define PB_LOOP         (OFF_PROG + 0x8000)
#define PB_RATE         (OFF_PROG + 0xC000)
#define PB_DMA          (OFF_PROG + 0x10000)
#define PB_ATTN         (OFF_PROG + 0x14000)
#define PB_ERR          (OFF_PROG + 0x18000)
#define PB_TPUT         (OFF_PROG + 0x1C000)
#define PB_SMAX         (OFF_PROG + 0x20000)
#define PB_HEAD         (OFF_PROG + 0x24000)

static inline uint32_t VB(int n) { return (uint32_t)n * 0x40000u; }   // vmem bank base

// -----------------------------------------------------------------------------
// Globals
// -----------------------------------------------------------------------------
static mtpu_dev_t dev;
static uint8_t *ubuf;                       // dev.buf
static bool verbose;
static const char *fw_src = "built-in image";
static uint32_t fw_img[MINI_TPU_IMEM_WORDS];
static int fw_words;

// -----------------------------------------------------------------------------
// Small helpers
// -----------------------------------------------------------------------------
static inline uint32_t rd(uint32_t off) { return mtpu_rd(&dev, off); }
static inline void wr(uint32_t off, uint32_t v) { mtpu_wr(&dev, off, v); }

static uint32_t rng_state = 1;
static uint32_t rnd32(void)                // xorshift32
{
    uint32_t x = rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return rng_state = x;
}
static int rnd_i8(void) { return (int8_t)(rnd32() & 0xFF); }

// -----------------------------------------------------------------------------
// Reference math (same as the testbench golden models)
// -----------------------------------------------------------------------------
static int32_t apply_scale_ref(int32_t v, int32_t m, int sh, int dr)
{
    int64_t prod = (int64_t)v * (int64_t)m;
    int64_t r = (sh == 0) ? 0 : ((int64_t)1 << (sh - 1));
    if (dr && sh > 31)
        r += (v >= 0) ? ((int64_t)1 << 30) : -((int64_t)1 << 30);
    return (int32_t)((prod + r) >> sh);
}

static int clamp_ref(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }

// VEC_SOFTMAX reference model: the integer algorithm of INTERFACE_SPEC §5.7
static uint32_t EXP2_LUT[256];

static void init_exp2_lut(void)
{
    for (int i = 0; i < 256; i++) EXP2_LUT[i] = (uint32_t)floor(65536.0 * pow(2.0, -i / 256.0) + 0.5);
}

static int64_t exp2neg_ref(int64_t z)                 // 2^-z, z >= 0 in Q16 -> Q0.16
{
    int64_t zr = z + 128, ip = zr >> 16;
    return ip > 16 ? 0 : (int64_t)(EXP2_LUT[(zr >> 8) & 255] >> ip);
}

static int32_t sat32(int64_t v) { return v > INT32_MAX ? INT32_MAX : v < INT32_MIN ? INT32_MIN : (int32_t)v; }

typedef struct {
    int32_t in_mult, in_zp, out_mult; int in_shift, out_shift, out_zp;
    double in_scale, out_inv;                          // real values the integer ones approximate
} smax_params_t;

// real scale -> (mult, shift), mult in [2^30, 2^31)
static void real_to_ms(double r, int32_t *m, int *sh)
{
    int e = 30 - (int)floor(log2(r));
    double v = floor(r * ldexp(1.0, e) + 0.5);
    if (v >= 2147483648.0) { v = floor(v / 2.0); e--; }
    *m = (int32_t)v; *sh = e;
}

static void smax_params(smax_params_t *p, double in_scale, int32_t in_zp, double out_inv, int out_zp)
{
    p->in_scale = in_scale; p->in_zp = in_zp; p->out_inv = out_inv; p->out_zp = out_zp;
    real_to_ms(in_scale * 1.4426950408889634 * 65536.0, &p->in_mult, &p->in_shift);
    real_to_ms(out_inv / 65536.0, &p->out_mult, &p->out_shift);
}

// one row: q (inputs), mask (t units) or NULL -> out (i8)
static void softmax_ref(const int32_t *q, const int32_t *mask, int len, const smax_params_t *p, int8_t *out)
{
    static int32_t t[1024];
    for (int i = 0; i < len; i++)
        t[i] = sat32((int64_t)apply_scale_ref(q[i] - p->in_zp, p->in_mult, p->in_shift, 0) + (mask ? mask[i] : 0));
    int64_t mx = INT32_MIN, sum = 0;
    for (int b = 0; b < len; b += 16) {
        int64_t bm = t[b];
        for (int i = b + 1; i < b + 16; i++) if (t[i] > bm) bm = t[i];
        int64_t mnew = bm > mx ? bm : mx, bsum = 0;
        for (int i = b; i < b + 16; i++) bsum += exp2neg_ref(mnew - t[i]);
        sum = ((sum * exp2neg_ref(mnew - mx)) >> 16) + bsum;
        mx = mnew;
    }
    int64_t R = ((int64_t)1 << 46) / sum;
    for (int i = 0; i < len; i++) {
        int32_t pq = apply_scale_ref((int32_t)exp2neg_ref(mx - t[i]), (int32_t)R, 30, 0);
        out[i] = (int8_t)clamp_ref(apply_scale_ref(pq, p->out_mult, p->out_shift, 0) + p->out_zp, -128, 127);
    }
}

// accuracy of one row against double-precision softmax: worst |diff| in i8 steps
static int smax_diff, smax_worst;
static void softmax_accuracy(const int32_t *q, const double *mask_real, int len, const smax_params_t *p,
                             const int8_t *got)
{
    double xm = -1e300, den = 0.0, x[1024];
    for (int i = 0; i < len; i++) {
        x[i] = (q[i] - p->in_zp) * p->in_scale + (mask_real ? mask_real[i] : 0.0);
        if (x[i] > xm) xm = x[i];
    }
    for (int i = 0; i < len; i++) den += exp(x[i] - xm);
    for (int i = 0; i < len; i++) {
        int ref = clamp_ref((int)floor(exp(x[i] - xm) / den * p->out_inv + 0.5) + p->out_zp, -128, 127);
        int d = got[i] > ref ? got[i] - ref : ref - got[i];
        if (d > smax_worst) smax_worst = d;
        if (d) smax_diff++;
    }
}

// requant table (§9.1): 16 entries x 16 B {bias i32, mult i32, shift u8}
static int32_t tb_bias[N], tb_mult[N];
static int tb_sh[N];

static void put_table(uint32_t off, int shmin, int shmax)
{
    for (int j = 0; j < N; j++) {
        tb_bias[j] = (int32_t)(rnd32() % 20001) - 10000;
        tb_mult[j] = (int32_t)(0x40000000u | (rnd32() & 0x3FFFFFFFu));
        tb_sh[j]   = shmin + (int)(rnd32() % (uint32_t)(shmax - shmin + 1));
        memset(ubuf + off + 16 * j, 0, 16);
        mtpu_put32(ubuf, off + 16 * j, (uint32_t)tb_bias[j]);
        mtpu_put32(ubuf, off + 16 * j + 4, (uint32_t)tb_mult[j]);
        mtpu_put8(ubuf, off + 16 * j + 8, (uint32_t)tb_sh[j]);
    }
}

// -----------------------------------------------------------------------------
// Firmware and regions
// -----------------------------------------------------------------------------
static void set_regions(void)
{
    const uint32_t offs[5] = { OFF_R0, OFF_R1, OFF_R2, OFF_R3, OFF_R4 };
    for (unsigned r = 0; r < MINI_TPU_NUM_REGIONS; r++)
        mtpu_set_region(&dev, r, r < 5 ? dev.buf_phys + offs[r] : 0);
    wr(MINI_TPU_R_WATCHDOG, 100000000u);          // 1 s at 100 MHz without unit progress
}

// hold the core in reset, write I-mem, verify, release, wait for FW_VERSION
static int boot_core(void) { return mtpu_boot(&dev, fw_img, fw_words); }

// -----------------------------------------------------------------------------
// Programs (descriptor encoders: ../libmtpu/mtpu_desc.c)
// -----------------------------------------------------------------------------
static mtpu_desc_t d_loop(uint32_t count, int body, int32_t s0, int32_t s3)   // strides on R0 and R3
{
    return mtpu_d_loop_stride(mtpu_d_loop_stride(mtpu_d_loop(count, (unsigned)body), 0, s0), 3, s3);
}

static mtpu_desc_t d_pass(uint32_t a, uint32_t o, uint32_t params)          // VEC_ELTWISE PASS_A, 1 x 16 i8
{
    return mtpu_d_eltwise(MINI_TPU_EW_PASS_A, MINI_TPU_DT_I8, MINI_TPU_DT_I8, MINI_TPU_DT_I8, MINI_TPU_EW_BCAST_FULL, 1, 16, a, a, o,
                          16, 16, 16, params);
}

static mtpu_desc_t d_softmax(bool i32, int rows, int len, int ipitch, uint32_t in, uint32_t out, int opitch,
                             const smax_params_t *p, uint32_t mask, bool has_mask)
{
    mtpu_smax_q_t q = { p->in_mult, p->in_zp, p->out_mult, p->in_shift, p->out_shift, p->out_zp };
    return mtpu_d_softmax(i32, (unsigned)rows, (unsigned)len, (unsigned)ipitch, in, out, (unsigned)opitch, &q,
                          mask, has_mask);
}

static void put_desc(uint32_t off, mtpu_desc_t d) { memcpy(ubuf + off, d.b, MINI_TPU_DESC_SIZE); }

static void put_header(uint32_t pb, uint32_t first_off, int first_len, int total_descs, uint8_t rmask)
{
    mtpu_put_header(ubuf + pb, HW_ARCH_EXPECT, rmask, first_off, (unsigned)first_len,
                    MINI_TPU_PROG_HDR_SIZE + MINI_TPU_DESC_SIZE * (uint32_t)total_descs, 0xF2B0);
}

// single-chunk program right after the header
static mtpu_desc_t prog[256];
static int pn;

static void emit_single(uint32_t pb)
{
    put_header(pb, 64, pn, pn, 0x1F);
    for (int i = 0; i < pn; i++) put_desc(pb + 64 + 64 * (uint32_t)i, prog[i]);
}

// -----------------------------------------------------------------------------
// Run a program; returns STATUS (or MTPU_RUN_FAILED)
// -----------------------------------------------------------------------------
static void dump_perf(const char *tag)
{
    mtpu_perf_t p;
    mtpu_read_perf(&dev, &p);
    printf("    %s: %" PRIu64 " cycles, %u descriptors\n", tag, p.cycles, p.desc_count);
    if (!verbose) return;
    static const char *names[4] = { "LOAD  ", "MATRIX", "VECTOR", "STORE " };
    for (int u = 0; u < 4; u++)
        printf("      %s busy %10u  wait %10u  starved %10u\n", names[u], p.busy[u], p.wait[u], p.starved[u]);
}

static void dump_error(uint32_t st)
{
    mtpu_error_t e;
    mtpu_read_error(&dev, st, &e);
    printf("    STATUS 0x%08x: %s (0x%02x), chunk %u desc %u, unit 0x%x, info 0x%08x 0x%08x\n", st,
           mtpu_err_name(e.code), e.code, e.chunk, e.desc, e.unit, e.info0, e.info1);
    if (e.has_trace)
        printf("    last trace record: cycle %u chunk/desc 0x%08x op/event 0x%08x info 0x%08x\n", e.trace[0],
               e.trace[1], e.trace[2], e.trace[3]);
}

static uint32_t run_prog(uint32_t pb, uint64_t *wall_us)
{
    uint32_t st = mtpu_run(&dev, pb, SYNC_SPAN, RUN_TIMEOUT_US, wall_us);
    if (st == MTPU_RUN_FAILED && MINI_TPU_ST_STATE(rd(MINI_TPU_R_STATUS)) == MINI_TPU_ST_RUNNING) dump_perf("at timeout");
    return st;
}

static bool expect_done(uint32_t st)
{
    if (st == MINI_TPU_ST_DONE) return true;
    if (st == MTPU_RUN_FAILED) return false;
    if (MINI_TPU_ST_STATE(st) == MINI_TPU_ST_ERROR) dump_error(st);
    else printf("    STATUS 0x%08x, expected DONE\n", st);
    return false;
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------
static int checked, mismatches;

static void mismatch(const char *what, int a, int b, long got, long exp)
{
    if (mismatches < 10) printf("    MISMATCH %s [%d][%d]: got %ld expected %ld\n", what, a, b, got, exp);
    mismatches++;
}

static bool t_id(void)
{
    uint32_t id = rd(MINI_TPU_R_HW_ID), ver = rd(MINI_TPU_R_HW_VERSION), arch = rd(MINI_TPU_R_HW_ARCH);
    printf("    HW_ID 0x%08x, HW_VERSION 0x%08x, HW_ARCH 0x%08x, CORE_CTRL 0x%08x\n", id, ver, arch,
           rd(MINI_TPU_R_CORE_CTRL));
    if (id != MINI_TPU_HW_ID) { printf("    not a mini-tpu bitstream (expected 0x%08x)\n", MINI_TPU_HW_ID); return false; }
    if (arch != HW_ARCH_EXPECT) { printf("    unexpected HW_ARCH\n"); return false; }
    return true;
}

static bool t_boot(void)
{
    set_regions();
    if (boot_core()) return false;
    uint32_t h0 = rd(MINI_TPU_R_FW_HEARTBEAT);
    usleep(1000);
    uint32_t h1 = rd(MINI_TPU_R_FW_HEARTBEAT);
    printf("    firmware 0x%08x booted from %s (%d words), heartbeat %u -> %u\n", rd(MINI_TPU_R_FW_VERSION), fw_src,
           fw_words, h0, h1);
    return h1 != h0;
}

static bool t_halt(void)
{
    pn = 0;
    prog[pn++] = mtpu_d_halt();
    emit_single(PB_HALT);
    uint64_t us;
    uint32_t st = run_prog(PB_HALT, &us);
    if (!expect_done(st)) return false;
    dump_perf("HALT-only program");
    printf("    wall time %" PRIu64 " us (doorbell -> DONE seen by the ARM)\n", us);
    return rd(MINI_TPU_R_DESC_COUNT) == 1;
}

// strided round trip with rows crossing 4 KB, then 64 KB bandwidth
static bool t_dma(void)
{
    const int rows = 5, rb = 4096 - 48;
    const uint32_t src = 0x2F00, sstr = rb + 32, dst = 0x10100, dstr = rb + 80;
    const uint32_t vm = VB(5), vstr = rb + 16;
    for (int r = 0; r < rows; r++) {
        for (uint32_t w = 0; w < dstr; w += 4) mtpu_put32(ubuf, OFF_R3 + dst + r * dstr + w, 0x5E471E00u + w);
        for (int b = 0; b < rb; b += 4) mtpu_put32(ubuf, OFF_R2 + src + r * sstr + b, rnd32());
    }
    pn = 0;
    prog[pn++] = mtpu_d_load(false, 2, src, vm, rows, rb, sstr, vstr);
    prog[pn++] = mtpu_d_wait(mtpu_d_load(true, 3, dst, vm, rows, rb, dstr, vstr), MINI_TPU_UNIT_LOAD, 0);
    prog[pn++] = mtpu_d_halt();
    emit_single(PB_DMA);
    if (!expect_done(run_prog(PB_DMA, NULL))) return false;
    int bad = mismatches;
    for (int r = 0; r < rows; r++)
        for (uint32_t w = 0; w < dstr; w += 4) {
            int32_t got = mtpu_get32s(ubuf, OFF_R3 + dst + r * dstr + w);
            int32_t exp = w < (uint32_t)rb ? mtpu_get32s(ubuf, OFF_R2 + src + r * sstr + w) : (int32_t)(0x5E471E00u + w);
            if (got != exp) mismatch(w < (uint32_t)rb ? "dma row" : "dma gap", r, (int)w, got, exp);
            checked++;
        }
    printf("    round trip: %d rows x %d B, strides %u / %u / %u\n", rows, rb, sstr, vstr, dstr);

    // bandwidth: 64 KB LOAD, then 64 KB STORE
    for (uint32_t w = 0; w < 0x10000; w += 4) mtpu_put32(ubuf, OFF_R2 + 0x20000 + w, rnd32());
    pn = 0;
    prog[pn++] = mtpu_d_load(false, 2, 0x20000, VB(6), 16, 4096, 4096, 4096);
    prog[pn++] = mtpu_d_wait(mtpu_d_load(true, 3, 0x20000, VB(6), 16, 4096, 4096, 4096), MINI_TPU_UNIT_LOAD, 0);
    prog[pn++] = mtpu_d_halt();
    emit_single(PB_DMA);
    if (!expect_done(run_prog(PB_DMA, NULL))) return false;
    if (memcmp(ubuf + OFF_R3 + 0x20000, ubuf + OFF_R2 + 0x20000, 0x10000)) mismatch("dma 64KB", 0, 0, 1, 0);
    checked += 0x10000 / 4;
    uint32_t lb = rd(MINI_TPU_R_UNIT_BUSY(MINI_TPU_UNIT_LOAD)), sb = rd(MINI_TPU_R_UNIT_BUSY(MINI_TPU_UNIT_STORE));
    printf("    64 KB: LOAD %u cycles (%.2f B/cycle, %.0f MB/s), STORE %u cycles (%.2f B/cycle, %.0f MB/s) at 100 MHz\n",
           lb, 65536.0 / lb, 6553.6 / lb * 1000.0, sb, 65536.0 / sb, 6553.6 / sb * 1000.0);
    return mismatches == bad;
}

// P1: LOAD x3 -> GEMM -> VEC_EPILOGUE (per-channel requant) -> STORE
static int8_t A[32][128];
static int8_t W[4][128][N];

static bool t_gemm(void)
{
    for (int r = 0; r < 32; r++)
        for (int t = 0; t < 64; t++) { A[r][t] = (int8_t)rnd_i8(); ubuf[OFF_R2 + r * 64 + t] = (uint8_t)A[r][t]; }
    for (int t = 0; t < 64; t++)
        for (int j = 0; j < N; j++) { W[0][t][j] = (int8_t)rnd_i8(); ubuf[OFF_R0 + t * 16 + j] = (uint8_t)W[0][t][j]; }
    put_table(OFF_R1, 30, 40);
    memset(ubuf + OFF_R3, 0xA5, 32 * 16);

    pn = 0;
    prog[pn++] = mtpu_d_load(false, 1, 0, VB(1), 1, 256, 256, 256);               // params
    prog[pn++] = mtpu_d_load(false, 2, 0, VB(0), 32, 64, 64, 64);                 // A
    prog[pn++] = mtpu_d_load_wgt(0, 0, 0, 64);                                       // W -> half 0
    prog[pn++] = mtpu_d_wait(mtpu_d_gemm(VB(0), 64, 2, 64, 0), MINI_TPU_UNIT_LOAD, 0);
    prog[pn++] = mtpu_d_wait(mtpu_d_epilogue(0, 0x2, 5, -128, 127, 32, 16, VB(3), VB(1)), MINI_TPU_UNIT_LOAD, 0);
    prog[pn++] = mtpu_d_wait(mtpu_d_load(true, 3, 0, VB(3), 32, 16, 16, 16), MINI_TPU_UNIT_VECTOR, 0);
    prog[pn++] = mtpu_d_halt();
    emit_single(PB_GEMM);
    if (!expect_done(run_prog(PB_GEMM, NULL))) return false;
    dump_perf("GEMM 32x64x16 + requant");

    int bad = mismatches;
    for (int r = 0; r < 32; r++)
        for (int j = 0; j < N; j++) {
            int32_t c = 0;
            for (int t = 0; t < 64; t++) c += A[r][t] * W[0][t][j];
            int exp = clamp_ref(apply_scale_ref(c + tb_bias[j], tb_mult[j], tb_sh[j], 1) + 5, -128, 127);
            int got = (int8_t)ubuf[OFF_R3 + r * 16 + j];
            if (got != exp) mismatch("gemm", r, j, got, exp);
            checked++;
        }
    return mismatches == bad && rd(MINI_TPU_R_DESC_COUNT) == 7;
}

// P2: two chunks, NEXT_CHUNK prefetch, LOOP x2, weight ping-pong, waits across all units
static bool t_loop(void)
{
    for (int r = 0; r < 32; r++)
        for (int t = 0; t < 128; t++) { A[r][t] = (int8_t)rnd_i8(); ubuf[OFF_R2 + 0x1000 + r * 128 + t] = (uint8_t)A[r][t]; }
    for (int s = 0; s < 4; s++)
        for (int t = 0; t < 128; t++)
            for (int j = 0; j < N; j++) {
                W[s][t][j] = (int8_t)rnd_i8();
                ubuf[OFF_R0 + 0x2000 + s * 2048 + t * 16 + j] = (uint8_t)W[s][t][j];
            }
    put_table(OFF_R1 + 0x100, 26, 34);
    memset(ubuf + OFF_R3 + 0x1000, 0xA5, 32 * 64);

    const uint32_t pb = PB_LOOP, c1 = pb + 0x400;
    put_header(pb, 64, 4, 15, 0x0F);
    put_desc(pb + 64 + 64 * 0, mtpu_d_load(false, 1, 0x100, VB(1) + 0x100, 1, 256, 256, 256));
    put_desc(pb + 64 + 64 * 1, mtpu_d_load(false, 2, 0x1000, VB(0) + 0x8000, 32, 128, 128, 128));
    put_desc(pb + 64 + 64 * 2, mtpu_d_nop());                                       // NOP
    put_desc(pb + 64 + 64 * 3, mtpu_d_next_chunk(c1 - pb, 11));                                // NEXT_CHUNK

    put_desc(c1 + 64 * 0, d_loop(2, 8, 4096, 32));
    put_desc(c1 + 64 * 1, mtpu_d_wait(mtpu_d_load_wgt(0, 0x2000, 0, 128), MINI_TPU_UNIT_MATRIX, 1));             // WAR on half 0
    put_desc(c1 + 64 * 2, mtpu_d_wait(mtpu_d_load_wgt(0, 0x2000 + 2048, 1, 128), MINI_TPU_UNIT_MATRIX, 0));      // WAR on half 1
    put_desc(c1 + 64 * 3, mtpu_d_wait(mtpu_d_gemm(VB(0) + 0x8000, 128, 2, 128, 0), MINI_TPU_UNIT_LOAD, 1));
    put_desc(c1 + 64 * 4, mtpu_d_wait(mtpu_d_wait(mtpu_d_epilogue(0, 0, -7, -128, 127, 32, 16, VB(3) + 0x4000, VB(1) + 0x100),
                                        MINI_TPU_UNIT_LOAD, 3), MINI_TPU_UNIT_STORE, 1));
    put_desc(c1 + 64 * 5, mtpu_d_wait(mtpu_d_gemm(VB(0) + 0x8000, 128, 2, 128, 1), MINI_TPU_UNIT_LOAD, 0));
    put_desc(c1 + 64 * 6, mtpu_d_wait(mtpu_d_epilogue(0, 0, -7, -128, 127, 32, 16, VB(3) + 0x5000, VB(1) + 0x100), MINI_TPU_UNIT_STORE, 0));
    put_desc(c1 + 64 * 7, mtpu_d_wait(mtpu_d_load(true, 3, 0x1000, VB(3) + 0x4000, 32, 16, 64, 16), MINI_TPU_UNIT_VECTOR, 1));
    put_desc(c1 + 64 * 8, mtpu_d_wait(mtpu_d_load(true, 3, 0x1010, VB(3) + 0x5000, 32, 16, 64, 16), MINI_TPU_UNIT_VECTOR, 0));
    put_desc(c1 + 64 * 9, mtpu_d_fence());                                            // FENCE
    put_desc(c1 + 64 * 10, mtpu_d_halt());                                           // HALT

    if (!expect_done(run_prog(pb, NULL))) return false;
    dump_perf("2 chunks, LOOP x2, 4 weight tiles");

    int bad = mismatches;
    for (int s = 0; s < 4; s++)
        for (int r = 0; r < 32; r++)
            for (int j = 0; j < N; j++) {
                int32_t c = 0;
                for (int t = 0; t < 128; t++) c += A[r][t] * W[s][t][j];
                int exp = clamp_ref(apply_scale_ref(c + tb_bias[j], tb_mult[j], tb_sh[j], 0) - 7, -128, 127);
                int got = (int8_t)ubuf[OFF_R3 + 0x1000 + r * 64 + s * 16 + j];
                if (got != exp) mismatch("loop", r, s * 16 + j, got, exp);
                checked++;
            }
    return mismatches == bad && rd(MINI_TPU_R_DESC_COUNT) == 23;
}

// P7: one attention head on-chip, weights from vmem (LOAD_WGT_VM)
static bool t_attn(void)
{
    enum { M = 32, D = 32, S = 16 };
    static int8_t Q[M][D], K[S][D], V[S][N];
    static int S8[M][S];
    for (int r = 0; r < M; r++) for (int t = 0; t < D; t++) { Q[r][t] = (int8_t)rnd_i8(); ubuf[OFF_R2 + 0x8000 + r * D + t] = (uint8_t)Q[r][t]; }
    for (int j = 0; j < S; j++) for (int t = 0; t < D; t++) { K[j][t] = (int8_t)rnd_i8(); ubuf[OFF_R2 + 0x9000 + j * D + t] = (uint8_t)K[j][t]; }
    for (int t = 0; t < S; t++) for (int j = 0; j < N; j++) { V[t][j] = (int8_t)rnd_i8(); ubuf[OFF_R2 + 0xA000 + t * N + j] = (uint8_t)V[t][j]; }
    put_table(OFF_R1 + 0x300, 37, 39);
    memset(ubuf + OFF_R3 + 0x30000, 0xA5, M * 64);

    const uint32_t qv = VB(0) + 0x10000, kv = VB(1) + 0x10000, vv = VB(2) + 0x10000;
    const uint32_t sv = VB(3) + 0x10000, ov = VB(4) + 0x10000, pv = VB(5) + 0x10000;
    pn = 0;
    prog[pn++] = mtpu_d_load(false, 1, 0x300, pv, 1, 256, 256, 256);
    prog[pn++] = mtpu_d_load(false, 2, 0x8000, qv, M, D, D, D);
    prog[pn++] = mtpu_d_load(false, 2, 0x9000, kv, S, D, D, D);
    prog[pn++] = mtpu_d_load(false, 2, 0xA000, vv, S, N, N, N);
    prog[pn++] = mtpu_d_wait(mtpu_d_load_wgt_vm(kv, D, D, 0, true), MINI_TPU_UNIT_LOAD, 0);                          // K^T -> half 0
    prog[pn++] = mtpu_d_wait(mtpu_d_gemm(qv, D, M / 16, D, 0), MINI_TPU_UNIT_LOAD, 0);                      // Q . K^T
    prog[pn++] = mtpu_d_wait(mtpu_d_epilogue(0, 0x3, 3, -128, 127, M, S, sv, pv), MINI_TPU_UNIT_LOAD, 0);        // scores, per tensor
    prog[pn++] = mtpu_d_wait(mtpu_d_load_wgt_vm(vv, N, S, 1, false), MINI_TPU_UNIT_LOAD, 0);                         // V -> half 1
    prog[pn++] = mtpu_d_wait(mtpu_d_gemm(sv, S, M / 16, S, 1), MINI_TPU_UNIT_VECTOR, 0);                    // S . V
    prog[pn++] = mtpu_d_epilogue(1, 0, 0, 0, 0, M, 64, ov, pv);                                 // STORE_I32
    prog[pn++] = mtpu_d_wait(mtpu_d_load(true, 3, 0x30000, ov, M, 64, 64, 64), MINI_TPU_UNIT_VECTOR, 0);
    prog[pn++] = mtpu_d_halt();
    emit_single(PB_ATTN);
    if (!expect_done(run_prog(PB_ATTN, NULL))) return false;
    dump_perf("attention head Q.K^T -> S.V");

    for (int r = 0; r < M; r++)
        for (int j = 0; j < S; j++) {
            int32_t c = 0;
            for (int t = 0; t < D; t++) c += Q[r][t] * K[j][t];
            S8[r][j] = clamp_ref(apply_scale_ref(c + tb_bias[0], tb_mult[0], tb_sh[0], 1) + 3, -128, 127);
        }
    int bad = mismatches;
    for (int r = 0; r < M; r++)
        for (int j = 0; j < N; j++) {
            int32_t c = 0;
            for (int t = 0; t < S; t++) c += S8[r][t] * V[t][j];
            int32_t got = mtpu_get32s(ubuf, OFF_R3 + 0x30000 + (uint32_t)(r * 64 + j * 4));
            if (got != c) mismatch("attn", r, j, got, c);
            checked++;
        }
    return mismatches == bad;
}

// GEMM throughput: A (256 x 512) . W (512 x 64) as 4 weight tiles of K = 512, each
// GEMM 16 M-tiles; the next weight tile loads into the other half while the
// current GEMM runs. Peak is 256 MAC/cycle (16 x 16 array).
static bool t_tput(void)
{
    enum { M = 256, K = 512, T = 4 };
    static int8_t Ab[M][K], Wb[T][K][N];
    for (int r = 0; r < M; r++)
        for (int t = 0; t < K; t++) { Ab[r][t] = (int8_t)rnd_i8(); ubuf[OFF_R2 + 0x100000 + r * K + t] = (uint8_t)Ab[r][t]; }
    for (int s = 0; s < T; s++)
        for (int t = 0; t < K; t++)
            for (int j = 0; j < N; j++) {
                Wb[s][t][j] = (int8_t)rnd_i8();
                ubuf[OFF_R0 + 0x10000 + s * K * 16 + t * 16 + j] = (uint8_t)Wb[s][t][j];
            }
    put_table(OFF_R1 + 0x400, 36, 40);
    memset(ubuf + OFF_R3 + 0x40000, 0xA5, M * T * 16);

    pn = 0;
    prog[pn++] = mtpu_d_load(false, 1, 0x400, VB(1) + 0x400, 1, 256, 256, 256);          // params
    prog[pn++] = mtpu_d_load(false, 2, 0x100000, VB(0), M, K, K, K);                      // A, 128 KB
    for (int s = 0; s < T; s++) {
        prog[pn++] = mtpu_d_wait(mtpu_d_load_wgt(0, 0x10000 + (uint32_t)(s * K * 16), s & 1, K), MINI_TPU_UNIT_MATRIX, 1);  // WAR on the half
        prog[pn++] = mtpu_d_wait(mtpu_d_gemm(VB(0), K, M / 16, K, s & 1), MINI_TPU_UNIT_LOAD, 0);
        prog[pn++] = mtpu_d_wait(mtpu_d_epilogue(0, 0x2, 0, -128, 127, M, T * 16, VB(3) + (uint32_t)(16 * s), VB(1) + 0x400),
                            MINI_TPU_UNIT_LOAD, 0);
    }
    prog[pn++] = mtpu_d_wait(mtpu_d_load(true, 3, 0x40000, VB(3), M, T * 16, T * 16, T * 16), MINI_TPU_UNIT_VECTOR, 0);
    prog[pn++] = mtpu_d_halt();
    emit_single(PB_TPUT);
    if (!expect_done(run_prog(PB_TPUT, NULL))) return false;
    dump_perf("GEMM 256x512x64 (4 weight tiles)");

    uint64_t cyc = (uint64_t)rd(MINI_TPU_R_CYCLES_HI) << 32 | rd(MINI_TPU_R_CYCLES_LO);
    uint32_t mb = rd(MINI_TPU_R_UNIT_BUSY(MINI_TPU_UNIT_MATRIX)), lb = rd(MINI_TPU_R_UNIT_BUSY(MINI_TPU_UNIT_LOAD)), vb = rd(MINI_TPU_R_UNIT_BUSY(MINI_TPU_UNIT_VECTOR));
    double macs = (double)M * K * T * N;
    printf("    %.2f M MACs: %.1f MAC/cycle over the program (%.0f %% of 256 peak),\n", macs / 1e6, macs / cyc,
           100.0 * macs / cyc / 256.0);
    printf("    %.1f MAC/cycle while MATRIX busy (%.0f %% of peak); busy LOAD %u, MATRIX %u, VECTOR %u of %" PRIu64 " cycles\n",
           macs / mb, 100.0 * macs / mb / 256.0, lb, mb, vb, cyc);
    printf("    %.2f GMAC/s at 100 MHz\n", macs / cyc * 0.1);

    int bad = mismatches;
    for (int s = 0; s < T; s++)
        for (int r = 0; r < M; r++)
            for (int j = 0; j < N; j++) {
                int32_t c = 0;
                for (int t = 0; t < K; t++) c += Ab[r][t] * Wb[s][t][j];
                int exp = clamp_ref(apply_scale_ref(c + tb_bias[j], tb_mult[j], tb_sh[j], 1), -128, 127);
                int got = (int8_t)ubuf[OFF_R3 + 0x40000 + r * T * 16 + s * 16 + j];
                if (got != exp) mismatch("tput", r, s * 16 + j, got, exp);
                checked++;
            }
    return mismatches == bad && rd(MINI_TPU_R_DESC_COUNT) == 2 + 3 * T + 2;
}

// VEC_SOFTMAX on its own: DDR -> vmem -> softmax -> vmem -> DDR, three cases
static bool t_smax(void)
{
    struct { bool i32; int rows, len, pad; double scale; int zp, qlo, qhi; bool mask; } cs[3] = {
        { false, 4, 64,  0, 0.08,   3, -128, 127,     false },     // i8 rows
        { true,  3, 48, 32, 0.0015, 0, -20000, 20000, true  },     // i32 rows, mask, padded pitch
        { false, 8, 16,  0, 0.02,   0, -128, 127,     false },     // flat: every output mid-range
    };
    static int32_t q[8][1024], mask[1024];
    static double mask_real[1024];
    static int8_t ref[1024];
    int bad = mismatches, n_out = 0;
    smax_diff = smax_worst = 0;

    for (int c = 0; c < 3; c++) {
        smax_params_t p;
        smax_params(&p, cs[c].scale, cs[c].zp, 256.0, -128);
        const int eb = cs[c].i32 ? 4 : 1, ip = cs[c].len * eb + cs[c].pad, op = cs[c].len + cs[c].pad;
        const uint32_t in_d = 0x40000 + (uint32_t)c * 0x4000, mask_d = 0x50000 + (uint32_t)c * 0x1000;  // R2
        const uint32_t out_d = 0x50000 + (uint32_t)c * 0x1000;                                         // R3
        const uint32_t in_v = VB(4) + (uint32_t)c * 0x8000, mask_v = VB(5) + (uint32_t)c * 0x1000;
        const uint32_t out_v = VB(6) + (uint32_t)c * 0x1000;

        for (int i = 0; i < cs[c].len; i++) {
            mask_real[i] = 0.0;
            if (cs[c].mask) {
                uint32_t k = rnd32() % 4;
                mask_real[i] = k == 0 ? -10000.0 : k == 1 ? -(double)(rnd32() % 1000) / 250.0 : 0.0;
            }
            mask[i] = (int32_t)floor(mask_real[i] * 1.4426950408889634 * 65536.0 + 0.5);
            mtpu_put32(ubuf, OFF_R2 + mask_d + 4 * (uint32_t)i, (uint32_t)mask[i]);
        }
        for (int r = 0; r < cs[c].rows; r++)
            for (int i = 0; i < cs[c].len; i++) {
                q[r][i] = cs[c].qlo + (int32_t)(rnd32() % (uint32_t)(cs[c].qhi - cs[c].qlo + 1));
                if (cs[c].i32) mtpu_put32(ubuf, OFF_R2 + in_d + (uint32_t)(r * ip + 4 * i), (uint32_t)q[r][i]);
                else           ubuf[OFF_R2 + in_d + (uint32_t)(r * ip + i)] = (uint8_t)q[r][i];
            }
        memset(ubuf + OFF_R3 + out_d, 0x5A, (size_t)(cs[c].rows * op));

        pn = 0;
        prog[pn++] = mtpu_d_load(false, 2, in_d, in_v, cs[c].rows, cs[c].len * eb, (uint32_t)ip, (uint32_t)ip);
        if (cs[c].mask) prog[pn++] = mtpu_d_load(false, 2, mask_d, mask_v, 1, cs[c].len * 4, 0, 0);
        prog[pn++] = mtpu_d_wait(d_softmax(cs[c].i32, cs[c].rows, cs[c].len, ip, in_v, out_v, op, &p, mask_v,
                                      cs[c].mask), MINI_TPU_UNIT_LOAD, 0);
        prog[pn++] = mtpu_d_wait(mtpu_d_load(true, 3, out_d, out_v, cs[c].rows, cs[c].len, (uint32_t)op, (uint32_t)op),
                            MINI_TPU_UNIT_VECTOR, 0);
        prog[pn++] = mtpu_d_halt();
        emit_single(PB_SMAX);
        if (!expect_done(run_prog(PB_SMAX, NULL))) return false;
        dump_perf(c == 0 ? "softmax i8 4 x 64" : c == 1 ? "softmax i32 + mask 3 x 48" : "softmax flat 8 x 16");

        for (int r = 0; r < cs[c].rows; r++) {
            softmax_ref(q[r], cs[c].mask ? mask : NULL, cs[c].len, &p, ref);
            int8_t *got = (int8_t *)(ubuf + OFF_R3 + out_d + (uint32_t)(r * op));
            for (int i = 0; i < cs[c].len; i++) {
                if (got[i] != ref[i]) mismatch("smax", r, i, got[i], ref[i]);
                checked++; n_out++;
            }
            softmax_accuracy(q[r], cs[c].mask ? mask_real : NULL, cs[c].len, &p, got);
        }
    }
    printf("    vs double-precision softmax: %d of %d outputs differ, worst %d i8 step(s)\n",
           smax_diff, n_out, smax_worst);
    return mismatches == bad && smax_worst <= 1;
}

// Attention head fully on-chip: S = requant(Q.K^T) (32 keys = 2 K^T tiles), P = softmax(S),
// O = P.V (i32). Every step is checked against the reference models.
static bool t_head(void)
{
    enum { M = 32, D = 32, S = 32 };
    static int8_t Q[M][D], K[S][D], V[S][N];
    static int32_t Sq[M][S];
    static int8_t P[M][S];
    for (int r = 0; r < M; r++) for (int t = 0; t < D; t++) { Q[r][t] = (int8_t)rnd_i8(); ubuf[OFF_R2 + 0x60000 + r * D + t] = (uint8_t)Q[r][t]; }
    for (int j = 0; j < S; j++) for (int t = 0; t < D; t++) { K[j][t] = (int8_t)rnd_i8(); ubuf[OFF_R2 + 0x61000 + j * D + t] = (uint8_t)K[j][t]; }
    for (int t = 0; t < S; t++) for (int j = 0; j < N; j++) { V[t][j] = (int8_t)rnd_i8(); ubuf[OFF_R2 + 0x62000 + t * N + j] = (uint8_t)V[t][j]; }
    put_table(OFF_R1 + 0x500, 36, 38);
    memset(ubuf + OFF_R3 + 0x58000, 0xA5, M * 64);
    smax_params_t sp;
    smax_params(&sp, 0.05, 0, 127.0, 0);                    // P = round(p * 127), zero point 0

    const uint32_t qv = VB(0) + 0x20000, kv = VB(1) + 0x20000, vv = VB(2) + 0x20000, sv = VB(3) + 0x20000;
    const uint32_t ppv = VB(5) + 0x20000, ov = VB(4) + 0x20000, tv = VB(6) + 0x20000;
    pn = 0;
    prog[pn++] = mtpu_d_load(false, 1, 0x500, tv, 1, 256, 256, 256);                  // requant table
    prog[pn++] = mtpu_d_load(false, 2, 0x60000, qv, M, D, D, D);
    prog[pn++] = mtpu_d_load(false, 2, 0x61000, kv, S, D, D, D);
    prog[pn++] = mtpu_d_load(false, 2, 0x62000, vv, S, N, N, N);
    for (int h = 0; h < 2; h++) {                                                // 16 keys per K^T tile
        prog[pn++] = mtpu_d_wait(mtpu_d_load_wgt_vm(kv + (uint32_t)(16 * h * D), D, D, h, true), MINI_TPU_UNIT_LOAD, 0);
        prog[pn++] = mtpu_d_wait(mtpu_d_gemm(qv, D, M / 16, D, h), MINI_TPU_UNIT_LOAD, 0);
        prog[pn++] = mtpu_d_wait(mtpu_d_epilogue(0, 0x3, 0, -128, 127, M, S, sv + (uint32_t)(16 * h), tv), MINI_TPU_UNIT_LOAD, 0);
    }
    prog[pn++] = d_softmax(false, M, S, S, sv, ppv, S, &sp, 0, false);           // VECTOR, in order after the epilogues
    prog[pn++] = mtpu_d_load_wgt_vm(vv, N, S, 0, false);                                       // V -> half 0 (after both GEMMs)
    prog[pn++] = mtpu_d_wait(mtpu_d_gemm(ppv, S, M / 16, S, 0), MINI_TPU_UNIT_VECTOR, 0);              // P . V once softmax is done
    prog[pn++] = mtpu_d_epilogue(1, 0, 0, 0, 0, M, 64, ov, tv);                            // STORE_I32
    prog[pn++] = mtpu_d_wait(mtpu_d_load(true, 3, 0x58000, ov, M, 64, 64, 64), MINI_TPU_UNIT_VECTOR, 0);
    prog[pn++] = mtpu_d_halt();
    emit_single(PB_HEAD);
    if (!expect_done(run_prog(PB_HEAD, NULL))) return false;
    dump_perf("attention head Q.K^T -> softmax -> P.V (32 queries, 32 keys)");

    for (int r = 0; r < M; r++)
        for (int j = 0; j < S; j++) {
            int32_t c = 0;
            for (int t = 0; t < D; t++) c += Q[r][t] * K[j][t];
            Sq[r][j] = clamp_ref(apply_scale_ref(c + tb_bias[0], tb_mult[0], tb_sh[0], 1), -128, 127);
        }
    smax_diff = smax_worst = 0;
    for (int r = 0; r < M; r++) {
        softmax_ref(Sq[r], NULL, S, &sp, P[r]);
        softmax_accuracy(Sq[r], NULL, S, &sp, P[r]);
    }
    int bad = mismatches;
    for (int r = 0; r < M; r++)
        for (int j = 0; j < N; j++) {
            int32_t c = 0;
            for (int t = 0; t < S; t++) c += P[r][t] * V[t][j];
            int32_t got = mtpu_get32s(ubuf, OFF_R3 + 0x58000 + (uint32_t)(r * 64 + j * 4));
            if (got != c) mismatch("head", r, j, got, c);
            checked++;
        }
    printf("    softmax step vs double precision: worst %d i8 step(s)\n", smax_worst);
    return mismatches == bad;
}

// P3: dispatch rate, LOOP x64 over 4 tiny ops
static bool t_rate(void)
{
    memset(ubuf + OFF_R4, 0, 0x1000);
    pn = 0;
    prog[pn++] = mtpu_d_load(false, 4, 0, VB(4), 1, 512, 512, 512);      // zero PASS inputs + params
    prog[pn++] = mtpu_d_fence();                                    // FENCE (once, outside the loop)
    prog[pn++] = d_loop(64, 4, 0, 0);
    prog[pn++] = d_pass(VB(4), VB(6), VB(4) + 0x100);
    prog[pn++] = mtpu_d_load(false, 2, 0, VB(7), 1, 16, 16, 16);
    prog[pn++] = d_pass(VB(4) + 16, VB(6) + 16, VB(4) + 0x100);
    prog[pn++] = mtpu_d_load(false, 2, 16, VB(7) + 16, 1, 16, 16, 16);
    prog[pn++] = mtpu_d_halt();
    emit_single(PB_RATE);
    uint64_t us;
    if (!expect_done(run_prog(PB_RATE, &us))) return false;
    uint64_t cyc = (uint64_t)rd(MINI_TPU_R_CYCLES_HI) << 32 | rd(MINI_TPU_R_CYCLES_LO);
    uint32_t dc = rd(MINI_TPU_R_DESC_COUNT);
    printf("    %u descriptors in %" PRIu64 " cycles: %.1f cycles/descriptor (simulation: ~141)\n", dc, cyc,
           (double)cyc / dc);
    printf("    wall time %" PRIu64 " us\n", us);
    return dc == 260;
}

// bad opcode -> ERROR, then CORE_RESET recovery and a clean GEMM
static bool t_err(void)
{
    pn = 0;
    prog[pn++] = mtpu_d_nop();
    prog[pn++] = mtpu_d_nop();
    prog[pn++] = mtpu_d_op(0x77, 0, 0);
    prog[pn++] = mtpu_d_halt();
    emit_single(PB_ERR);
    uint32_t st = run_prog(PB_ERR, NULL);
    bool ok = (st == 0x00001004u) && rd(MINI_TPU_R_ERR_DESC) == 2 && rd(MINI_TPU_R_ERR_INFO0) == 0x77;
    printf("    bad opcode: STATUS 0x%08x (expect 0x00001004), ERR_DESC %u, ERR_INFO0 0x%x\n", st,
           rd(MINI_TPU_R_ERR_DESC), rd(MINI_TPU_R_ERR_INFO0));
    if (boot_core()) return false;                                   // CORE_RESET + reload
    printf("    recovered with CORE_RESET; rerunning the GEMM test\n");
    return ok && t_gemm();
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
typedef struct { const char *name; bool (*fn)(void); const char *what; } test_t;

static const test_t tests[] = {
    { "id",   t_id,   "identity registers" },
    { "boot", t_boot, "firmware load + boot" },
    { "halt", t_halt, "HALT-only program (descriptor fetch from DDR)" },
    { "dma",  t_dma,  "LOAD/STORE round trip + bandwidth" },
    { "gemm", t_gemm, "GEMM + per-channel requant" },
    { "loop", t_loop, "two chunks, LOOP, weight ping-pong, waits" },
    { "attn", t_attn, "attention head with LOAD_WGT_VM" },
    { "smax", t_smax, "VEC_SOFTMAX: i8, i32 + mask, flat rows" },
    { "head", t_head, "attention head: Q.K^T -> softmax -> P.V" },
    { "tput", t_tput, "GEMM throughput (256 x 512 x 64)" },
    { "rate", t_rate, "dispatch rate" },
    { "err",  t_err,  "error report + CORE_RESET recovery" },
};
#define NTESTS ((int)(sizeof tests / sizeof tests[0]))

static void usage(const char *p)
{
    printf("usage: sudo %s [-f fw.hex] [-s seed] [-t test[,test...]] [-v]\ntests:", p);
    for (int i = 0; i < NTESTS; i++) printf(" %s", tests[i].name);
    printf("\n");
}

int main(int argc, char **argv)
{
    char *sel = NULL;
    const char *fw_path = NULL;
    int c;
    rng_state = 0x2545F491u;
    while ((c = getopt(argc, argv, "f:s:t:vh")) != -1) {
        switch (c) {
        case 'f': fw_path = optarg; break;
        case 's': rng_state = (uint32_t)strtoul(optarg, NULL, 0) | 1u; break;
        case 't': sel = optarg; break;
        case 'v': verbose = true; break;
        default:  usage(argv[0]); return c == 'h' ? 0 : 2;
        }
    }
    init_exp2_lut();
    if (fw_path) {
        if ((fw_words = mtpu_read_hex(fw_path, fw_img, MINI_TPU_IMEM_WORDS)) < 0) return 1;
        fw_src = fw_path;
    } else {
#ifdef HAVE_FW_IMAGE
        memcpy(fw_img, mini_tpu_fw_image, sizeof mini_tpu_fw_image);
        fw_words = MINI_TPU_FW_IMAGE_WORDS;
#else
        fprintf(stderr, "no firmware built in (fw_image.h not found at build time): pass -f fw.hex\n");
        return 1;
#endif
    }
    if (mtpu_open(&dev, MTPU_CTRL_PHYS_DEFAULT, NEED_BYTES)) return 1;
    ubuf = dev.buf;
    printf("mtpu_test: ctrl 0x%08" PRIx64 ", udmabuf0 phys 0x%" PRIx64 " size %" PRIu64 " MB, firmware %s (%d words)\n",
           dev.ctrl_phys, dev.buf_phys, dev.buf_size >> 20, fw_src, fw_words);

    bool want[NTESTS];
    for (int i = 0; i < NTESTS; i++) want[i] = (sel == NULL);
    if (sel) {
        for (char *tok = strtok(sel, ","); tok; tok = strtok(NULL, ",")) {
            int i;
            for (i = 0; i < NTESTS && strcmp(tok, tests[i].name); i++) ;
            if (i == NTESTS) { fprintf(stderr, "unknown test '%s'\n", tok); usage(argv[0]); return 2; }
            want[i] = true;
        }
        want[0] = want[1] = true;                                    // id + boot always run first
    }

    int pass = 0, fail = 0;
    for (int i = 0; i < NTESTS; i++) {
        if (!want[i]) continue;
        printf("[%-4s] %s\n", tests[i].name, tests[i].what);
        bool ok = tests[i].fn();
        printf("[%-4s] %s\n", tests[i].name, ok ? "PASS" : "FAIL");
        ok ? pass++ : fail++;
        if (!ok && i < 2) { printf("stopping: the remaining tests need the basics to work\n"); break; }
    }
    printf("\n%d passed, %d failed; %d values checked, %d mismatches\n", pass, fail, checked, mismatches);
    return fail ? 1 : 0;
}
