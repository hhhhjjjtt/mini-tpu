// mini-tpu controller firmware — thin descriptor router.
//
// Idle loop: heartbeat, wait for DOORBELL = RUN.
// Program run (INTERFACE_SPEC §3-§6):
//   1. Fetch and validate the 64 B program header.
//   2. Fetch chunk 0; whenever a chunk lands, prefetch the next one into the
//      other bank (read from the chunk's trailing NEXT_CHUNK).
//   3. Runs of unit descriptors go to the hardware issue engine (§8.7), which
//      builds and pushes the 16-word commands (wait targets, DDR address,
//      payload). Firmware handles the control ops (LOOP, NEXT_CHUNK, BREAK,
//      FENCE, HALT), keeps the engine's per-region LOOP offsets current, and
//      reports errors.
//
// RV32I only: no multiply or divide in this file.

#include "mini_tpu_hw.h"

#define FW_ARCH         MINI_TPU_ARCH_WORD(MINI_TPU_TILE_N, 0x0F, MINI_TPU_VMEM_KB)    // built for this config
#define W(off)          ((off) >> 2)    // byte offset -> word index in a descriptor / header

typedef struct {
    uint32_t start, end, remaining;
    int32_t  stride[8];
    int32_t  acc[8];                    // stride sum applied so far by this level
} loop_t;

typedef volatile const uint32_t desc_t;

static uint32_t prog_lo, prog_hi;
static int32_t  loop_add[8];            // per-region DDR offset added by active loops
static loop_t   loops[MINI_TPU_LOOP_MAX_DEPTH];
static uint32_t loop_sp;
static uint32_t chunk_no, desc_count;
static desc_t  *cur_desc;

enum { R_STOP = 0, R_NEXT = 1 };

// -----------------------------------------------------------------------------
// Status / error reporting
// -----------------------------------------------------------------------------
static uint32_t cur_desc_idx(void)
{
    return ((uint32_t)cur_desc & 0x3FFFu) >> 6;
}

static void trace(uint32_t event, uint32_t info)
{
    TM_TRACE_W(0) = TM_CYCLE_LO;
    TM_TRACE_W(1) = (chunk_no & 0xFFFFu) | (cur_desc_idx() << 16);
    TM_TRACE_W(2) = (cur_desc ? (cur_desc[0] & 0xFFu) : 0u) | (event << 8);
    TM_TRACE_W(3) = info;
    TM_TRACE_PUSH = 1;
}

static void fail(uint32_t code, uint32_t unit, uint32_t info0, uint32_t info1)
{
    MB_ERR_DESC   = (chunk_no << 16) | cur_desc_idx();
    MB_ERR_UNIT   = unit;
    MB_ERR_INFO0  = info0;
    MB_ERR_INFO1  = info1;
    MB_DESC_COUNT = desc_count;
    trace(2, code);
    MB_STATUS     = MINI_TPU_ST_ERROR | (code << 8);
}

// Unit error, watchdog, or ABORT pending? Returns nonzero if the run must stop
// (STATUS already set).
static __attribute__((noinline)) int health_slow(uint32_t s)
{
    if (s & SUM_WATCHDOG)
        return 1;                                   // hardware reported it
    if (s & SUM_ERR_MASK) {
        for (uint32_t u = 0; u < MINI_TPU_NUM_UNITS; u++)
            if (s & (1u << u)) {
                fail((UNIT_STATUS(u) >> 8) & 0xFFu, u, 0, 0);
                return 1;
            }
    }
    if (s & SUM_DOORBELL) {
        if (MB_DOORBELL == MINI_TPU_DB_ABORT) {
            fail(MINI_TPU_ERR_ABORTED, MINI_TPU_UNIT_FW, 0, 0);
            return 1;
        }
    }
    return 0;                                       // RUN / RESUME while running: ignored
}

static inline int check_health(void)
{
    uint32_t s = MB_UNIT_SUMMARY;
    if (__builtin_expect((s & (SUM_ERR_MASK | SUM_WATCHDOG | SUM_DOORBELL)) == 0, 1))
        return 0;
    return health_slow(s);
}

// -----------------------------------------------------------------------------
// Fetch DMA
// -----------------------------------------------------------------------------
static void fetch_start(uint32_t bank, uint32_t off, uint32_t bytes)
{
    uint32_t lo = prog_lo + off;
    FETCH_SRC_LO = lo;
    FETCH_SRC_HI = prog_hi + (lo < off);
    FETCH_LEN    = bytes;
    FETCH_BANK   = bank;
    FETCH_START  = 1;
}

static int fetch_wait(void)
{
    uint32_t s;
    while ((s = FETCH_STATUS) & FETCH_BUSY)
        if (check_health())
            return 1;
    FETCH_STATUS = FETCH_DONE | FETCH_AXI_ERR;
    if (s & FETCH_AXI_ERR) {
        fail(MINI_TPU_ERR_FETCH_AXI, MINI_TPU_UNIT_FW, FETCH_SRC_LO, FETCH_SRC_HI);
        return 1;
    }
    return 0;
}

// -----------------------------------------------------------------------------
// Unit issue (hardware issue engine)
// -----------------------------------------------------------------------------
#define ISE_ABANDONED 0xFFFFFFFFu

static inline int is_unit_op(uint32_t op)
{
    return op == MINI_TPU_OP_LOAD || op == MINI_TPU_OP_LOAD_WGT || op == MINI_TPU_OP_STORE || op == MINI_TPU_OP_GEMM || op == MINI_TPU_OP_LOAD_WGT_VM ||
           op == MINI_TPU_OP_VEC_EPILOGUE || op == MINI_TPU_OP_VEC_ELTWISE || op == MINI_TPU_OP_VEC_SOFTMAX ||
           op == MINI_TPU_OP_VEC_LAYERNORM || op == MINI_TPU_OP_VEC_LUT;
}

// Issue up to n descriptors from (bank, i). Returns the engine status, or
// ISE_ABANDONED if the run had to stop (STATUS already set).
static uint32_t ise_run(uint32_t bank, uint32_t i, uint32_t n)
{
    ISE_POS   = (bank << 8) | i;
    ISE_COUNT = n;
    ISE_CTRL  = ISE_GO;
    for (;;) {
        uint32_t s = ISE_STATUS;
        if (!(s & ISE_ST_BUSY))
            return s;
        uint32_t h = MB_UNIT_SUMMARY;
        if (__builtin_expect(h & (SUM_ERR_MASK | SUM_WATCHDOG | SUM_DOORBELL), 0)) {
            ISE_CTRL = ISE_STOP;                    // descriptor buffer is ours again
            while ((s = ISE_STATUS) & ISE_ST_BUSY)
                ;
            if (health_slow(h))
                return ISE_ABANDONED;
            return s;                               // stopped early; caller resumes at pos
        }
    }
}

static int fence(void)
{
    for (uint32_t u = 0; u < MINI_TPU_NUM_UNITS; u++) {
        uint32_t t = ISE_ISSUED(u);
        while (UNIT_DONE(u) < t)
            if (check_health())
                return 1;
    }
    return 0;
}

static void set_loop_add(uint32_t r, int32_t v)
{
    loop_add[r] = v;
    ISE_LOOP_ADD(r) = (uint32_t)v;
}

// -----------------------------------------------------------------------------
// Control ops
// -----------------------------------------------------------------------------
static int do_loop(desc_t *d, uint32_t i, uint32_t len)
{
    uint32_t count = d[W(MINI_TPU_LOOP_COUNT)];
    uint32_t body  = d[W(MINI_TPU_LOOP_BODY_LEN)] & 0xFFFFu;
    if (count == 0 || body == 0) {
        fail(MINI_TPU_ERR_BAD_FIELD, MINI_TPU_UNIT_FW, count == 0 ? MINI_TPU_LOOP_COUNT : MINI_TPU_LOOP_BODY_LEN, 0);
        return 1;
    }
    if (loop_sp == MINI_TPU_LOOP_MAX_DEPTH) {
        fail(MINI_TPU_ERR_LOOP_DEPTH, MINI_TPU_UNIT_FW, 0, 0);
        return 1;
    }
    uint32_t end = i + 1 + body;
    // the body may not contain the chunk terminator or leave an enclosing body
    if (end > len - 1 || (loop_sp && end > loops[loop_sp - 1].end)) {
        fail(MINI_TPU_ERR_LOOP_BODY, MINI_TPU_UNIT_FW, end, len);
        return 1;
    }
    loop_t *L = &loops[loop_sp++];
    L->start = i + 1;
    L->end = end;
    L->remaining = count;
    for (uint32_t r = 0; r < 8; r++) {
        L->stride[r] = (int32_t)d[W(MINI_TPU_LOOP_STRIDE(0)) + r];
        L->acc[r] = 0;
    }
    return 0;
}

static int do_break(desc_t *d)
{
    if (fence())
        return 1;
    MB_DESC_COUNT = desc_count;
    trace(1, d[W(MINI_TPU_BREAK_ID)] & 0xFFFFu);
    MB_STATUS = MINI_TPU_ST_BREAK | ((d[W(MINI_TPU_BREAK_ID)] & 0xFFFFu) << 16);
    for (;;) {
        if (MB_UNIT_SUMMARY & SUM_DOORBELL) {
            uint32_t db = MB_DOORBELL;
            if (db == MINI_TPU_DB_RESUME)
                return 0;                           // hardware set STATUS = RUNNING
            if (db == MINI_TPU_DB_ABORT) {
                fail(MINI_TPU_ERR_ABORTED, MINI_TPU_UNIT_FW, 0, 0);
                return 1;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Chunk interpreter. Returns R_NEXT at NEXT_CHUNK, R_STOP at HALT or error.
// -----------------------------------------------------------------------------
static int run_chunk(uint32_t bank, uint32_t len)
{
    desc_t *base = (desc_t *)DESC_BANK_BASE(bank);
    uint32_t i = 0;

    while (i < len) {
        desc_t *d = base + (i << 4);
        cur_desc = d;
        if (check_health())
            return R_STOP;

        uint32_t w0 = d[0];
        uint32_t op = w0 & 0xFFu;
        if (is_unit_op(op)) {
            // hand the engine everything up to the end of the innermost loop body
            uint32_t limit = loop_sp ? loops[loop_sp - 1].end : len;
            uint32_t s = ise_run(bank, i, limit - i);
            if (s == ISE_ABANDONED)
                return R_STOP;
            i += ISE_N(s);
            desc_count += ISE_N(s);
            uint32_t why = ISE_REASON(s);
            if (why == ISE_R_REGION || why == ISE_R_OVF) {
                cur_desc = base + (i << 4);
                if (why == ISE_R_REGION)
                    fail(MINI_TPU_ERR_BAD_FIELD, MINI_TPU_UNIT_FW, MINI_TPU_D_REGION, cur_desc[W(MINI_TPU_D_REGION)] & 0xFFu);
                else
                    fail(MINI_TPU_ERR_COUNTER_OVERFLOW, MINI_TPU_UNIT_FW, ISE_OP(s), 0);
                return R_STOP;
            }
            // ISE_R_DONE: reached limit. ISE_R_CTRL: next descriptor is not a unit op.
        } else {
            int err;
            desc_count++;
            switch (op) {
            case MINI_TPU_OP_NOP:   err = 0;                  break;
            case MINI_TPU_OP_LOOP:  err = do_loop(d, i, len); break;
            case MINI_TPU_OP_FENCE: err = fence();            break;
            case MINI_TPU_OP_BREAK: err = do_break(d);        break;
            case MINI_TPU_OP_NEXT_CHUNK:
                if (i != len - 1) {
                    fail(MINI_TPU_ERR_CHUNK, MINI_TPU_UNIT_FW, i, len);
                    return R_STOP;
                }
                return R_NEXT;
            case MINI_TPU_OP_HALT:
                if (fence())
                    return R_STOP;
                MB_DESC_COUNT = desc_count;
                MB_STATUS = MINI_TPU_ST_DONE;
                return R_STOP;
            default:
                fail(MINI_TPU_ERR_BAD_OPCODE, MINI_TPU_UNIT_FW, op, 0);
                return R_STOP;
            }
            if (err)
                return R_STOP;
            i++;
        }

        // close finished loop levels (several may end here)
        while (loop_sp && i == loops[loop_sp - 1].end) {
            loop_t *L = &loops[loop_sp - 1];
            if (--L->remaining) {
                for (uint32_t r = 0; r < 8; r++)
                    if (L->stride[r]) {
                        L->acc[r] += L->stride[r];
                        set_loop_add(r, loop_add[r] + L->stride[r]);
                    }
                i = L->start;
                break;
            }
            for (uint32_t r = 0; r < 8; r++)
                if (L->acc[r])
                    set_loop_add(r, loop_add[r] - L->acc[r]);
            loop_sp--;
        }
    }
    fail(MINI_TPU_ERR_CHUNK, MINI_TPU_UNIT_FW, i, len);             // ran off the end
    return R_STOP;
}

// -----------------------------------------------------------------------------
// Program
// -----------------------------------------------------------------------------
static int chunk_len_ok(uint32_t off, uint32_t len)
{
    return len >= 1 && len <= MINI_TPU_CHUNK_MAX_DESCS && (off & 0xFu) == 0;
}

static void run_program(void)
{
    for (uint32_t r = 0; r < 8; r++)
        set_loop_add(r, 0);
    loop_sp = 0;
    chunk_no = 0;
    desc_count = 0;
    cur_desc = 0;
    prog_lo = MB_PROG_BASE_LO;
    prog_hi = MB_PROG_BASE_HI;
    MB_UNIT_CTRL = 1;                               // DONE and issued counters -> 0

    // a prefetch left over from an early HALT may still be running
    while (FETCH_STATUS & FETCH_BUSY)
        ;

    // ---- header ----
    fetch_start(0, 0, MINI_TPU_PROG_HDR_SIZE);
    if (fetch_wait())
        return;
    desc_t *h = (desc_t *)DESC_BANK_BASE(0);
    if (h[W(MINI_TPU_HDR_MAGIC)] != MINI_TPU_PROG_MAGIC) {
        fail(MINI_TPU_ERR_BAD_MAGIC, MINI_TPU_UNIT_FW, h[W(MINI_TPU_HDR_MAGIC)], 0);
        return;
    }
    if ((h[W(MINI_TPU_HDR_FORMAT)] & 0xFFFFu) != MINI_TPU_PROG_FORMAT) {
        fail(MINI_TPU_ERR_VERSION, MINI_TPU_UNIT_FW, h[W(MINI_TPU_HDR_FORMAT)] & 0xFFFFu, 0);
        return;
    }
    uint32_t arch = h[W(MINI_TPU_HDR_ARCH)], hw = MB_HW_ARCH;
    if (MINI_TPU_ARCH_N(arch) != MINI_TPU_ARCH_N(hw) ||
        (MINI_TPU_ARCH_UNITS(arch) & ~MINI_TPU_ARCH_UNITS(hw)) ||
        MINI_TPU_ARCH_VMEM_KB(arch) > MINI_TPU_ARCH_VMEM_KB(hw)) {
        fail(MINI_TPU_ERR_ARCH, MINI_TPU_UNIT_FW, arch, hw);
        return;
    }
    uint32_t missing = 0;
    for (uint32_t r = 0; r < 8; r++)
        if ((h[W(MINI_TPU_HDR_REGIONS)] & (1u << r)) && MB_REGION_LO(r) == 0 && MB_REGION_HI(r) == 0)
            missing |= 1u << r;
    if (missing) {
        fail(MINI_TPU_ERR_REGION_UNSET, MINI_TPU_UNIT_FW, missing, 0);
        return;
    }
    uint32_t off = h[W(MINI_TPU_HDR_FIRST_OFF)], len = h[W(MINI_TPU_HDR_FIRST_LEN)] & 0xFFFFu;
    if (!chunk_len_ok(off, len)) {
        fail(MINI_TPU_ERR_CHUNK, MINI_TPU_UNIT_FW, off, len);
        return;
    }

    // ---- chunks ----
    uint32_t bank = 0;
    fetch_start(bank, off, len << 6);
    if (fetch_wait())
        return;
    for (;;) {
        desc_t *last = (desc_t *)DESC_BANK_BASE(bank) + ((len - 1) << 4);
        uint32_t lop = last[0] & 0xFFu, nlen = 0;
        if (lop == MINI_TPU_OP_NEXT_CHUNK) {
            uint32_t noff = last[W(MINI_TPU_NC_OFFSET)];
            nlen = last[W(MINI_TPU_NC_LEN)] & 0xFFFFu;
            if (!chunk_len_ok(noff, nlen)) {
                cur_desc = last;
                fail(MINI_TPU_ERR_CHUNK, MINI_TPU_UNIT_FW, noff, nlen);
                return;
            }
            fetch_start(bank ^ 1u, noff, nlen << 6);   // prefetch while this chunk runs
        } else if (lop != MINI_TPU_OP_HALT) {
            cur_desc = last;
            fail(MINI_TPU_ERR_CHUNK, MINI_TPU_UNIT_FW, lop, len);
            return;
        }

        if (run_chunk(bank, len) != R_NEXT)
            return;
        if (fetch_wait())
            return;
        bank ^= 1u;
        len = nlen;
        chunk_no++;
    }
}

int main(void)
{
    MB_FW_VERSION = MINI_TPU_FW_VERSION;
    MB_FW_ARCH    = FW_ARCH;

    uint32_t hb = 0;
    for (;;) {
        MB_FW_HEARTBEAT = ++hb;
        if (MB_UNIT_SUMMARY & SUM_DOORBELL)
            if (MB_DOORBELL == MINI_TPU_DB_RUN)
                run_program();
    }
}
