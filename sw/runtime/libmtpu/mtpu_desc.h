// libmtpu — descriptor and program-header builders (INTERFACE_SPEC.md §3-§5).
//
// Each builder returns one 64 B descriptor with the fields of its opcode set and
// everything else zero; mtpu_d_wait() adds a wait condition (§4.2). Used by the
// bring-up tests; the compiler emits the same bytes from Python (sw/compiler/).
#ifndef MTPU_DESC_H
#define MTPU_DESC_H

#include <stdbool.h>
#include <stdint.h>

#include "mini_tpu_isa.h"

typedef struct { uint8_t b[MINI_TPU_DESC_SIZE]; } mtpu_desc_t;

// VEC_SOFTMAX quantization fields (§5.7)
typedef struct {
    int32_t in_mult, in_zp, out_mult;
    int in_shift, out_shift, out_zp;
} mtpu_smax_q_t;

// little-endian field access
static inline void mtpu_put8(uint8_t *p, uint32_t off, uint32_t v) { p[off] = (uint8_t)v; }
static inline void mtpu_put16(uint8_t *p, uint32_t off, uint32_t v)
{
    p[off] = (uint8_t)v; p[off + 1] = (uint8_t)(v >> 8);
}
static inline void mtpu_put32(uint8_t *p, uint32_t off, uint32_t v)
{
    for (int i = 0; i < 4; i++) p[off + i] = (uint8_t)(v >> (8 * i));
}
static inline int32_t mtpu_get32s(const uint8_t *p, uint32_t off)
{
    return (int32_t)((uint32_t)p[off] | (uint32_t)p[off + 1] << 8 | (uint32_t)p[off + 2] << 16 |
                     (uint32_t)p[off + 3] << 24);
}

// Program header (§3.1). arch = MINI_TPU_ARCH_WORD(N, units, vmem KB).
void mtpu_put_header(uint8_t *h, uint32_t arch, uint8_t regions_mask, uint32_t first_off, unsigned first_len,
                     uint32_t total_bytes, uint32_t program_id);
// A compiled program fits if its magic, compiler hash (MINI_TPU_CONFIG_HASH), arch word (HW_ARCH)
// and program_id match. Otherwise prints why, naming the file, and returns -1.
int mtpu_check_header(const uint8_t *h, const char *name, uint32_t hw_arch, uint32_t program_id);

mtpu_desc_t mtpu_d_new(uint8_t opcode);
mtpu_desc_t mtpu_d_wait(mtpu_desc_t d, unsigned unit, unsigned dist);

// memory ops (§5.1-§5.3)
mtpu_desc_t mtpu_d_load(bool store, unsigned region, uint32_t ddr_off, uint32_t vmem, unsigned rows,
                        unsigned row_bytes, uint32_t ddr_stride, uint32_t vmem_stride);
mtpu_desc_t mtpu_d_load_wgt(unsigned region, uint32_t ddr_off, unsigned half, unsigned k);

// matrix (§5.4, §5.4.1)
mtpu_desc_t mtpu_d_gemm(uint32_t act, unsigned act_pitch, unsigned m_tiles, unsigned k, unsigned half);
mtpu_desc_t mtpu_d_load_wgt_vm(uint32_t src, unsigned src_pitch, unsigned k, unsigned half, bool transpose);

// vector (§5.5-§5.7)
mtpu_desc_t mtpu_d_epilogue(unsigned mode, unsigned flags, int out_zp, int clamp_min, int clamp_max,
                            unsigned rows, unsigned dst_pitch, uint32_t dst, uint32_t params);
mtpu_desc_t mtpu_d_epilogue_partial(mtpu_desc_t d, uint32_t partial, unsigned partial_pitch);   // ACC_* modes
mtpu_desc_t mtpu_d_eltwise(unsigned op, unsigned a_type, unsigned b_type, unsigned out_type, unsigned b_bcast,
                           unsigned rows, unsigned cols, uint32_t a, uint32_t b, uint32_t out, unsigned a_pitch,
                           unsigned b_pitch, unsigned out_pitch, uint32_t params);
mtpu_desc_t mtpu_d_softmax(bool in_i32, unsigned rows, unsigned len, unsigned in_pitch, uint32_t in, uint32_t out,
                           unsigned out_pitch, const mtpu_smax_q_t *q, uint32_t mask, bool has_mask);

// control (§5.8)
mtpu_desc_t mtpu_d_op(uint8_t opcode, uint32_t p0, uint32_t p1);   // p0 at 0x18 (u32), p1 at 0x1C (u16)
mtpu_desc_t mtpu_d_nop(void);
mtpu_desc_t mtpu_d_halt(void);
mtpu_desc_t mtpu_d_fence(void);
mtpu_desc_t mtpu_d_break(unsigned break_id);
mtpu_desc_t mtpu_d_next_chunk(uint32_t offset, unsigned len);
mtpu_desc_t mtpu_d_loop(uint32_t count, unsigned body_len);
mtpu_desc_t mtpu_d_loop_stride(mtpu_desc_t d, unsigned region, int32_t stride);

#endif
