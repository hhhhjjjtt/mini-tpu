// libmtpu — descriptor and program-header builders (see mtpu_desc.h).

#include "mtpu_desc.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

void mtpu_put_header(uint8_t *h, uint32_t arch, uint8_t regions_mask, uint32_t first_off, unsigned first_len,
                     uint32_t total_bytes, uint32_t program_id)
{
    memset(h, 0, MINI_TPU_PROG_HDR_SIZE);
    mtpu_put32(h, MINI_TPU_HDR_MAGIC, MINI_TPU_PROG_MAGIC);
    mtpu_put16(h, MINI_TPU_HDR_FORMAT, MINI_TPU_PROG_FORMAT);
    mtpu_put16(h, MINI_TPU_HDR_HEADER_SIZE, MINI_TPU_PROG_HDR_SIZE);
    mtpu_put32(h, MINI_TPU_HDR_ARCH, arch);
    mtpu_put8(h, MINI_TPU_HDR_REGIONS, regions_mask);
    mtpu_put32(h, MINI_TPU_HDR_FIRST_OFF, first_off);
    mtpu_put16(h, MINI_TPU_HDR_FIRST_LEN, first_len);
    mtpu_put32(h, MINI_TPU_HDR_TOTAL_BYTES, total_bytes);
    mtpu_put32(h, MINI_TPU_HDR_PROGRAM_ID, program_id);
}

int mtpu_check_header(const uint8_t *h, const char *name, uint32_t hw_arch, uint32_t program_id)
{
    uint32_t magic = (uint32_t)mtpu_get32s(h, MINI_TPU_HDR_MAGIC), arch = (uint32_t)mtpu_get32s(h, MINI_TPU_HDR_ARCH);
    uint32_t id = (uint32_t)mtpu_get32s(h, MINI_TPU_HDR_PROGRAM_ID);
    uint64_t hash = (uint64_t)(uint32_t)mtpu_get32s(h, MINI_TPU_HDR_COMPILER_HASH + 4) << 32 |
                    (uint32_t)mtpu_get32s(h, MINI_TPU_HDR_COMPILER_HASH);
    if (magic != MINI_TPU_PROG_MAGIC) {
        fprintf(stderr, "%s: bad magic 0x%08x\n", name, magic);
        return -1;
    }
    if (hash != MINI_TPU_CONFIG_HASH) {
        fprintf(stderr, "%s: program built for config 0x%016" PRIx64 ", runtime for 0x%016llx: rebuild with make dist\n",
                name, hash, (unsigned long long)MINI_TPU_CONFIG_HASH);
        return -1;
    }
    if (arch != hw_arch) {
        fprintf(stderr, "%s: program arch 0x%08x, bitstream HW_ARCH 0x%08x: wrong bitstream?\n", name, arch, hw_arch);
        return -1;
    }
    if (id != program_id) {
        fprintf(stderr, "%s: program_id %u, expected %u\n", name, id, program_id);
        return -1;
    }
    return 0;
}

mtpu_desc_t mtpu_d_new(uint8_t opcode)
{
    mtpu_desc_t d;
    memset(&d, 0, sizeof d);
    d.b[MINI_TPU_D_OPCODE] = opcode;
    return d;
}

mtpu_desc_t mtpu_d_wait(mtpu_desc_t d, unsigned unit, unsigned dist)
{
    d.b[MINI_TPU_D_WAIT_MASK] |= (uint8_t)(1u << unit);
    mtpu_put16(d.b, MINI_TPU_D_WAIT_DIST(unit), dist);
    return d;
}

// -----------------------------------------------------------------------------
// Memory ops
// -----------------------------------------------------------------------------
mtpu_desc_t mtpu_d_load(bool store, unsigned region, uint32_t ddr_off, uint32_t vmem, unsigned rows,
                        unsigned row_bytes, uint32_t ddr_stride, uint32_t vmem_stride)
{
    mtpu_desc_t d = mtpu_d_new(store ? MINI_TPU_OP_STORE : MINI_TPU_OP_LOAD);
    mtpu_put8(d.b, MINI_TPU_D_REGION, region);
    mtpu_put32(d.b, MINI_TPU_D_DDR_OFFSET, ddr_off);
    mtpu_put32(d.b, MINI_TPU_LD_VMEM_ADDR, vmem);
    mtpu_put16(d.b, MINI_TPU_LD_ROWS, rows);
    mtpu_put16(d.b, MINI_TPU_LD_ROW_BYTES, row_bytes);
    mtpu_put32(d.b, MINI_TPU_LD_DDR_STRIDE, ddr_stride);
    mtpu_put32(d.b, MINI_TPU_LD_VMEM_STRIDE, vmem_stride);
    return d;
}

mtpu_desc_t mtpu_d_load_wgt(unsigned region, uint32_t ddr_off, unsigned half, unsigned k)
{
    mtpu_desc_t d = mtpu_d_new(MINI_TPU_OP_LOAD_WGT);
    mtpu_put8(d.b, MINI_TPU_D_REGION, region);
    mtpu_put32(d.b, MINI_TPU_D_DDR_OFFSET, ddr_off);
    mtpu_put8(d.b, MINI_TPU_LW_HALF, half);
    mtpu_put16(d.b, MINI_TPU_LW_K, k);
    return d;
}

// -----------------------------------------------------------------------------
// Matrix
// -----------------------------------------------------------------------------
mtpu_desc_t mtpu_d_gemm(uint32_t act, unsigned act_pitch, unsigned m_tiles, unsigned k, unsigned half)
{
    mtpu_desc_t d = mtpu_d_new(MINI_TPU_OP_GEMM);
    mtpu_put32(d.b, MINI_TPU_GM_ACT_ADDR, act);
    mtpu_put16(d.b, MINI_TPU_GM_ACT_PITCH, act_pitch);
    mtpu_put16(d.b, MINI_TPU_GM_M_TILES, m_tiles);
    mtpu_put16(d.b, MINI_TPU_GM_K, k);
    mtpu_put8(d.b, MINI_TPU_GM_WGT_HALF, half);
    return d;
}

mtpu_desc_t mtpu_d_load_wgt_vm(uint32_t src, unsigned src_pitch, unsigned k, unsigned half, bool transpose)
{
    mtpu_desc_t d = mtpu_d_new(MINI_TPU_OP_LOAD_WGT_VM);
    d.b[MINI_TPU_D_FLAGS] = transpose ? MINI_TPU_WV_F_TRANSPOSE : 0;
    mtpu_put32(d.b, MINI_TPU_WV_SRC_ADDR, src);
    mtpu_put16(d.b, MINI_TPU_WV_SRC_PITCH, src_pitch);
    mtpu_put16(d.b, MINI_TPU_WV_K, k);
    mtpu_put8(d.b, MINI_TPU_WV_WGT_HALF, half);
    return d;
}

// -----------------------------------------------------------------------------
// Vector
// -----------------------------------------------------------------------------
mtpu_desc_t mtpu_d_epilogue(unsigned mode, unsigned flags, int out_zp, int clamp_min, int clamp_max,
                            unsigned rows, unsigned dst_pitch, uint32_t dst, uint32_t params)
{
    mtpu_desc_t d = mtpu_d_new(MINI_TPU_OP_VEC_EPILOGUE);
    d.b[MINI_TPU_D_FLAGS] = (uint8_t)flags;
    mtpu_put8(d.b, MINI_TPU_EP_MODE, mode);
    mtpu_put8(d.b, MINI_TPU_EP_OUT_ZP, (uint32_t)out_zp);
    mtpu_put8(d.b, MINI_TPU_EP_CLAMP_MIN, (uint32_t)clamp_min);
    mtpu_put8(d.b, MINI_TPU_EP_CLAMP_MAX, (uint32_t)clamp_max);
    mtpu_put16(d.b, MINI_TPU_EP_ROWS, rows);
    mtpu_put16(d.b, MINI_TPU_EP_DST_PITCH, dst_pitch);
    mtpu_put32(d.b, MINI_TPU_EP_DST_ADDR, dst);
    mtpu_put32(d.b, MINI_TPU_EP_PARAMS_ADDR, params);
    return d;
}

mtpu_desc_t mtpu_d_epilogue_partial(mtpu_desc_t d, uint32_t partial, unsigned partial_pitch)
{
    mtpu_put32(d.b, MINI_TPU_EP_PARTIAL_ADDR, partial);
    mtpu_put16(d.b, MINI_TPU_EP_PARTIAL_PITCH, partial_pitch);
    return d;
}

mtpu_desc_t mtpu_d_eltwise(unsigned op, unsigned a_type, unsigned b_type, unsigned out_type, unsigned b_bcast,
                           unsigned rows, unsigned cols, uint32_t a, uint32_t b, uint32_t out, unsigned a_pitch,
                           unsigned b_pitch, unsigned out_pitch, uint32_t params)
{
    mtpu_desc_t d = mtpu_d_new(MINI_TPU_OP_VEC_ELTWISE);
    mtpu_put8(d.b, MINI_TPU_EW_OP, op);
    mtpu_put8(d.b, MINI_TPU_EW_TYPES, (a_type & 0xFu) | (b_type & 0xFu) << 4);
    mtpu_put8(d.b, MINI_TPU_EW_OUT_TYPE, out_type);
    mtpu_put8(d.b, MINI_TPU_EW_B_BCAST, b_bcast);
    mtpu_put16(d.b, MINI_TPU_EW_ROWS, rows);
    mtpu_put16(d.b, MINI_TPU_EW_COLS, cols);
    mtpu_put32(d.b, MINI_TPU_EW_A_ADDR, a);
    mtpu_put32(d.b, MINI_TPU_EW_B_ADDR, b);
    mtpu_put32(d.b, MINI_TPU_EW_OUT_ADDR, out);
    mtpu_put16(d.b, MINI_TPU_EW_A_PITCH, a_pitch);
    mtpu_put16(d.b, MINI_TPU_EW_B_PITCH, b_pitch);
    mtpu_put16(d.b, MINI_TPU_EW_OUT_PITCH, out_pitch);
    mtpu_put32(d.b, MINI_TPU_EW_PARAMS_ADDR, params);
    return d;
}

mtpu_desc_t mtpu_d_softmax(bool in_i32, unsigned rows, unsigned len, unsigned in_pitch, uint32_t in, uint32_t out,
                           unsigned out_pitch, const mtpu_smax_q_t *q, uint32_t mask, bool has_mask)
{
    mtpu_desc_t d = mtpu_d_new(MINI_TPU_OP_VEC_SOFTMAX);
    d.b[MINI_TPU_D_FLAGS] = has_mask ? MINI_TPU_SM_F_HAS_MASK : 0;
    mtpu_put8(d.b, MINI_TPU_SM_IN_TYPE, in_i32 ? MINI_TPU_DT_I32 : MINI_TPU_DT_I8);
    mtpu_put8(d.b, MINI_TPU_SM_OUT_TYPE, MINI_TPU_DT_I8);
    mtpu_put16(d.b, MINI_TPU_SM_ROWS, rows);
    mtpu_put16(d.b, MINI_TPU_SM_LEN, len);
    mtpu_put16(d.b, MINI_TPU_SM_IN_PITCH, in_pitch);
    mtpu_put32(d.b, MINI_TPU_SM_IN_ADDR, in);
    mtpu_put32(d.b, MINI_TPU_SM_OUT_ADDR, out);
    mtpu_put16(d.b, MINI_TPU_SM_OUT_PITCH, out_pitch);
    mtpu_put8(d.b, MINI_TPU_SM_IN_SHIFT, (uint32_t)q->in_shift);
    mtpu_put8(d.b, MINI_TPU_SM_OUT_SHIFT, (uint32_t)q->out_shift);
    mtpu_put32(d.b, MINI_TPU_SM_MASK_ADDR, mask);
    mtpu_put32(d.b, MINI_TPU_SM_IN_MULT, (uint32_t)q->in_mult);
    mtpu_put32(d.b, MINI_TPU_SM_IN_ZP, (uint32_t)q->in_zp);
    mtpu_put32(d.b, MINI_TPU_SM_OUT_MULT, (uint32_t)q->out_mult);
    mtpu_put8(d.b, MINI_TPU_SM_OUT_ZP, (uint32_t)q->out_zp);
    return d;
}

// -----------------------------------------------------------------------------
// Control
// -----------------------------------------------------------------------------
mtpu_desc_t mtpu_d_op(uint8_t opcode, uint32_t p0, uint32_t p1)
{
    mtpu_desc_t d = mtpu_d_new(opcode);
    mtpu_put32(d.b, MINI_TPU_D_PAYLOAD, p0);
    mtpu_put16(d.b, MINI_TPU_D_PAYLOAD + 4, p1);
    return d;
}

mtpu_desc_t mtpu_d_nop(void) { return mtpu_d_new(MINI_TPU_OP_NOP); }
mtpu_desc_t mtpu_d_halt(void) { return mtpu_d_new(MINI_TPU_OP_HALT); }
mtpu_desc_t mtpu_d_fence(void) { return mtpu_d_new(MINI_TPU_OP_FENCE); }

mtpu_desc_t mtpu_d_break(unsigned break_id)
{
    mtpu_desc_t d = mtpu_d_new(MINI_TPU_OP_BREAK);
    mtpu_put16(d.b, MINI_TPU_BREAK_ID, break_id);
    return d;
}

mtpu_desc_t mtpu_d_next_chunk(uint32_t offset, unsigned len)
{
    mtpu_desc_t d = mtpu_d_new(MINI_TPU_OP_NEXT_CHUNK);
    mtpu_put32(d.b, MINI_TPU_NC_OFFSET, offset);
    mtpu_put16(d.b, MINI_TPU_NC_LEN, len);
    return d;
}

mtpu_desc_t mtpu_d_loop(uint32_t count, unsigned body_len)
{
    mtpu_desc_t d = mtpu_d_new(MINI_TPU_OP_LOOP);
    mtpu_put32(d.b, MINI_TPU_LOOP_COUNT, count);
    mtpu_put16(d.b, MINI_TPU_LOOP_BODY_LEN, body_len);
    return d;
}

mtpu_desc_t mtpu_d_loop_stride(mtpu_desc_t d, unsigned region, int32_t stride)
{
    mtpu_put32(d.b, MINI_TPU_LOOP_STRIDE(region), (uint32_t)stride);
    return d;
}
