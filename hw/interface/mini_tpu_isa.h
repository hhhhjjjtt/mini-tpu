// mini-tpu — program format and descriptors (INTERFACE_SPEC.md §2-§5, §9).
//
// All offsets are byte offsets, little-endian. Descriptor payload offsets are
// absolute descriptor offsets, as in the spec tables.
#ifndef MINI_TPU_ISA_H
#define MINI_TPU_ISA_H

#include "mini_tpu_config.h"

// ---- Units (§2.1) ----
#define MINI_TPU_UNIT_LOAD      0u              // LOAD, LOAD_WGT
#define MINI_TPU_UNIT_MATRIX    1u              // GEMM, LOAD_WGT_VM
#define MINI_TPU_UNIT_VECTOR    2u              // VEC_*
#define MINI_TPU_UNIT_STORE     3u              // STORE
#define MINI_TPU_NUM_UNITS      4u
#define MINI_TPU_UNIT_FW        0xFFu           // ERR_UNIT for firmware-detected errors

// ---- DDR regions (§2.2) ----
#define MINI_TPU_NUM_REGIONS    8u
#define MINI_TPU_REGION_WEIGHTS 0u
#define MINI_TPU_REGION_PARAMS  1u
#define MINI_TPU_REGION_INPUT   2u
#define MINI_TPU_REGION_OUTPUT  3u
#define MINI_TPU_REGION_SCRATCH 4u
#define MINI_TPU_REGION_KV      5u              // GPT KV cache (compiler use, §2.2)
#define MINI_TPU_REGION_KV_ROW  6u              // KV cache + this token's row; set per token

// ---- Data types (§2.3) ----
#define MINI_TPU_DT_I8          0u
#define MINI_TPU_DT_I32         1u
#define MINI_TPU_DT_BF16        2u

// ---- vmem (§2.4) ----
#define MINI_TPU_VMEM_WORD      16u             // bytes; row starts and pitches are multiples
#define MINI_TPU_VMEM_BANK_BYTES    0x40000u        // 8 banks x 256 KB = 2 MB
#define MINI_TPU_WEIGHT_TILE_N  MINI_TPU_TILE_N    // output columns per weight tile (§2.5)

// ---- Program header (§3.1) ----
#define MINI_TPU_PROG_MAGIC     0x4750544Du     // "MTPG"
#define MINI_TPU_PROG_FORMAT    2u
#define MINI_TPU_PROG_HDR_SIZE  64u
#define MINI_TPU_HDR_MAGIC      0x00u           // u32
#define MINI_TPU_HDR_FORMAT     0x04u           // u16
#define MINI_TPU_HDR_HEADER_SIZE    0x06u           // u16
#define MINI_TPU_HDR_ARCH       0x08u           // u32 arch word: u8 N, u8 units mask, u16 vmem KB
#define MINI_TPU_HDR_ARCH_N     0x08u
#define MINI_TPU_HDR_UNITS      0x09u
#define MINI_TPU_HDR_VMEM_KB    0x0Au
#define MINI_TPU_HDR_REGIONS    0x0Cu           // u8 regions that must be set
#define MINI_TPU_HDR_FIRST_OFF  0x10u           // u32 bytes from PROG_BASE, 16 B aligned
#define MINI_TPU_HDR_FIRST_LEN  0x14u           // u16 descriptors in chunk 0
#define MINI_TPU_HDR_TOTAL_BYTES    0x18u           // u32
#define MINI_TPU_HDR_PROGRAM_ID 0x1Cu           // u32
#define MINI_TPU_HDR_COMPILER_HASH  0x20u           // u64

#define MINI_TPU_DESC_SIZE      64u
#define MINI_TPU_CHUNK_MAX_DESCS    256u            // 16 KB = one descriptor buffer bank

// ---- Descriptor common layout (§4) ----
#define MINI_TPU_D_OPCODE       0x00u           // u8
#define MINI_TPU_D_FLAGS        0x01u           // u8, opcode-specific
#define MINI_TPU_D_WAIT_MASK    0x02u           // u8, bit u = wait on unit u
#define MINI_TPU_D_WAIT_DIST(u) (0x04u + 2u * (u))  // u16 per unit (§4.2)
#define MINI_TPU_D_REGION       0x10u           // u8, memory ops only
#define MINI_TPU_D_DDR_OFFSET   0x14u           // u32, memory ops only
#define MINI_TPU_D_PAYLOAD      0x18u

// ---- Opcodes (§4.1) ----
#define MINI_TPU_OP_NOP         0x00u
#define MINI_TPU_OP_LOAD        0x01u
#define MINI_TPU_OP_LOAD_WGT    0x02u
#define MINI_TPU_OP_STORE       0x03u
#define MINI_TPU_OP_GEMM        0x10u
#define MINI_TPU_OP_LOAD_WGT_VM 0x11u
#define MINI_TPU_OP_VEC_EPILOGUE    0x20u
#define MINI_TPU_OP_VEC_ELTWISE 0x21u
#define MINI_TPU_OP_VEC_SOFTMAX 0x22u
#define MINI_TPU_OP_VEC_LAYERNORM   0x23u
#define MINI_TPU_OP_VEC_LUT     0x24u
#define MINI_TPU_OP_LOOP        0xF0u
#define MINI_TPU_OP_NEXT_CHUNK  0xF1u
#define MINI_TPU_OP_BREAK       0xF2u
#define MINI_TPU_OP_FENCE       0xF3u
#define MINI_TPU_OP_HALT        0xFFu

// ---- LOAD / STORE (§5.1, §5.2) ----
#define MINI_TPU_LD_VMEM_ADDR   0x18u           // u32
#define MINI_TPU_LD_ROWS        0x1Cu           // u16
#define MINI_TPU_LD_ROW_BYTES   0x1Eu           // u16, multiple of 16, <= 4096
#define MINI_TPU_LD_DDR_STRIDE  0x20u           // u32
#define MINI_TPU_LD_VMEM_STRIDE 0x24u           // u32

// ---- LOAD_WGT (§5.3) ----
#define MINI_TPU_LW_HALF        0x18u           // u8
#define MINI_TPU_LW_K           0x1Cu           // u16

// ---- GEMM (§5.4) ----
#define MINI_TPU_GM_ACT_ADDR    0x18u           // u32
#define MINI_TPU_GM_ACT_PITCH   0x1Cu           // u16
#define MINI_TPU_GM_M_TILES     0x1Eu           // u16
#define MINI_TPU_GM_K           0x20u           // u16
#define MINI_TPU_GM_WGT_HALF    0x22u           // u8

// ---- LOAD_WGT_VM (§5.4.1) ----
#define MINI_TPU_WV_SRC_ADDR    0x18u           // u32
#define MINI_TPU_WV_SRC_PITCH   0x1Cu           // u16
#define MINI_TPU_WV_K           0x20u           // u16
#define MINI_TPU_WV_WGT_HALF    0x22u           // u8
#define MINI_TPU_WV_F_TRANSPOSE 0x01u           // flag

// ---- VEC_EPILOGUE (§5.5) ----
#define MINI_TPU_EP_MODE        0x18u           // u8
#define MINI_TPU_EP_OUT_ZP      0x19u           // i8
#define MINI_TPU_EP_CLAMP_MIN   0x1Au           // i8
#define MINI_TPU_EP_CLAMP_MAX   0x1Bu           // i8
#define MINI_TPU_EP_ROWS        0x1Cu           // u16, = 16 x the paired GEMM's m_tiles
#define MINI_TPU_EP_DST_PITCH   0x1Eu           // u16
#define MINI_TPU_EP_DST_ADDR    0x20u           // u32
#define MINI_TPU_EP_PARTIAL_ADDR    0x24u           // u32
#define MINI_TPU_EP_PARTIAL_PITCH   0x28u           // u16
#define MINI_TPU_EP_PARAMS_ADDR 0x2Cu           // u32, requant table (§9.1)
#define MINI_TPU_EP_REQUANT_I8  0u              // modes
#define MINI_TPU_EP_STORE_I32   1u
#define MINI_TPU_EP_ACC_I32     2u
#define MINI_TPU_EP_ACC_REQUANT_I8  3u
#define MINI_TPU_EP_F_PER_TENSOR    0x01u           // flags
#define MINI_TPU_EP_F_DOUBLE_ROUND  0x02u

// ---- VEC_ELTWISE (§5.6) ----
#define MINI_TPU_EW_OP          0x18u           // u8
#define MINI_TPU_EW_TYPES       0x19u           // u8: a_type [3:0], b_type [7:4]
#define MINI_TPU_EW_OUT_TYPE    0x1Au           // u8
#define MINI_TPU_EW_B_BCAST     0x1Bu           // u8
#define MINI_TPU_EW_ROWS        0x1Cu           // u16
#define MINI_TPU_EW_COLS        0x1Eu           // u16, multiple of 16
#define MINI_TPU_EW_A_ADDR      0x20u           // u32
#define MINI_TPU_EW_B_ADDR      0x24u           // u32
#define MINI_TPU_EW_OUT_ADDR    0x28u           // u32
#define MINI_TPU_EW_A_PITCH     0x2Cu           // u16
#define MINI_TPU_EW_B_PITCH     0x2Eu           // u16
#define MINI_TPU_EW_OUT_PITCH   0x30u           // u16
#define MINI_TPU_EW_PARAMS_ADDR 0x34u           // u32, eltwise params (§9.2)
#define MINI_TPU_EW_ADD         0u              // ops
#define MINI_TPU_EW_SUB         1u
#define MINI_TPU_EW_MUL         2u
#define MINI_TPU_EW_MAX         3u
#define MINI_TPU_EW_MIN         4u
#define MINI_TPU_EW_PASS_A      5u
#define MINI_TPU_EW_BCAST_FULL  0u              // b_bcast
#define MINI_TPU_EW_BCAST_ROW   1u
#define MINI_TPU_EW_BCAST_SCALAR    2u

// ---- VEC_SOFTMAX (§5.7) ----
#define MINI_TPU_SM_IN_TYPE     0x18u           // u8, I8 / I32
#define MINI_TPU_SM_OUT_TYPE    0x19u           // u8, I8 only
#define MINI_TPU_SM_ROWS        0x1Au           // u16
#define MINI_TPU_SM_LEN         0x1Cu           // u16, multiple of 16, <= 1024
#define MINI_TPU_SM_IN_PITCH    0x1Eu           // u16
#define MINI_TPU_SM_IN_ADDR     0x20u           // u32
#define MINI_TPU_SM_OUT_ADDR    0x24u           // u32
#define MINI_TPU_SM_OUT_PITCH   0x28u           // u16
#define MINI_TPU_SM_IN_SHIFT    0x2Au           // u8
#define MINI_TPU_SM_OUT_SHIFT   0x2Bu           // u8
#define MINI_TPU_SM_MASK_ADDR   0x2Cu           // u32, one i32 row
#define MINI_TPU_SM_IN_MULT     0x30u           // i32
#define MINI_TPU_SM_IN_ZP       0x34u           // i32
#define MINI_TPU_SM_OUT_MULT    0x38u           // i32
#define MINI_TPU_SM_OUT_ZP      0x3Cu           // i8
#define MINI_TPU_SM_Q_POS       0x3Eu           // u16, CAUSAL: sequence position of row 0
#define MINI_TPU_SM_F_HAS_MASK  0x01u           // flags
#define MINI_TPU_SM_F_CAUSAL    0x02u

// ---- VEC_LAYERNORM (§5.7.1), same positions as VEC_SOFTMAX ----
#define MINI_TPU_LN_IN_TYPE     0x18u           // u8, I8 / I32
#define MINI_TPU_LN_OUT_TYPE    0x19u           // u8, I8 only
#define MINI_TPU_LN_ROWS        0x1Au           // u16
#define MINI_TPU_LN_LEN         0x1Cu           // u16, multiple of 16, <= 1024
#define MINI_TPU_LN_IN_PITCH    0x1Eu           // u16
#define MINI_TPU_LN_IN_ADDR     0x20u           // u32
#define MINI_TPU_LN_OUT_ADDR    0x24u           // u32
#define MINI_TPU_LN_OUT_PITCH   0x28u           // u16
#define MINI_TPU_LN_INV_SHIFT   0x2Au           // u8, 8 ... 63
#define MINI_TPU_LN_C_SHIFT     0x2Bu           // u8, <= 63
#define MINI_TPU_LN_EPS         0x2Cu           // u32, >= 1
#define MINI_TPU_LN_INV_MULT    0x30u           // i32
#define MINI_TPU_LN_C_MULT      0x38u           // i32
#define MINI_TPU_LN_OUT_ZP      0x3Cu           // i8

// ---- VEC_LUT (§5.7.2), same positions as VEC_ELTWISE ----
#define MINI_TPU_LU_ROWS        0x1Cu           // u16
#define MINI_TPU_LU_COLS        0x1Eu           // u16, multiple of 16
#define MINI_TPU_LU_IN_ADDR     0x20u           // u32
#define MINI_TPU_LU_OUT_ADDR    0x28u           // u32
#define MINI_TPU_LU_IN_PITCH    0x2Cu           // u16
#define MINI_TPU_LU_OUT_PITCH   0x30u           // u16
#define MINI_TPU_LU_TABLE_ADDR  0x34u           // u32, 256 B

// ---- Control opcodes (§5.8) ----
#define MINI_TPU_LOOP_COUNT     0x18u           // u32, >= 1
#define MINI_TPU_LOOP_BODY_LEN  0x1Cu           // u16, descriptors
#define MINI_TPU_LOOP_STRIDE(r) (0x20u + 4u * (r))  // i32 per region
#define MINI_TPU_LOOP_MAX_DEPTH 4u
#define MINI_TPU_NC_OFFSET      0x18u           // NEXT_CHUNK: u32 bytes from PROG_BASE
#define MINI_TPU_NC_LEN         0x1Cu           // NEXT_CHUNK: u16 descriptors
#define MINI_TPU_BREAK_ID       0x18u           // u16

// ---- Requant table entry (§9.1), one vmem word per output lane ----
#define MINI_TPU_RQ_ENTRY_SIZE  16u
#define MINI_TPU_RQ_BIAS        0x0u            // i32
#define MINI_TPU_RQ_MULT        0x4u            // i32
#define MINI_TPU_RQ_SHIFT       0x8u            // u8, 2 ... 62

// ---- Eltwise params (§9.2), 64 B ----
#define MINI_TPU_EWP_SIZE       64u
#define MINI_TPU_EWP_A_ZP       0x00u           // i32
#define MINI_TPU_EWP_A_MULT     0x04u           // i32
#define MINI_TPU_EWP_A_SHIFT    0x08u           // u8
#define MINI_TPU_EWP_B_ZP       0x10u
#define MINI_TPU_EWP_B_MULT     0x14u
#define MINI_TPU_EWP_B_SHIFT    0x18u
#define MINI_TPU_EWP_OUT_MULT   0x20u           // i32
#define MINI_TPU_EWP_OUT_SHIFT  0x24u           // u8
#define MINI_TPU_EWP_MUL_SHIFT  0x25u           // u8
#define MINI_TPU_EWP_OUT_ZP     0x30u           // i8
#define MINI_TPU_EWP_CLAMP_MIN  0x31u           // i8
#define MINI_TPU_EWP_CLAMP_MAX  0x32u           // i8
#define MINI_TPU_EWP_FLAGS      0x33u           // u8, bit 0 DOUBLE_ROUND

#endif
