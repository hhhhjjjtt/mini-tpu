`timescale 1ns/1ps

// =============================================================================
// vector_unit — mini-tpu vector unit (VEC_EPILOGUE, VEC_ELTWISE, VEC_SOFTMAX,
//               VEC_LAYERNORM, VEC_LUT)
// =============================================================================
//
// INTERFACE_SPEC.md §5.5 (VEC_EPILOGUE), §5.6 (VEC_ELTWISE), §5.7
// (VEC_SOFTMAX), §5.7.1 (VEC_LAYERNORM), §5.7.2 (VEC_LUT), §9 (params).
//
// LANES lanes, a power of two >= 4 (16 by default). One vmem read port and one
// vmem write port (128-bit words). Each step is sequential: read words -> compute
// LANES lanes -> write words. A block is LANES elements: LANES/4 words of i32,
// LANES/16 words of i8. Below 16 lanes an i8 block is a slice of a word: the word is
// read with its first block and written with its last.
//
//   VEC_EPILOGUE, per output row r, in N/LANES passes of LANES lanes (one pass of N
//   lanes if N <= LANES):
//     take one N x i32 row from the matrix stream (check tlast placement)
//     [ACC_*]  read the i32 partial block
//     compute  REQUANT_I8 / STORE_I32 / ACC_I32 / ACC_REQUANT_I8
//     write    the i8 or i32 block
//
//   VEC_ELTWISE, per row r, per block cb:
//     read A, read B per broadcast mode
//     compute  rescale A, rescale B, op, output rescale + clamp
//     write    the block
//
//   VEC_SOFTMAX, per row r (fixed point, base-2 domain; §5.7):
//     pass 1, per block: read input (+ mask), t = apply_scale(in - zp,
//       in_mult, in_shift) (+ mask) through the pipeline, t = -2^31 where CAUSAL
//       blocks the column, block max, e = 2^(t - max) (exp2neg), running sum rescaled
//       when the max grows
//     R = floor(2^46 / sum) (bit-serial divider, 47 cycles)
//     pass 2, per block: re-read, recompute t and e, then the pipeline's PASS_A
//       path: p = apply_scale(e, R, 30) (Q16 probability), out = clamp(
//       apply_scale(p, out_mult, out_shift) + out_zp) -> i8
//
//   VEC_LAYERNORM, per row (no gamma / beta; §5.7.1):
//     pass 1, per block: x = saturated input, x * x through the pipeline
//       multipliers, block sums into S1 and S2
//     D = n * S2 - S1^2 + eps (shift-add for n * S2; lane 0 squares S1, lane 1 forms
//       the Q8 mean); shift D left in pairs of bits until normalized; q = sqrt(top
//       32 bits), bit-serial; R = floor(2^46 / 2q) on the softmax divider
//     pass 2, per block: u = apply_scale((x << 8) - mean, R, 45 - pairs shifted), then
//       the softmax output stage -> i8
//
//   VEC_LUT, per command: read the 256 B table, write it into the lane RAMs (256
//     cycles); per block: read, look up LANES bytes, write (§5.7.2)
//
// Arithmetic is bit-exact with TOSA apply_scale as implemented in
// runtime/flexnpu_module.cc (apply_scale_tosa). Softmax is bit-exact with
// the reference model in sim/tb/tb_vector_path.sv.
//
// =============================================================================

module vector_unit #(
    parameter integer N     = 16,           // matrix stream width, a power of two >= 16
    parameter integer LANES = 16,
    parameter integer VM_AW = 17,
    parameter integer CMD_W = 512,
    parameter integer MAX_LEN = 1024        // longest softmax / layernorm row
)(
    input  wire                 clk,
    input  wire                 rst_n,

    // ---- Command (from unit_frontend) ----
    input  wire                 cmd_valid,
    input  wire [CMD_W-1:0]     cmd,
    output wire                 cmd_ready,
    output reg                  unit_done,
    output reg                  err,
    output reg  [7:0]           err_code,
    output wire                 busy,

    // ---- Matrix result stream ----
    input  wire                 s_valid,
    input  wire [N*32-1:0]      s_data,
    input  wire                 s_last,
    output wire                 s_ready,

    // ---- vmem read port ----
    output wire                 vm_rd_req,
    output wire [VM_AW-1:0]     vm_rd_addr,
    input  wire                 vm_rd_gnt,
    input  wire                 vm_rd_rvalid,
    input  wire [127:0]         vm_rd_rdata,

    // ---- vmem write port ----
    output wire                 vm_wr_req,
    output wire [VM_AW-1:0]     vm_wr_addr,
    output wire [127:0]         vm_wr_data,
    input  wire                 vm_wr_gnt
);

    // ------------------------------------------------------------
    // Constants
    // ------------------------------------------------------------
    localparam integer LB    = $clog2(LANES);
    localparam integer W32   = LANES / 4;                       // words of an i32 block
    localparam integer W8    = (LANES < 16) ? 1 : LANES / 16;   // words of an i8 block
    localparam integer SUB   = (LANES < 16) ? 16 / LANES : 1;   // i8 blocks per word
    localparam integer SB    = $clog2(SUB);
    localparam integer ALIGN = (LANES < 16) ? 4 : LB;           // row lengths: multiples of 2^ALIGN
    localparam integer NBW   = 16 - LB;                         // block index
    localparam integer BUF_W = LANES * 32;

    // VEC_EPILOGUE: EP passes of EL lanes per stream row
    localparam integer EL    = (N < LANES) ? N : LANES;
    localparam integer EP    = N / EL;
    localparam integer EW32  = EL / 4;
    localparam integer EW8   = (EL < 16) ? 1 : EL / 16;

    localparam integer RNW   = $clog2(N + 1);                   // read burst: up to N params
    localparam integer WNW   = $clog2(W32 + 1);                 // write burst
    localparam integer WIW   = (W32 <= 1) ? 1 : $clog2(W32);
    localparam integer ESW   = 17 + LB;                         // block sum of e
    localparam integer LG    = LANES / 4;                       // layernorm sum groups
    localparam integer LN_NB = ($clog2(MAX_LEN / LANES + 1) > 7) ? $clog2(MAX_LEN / LANES + 1) : 7;

    initial begin
        if (LANES < 4 || (LANES & (LANES - 1)) != 0)
            $error("LANES must be a power of two >= 4: an i32 block fills whole vmem words");
        if (N < 16 || (N & (N - 1)) != 0)
            $error("N must be a power of two >= 16");
        if (MAX_LEN % (1 << ALIGN) != 0)
            $error("MAX_LEN must be a multiple of 16 and of LANES");
    end

    localparam [7:0] OP_EPILOGUE = 8'h20;
    localparam [7:0] OP_ELTWISE  = 8'h21;
    localparam [7:0] OP_SOFTMAX  = 8'h22;
    localparam [7:0] OP_LAYERNORM = 8'h23;
    localparam [7:0] OP_LUT      = 8'h24;

    localparam [7:0] ERR_BAD_OPCODE      = 8'h10;
    localparam [7:0] ERR_BAD_FIELD       = 8'h11;
    localparam [7:0] ERR_STREAM_MISMATCH = 8'h23;

    localparam [1:0] M_REQUANT_I8     = 2'd0;
    localparam [1:0] M_STORE_I32      = 2'd1;
    localparam [1:0] M_ACC_I32        = 2'd2;
    localparam [1:0] M_ACC_REQUANT_I8 = 2'd3;

    localparam [2:0] E_ADD = 3'd0, E_SUB = 3'd1, E_MUL = 3'd2,
                     E_MAX = 3'd3, E_MIN = 3'd4, E_PASS_A = 3'd5;

    localparam [1:0] B_FULL = 2'd0, B_ROW = 2'd1, B_SCALAR = 2'd2;

    localparam [4:0] S_IDLE     = 5'd0,
                     S_PARAM    = 5'd1,
                     S_EPI_ROW  = 5'd2,
                     S_EPI_PART = 5'd3,
                     S_EPI_CALC = 5'd4,
                     S_ELT_RDA  = 5'd5,
                     S_ELT_RDB  = 5'd6,
                     S_ELT_CALC = 5'd7,
                     S_WR       = 5'd8,
                     S_SM_RDA   = 5'd9,      // read input block
                     S_SM_RDM   = 5'd10,     // read mask block
                     S_SM_T     = 5'd11,     // pipeline: t = scaled input (+ mask)
                     S_SM_MAX   = 5'd12,     // pass 1: block max
                     S_SM_EXP   = 5'd13,     // pass 1: e = 2^(t - max), rescale factor
                     S_SM_SUM1  = 5'd14,     // pass 1: block sum, sum * factor
                     S_SM_SUM2  = 5'd15,     // pass 1: accumulate
                     S_SM_DIV   = 5'd16,     // R = floor(2^46 / sum)
                     S_SM_E2    = 5'd17,     // pass 2: e = 2^(t - max)
                     S_SM_P     = 5'd18,     // pass 2: pipeline p, output scale
                     S_LN_RD    = 5'd19,     // read input block
                     S_LN_T     = 5'd20,     // pipeline: x * x, or the output in pass 2
                     S_LN_ACC   = 5'd21,     // pass 1: accumulate S1, S2
                     S_LN_R0    = 5'd22,     // row: start n * S2
                     S_LN_MUL   = 5'd23,     // row: n * S2, shift-add
                     S_LN_D     = 5'd24,     // row: D, mean
                     S_LN_NORM  = 5'd25,     // row: normalize D
                     S_LN_SQRT  = 5'd26,     // row: q = sqrt(D)
                     S_LUT_FILL = 5'd27,     // table -> lane RAMs
                     S_LUT_RD   = 5'd28,     // read input block
                     S_LUT_C    = 5'd29;     // take the lookups

    localparam [1:0] LN_SUM = 2'd0, LN_ROW = 2'd1, LN_OUT = 2'd2;   // what the lanes compute

    // ------------------------------------------------------------
    // Command decode (descriptor byte o >= 0x18 at cmd bit 192 + 8(o - 0x18))
    // ------------------------------------------------------------
    wire [7:0]  c_opcode = cmd[7:0];
    wire [7:0]  c_flags  = cmd[15:8];

    // VEC_EPILOGUE
    wire [7:0]  ce_mode     = cmd[192 +: 8];   // 0x18
    wire [7:0]  ce_out_zp   = cmd[200 +: 8];   // 0x19
    wire [7:0]  ce_cmin     = cmd[208 +: 8];   // 0x1A
    wire [7:0]  ce_cmax     = cmd[216 +: 8];   // 0x1B
    wire [15:0] ce_rows     = cmd[224 +: 16];  // 0x1C
    wire [15:0] ce_dpitch   = cmd[240 +: 16];  // 0x1E
    wire [31:0] ce_dst      = cmd[256 +: 32];  // 0x20
    wire [31:0] ce_part     = cmd[288 +: 32];  // 0x24
    wire [15:0] ce_ppitch   = cmd[320 +: 16];  // 0x28
    wire [31:0] ce_params   = cmd[352 +: 32];  // 0x2C

    // VEC_ELTWISE
    wire [7:0]  cw_op       = cmd[192 +: 8];   // 0x18
    wire [3:0]  cw_a_type   = cmd[200 +: 4];   // 0x19 [3:0]
    wire [3:0]  cw_b_type   = cmd[204 +: 4];   // 0x19 [7:4]
    wire [7:0]  cw_out_type = cmd[208 +: 8];   // 0x1A
    wire [7:0]  cw_bcast    = cmd[216 +: 8];   // 0x1B
    wire [15:0] cw_rows     = cmd[224 +: 16];  // 0x1C
    wire [15:0] cw_cols     = cmd[240 +: 16];  // 0x1E
    wire [31:0] cw_a        = cmd[256 +: 32];  // 0x20
    wire [31:0] cw_b        = cmd[288 +: 32];  // 0x24
    wire [31:0] cw_o        = cmd[320 +: 32];  // 0x28
    wire [15:0] cw_apitch   = cmd[352 +: 16];  // 0x2C
    wire [15:0] cw_bpitch   = cmd[368 +: 16];  // 0x2E
    wire [15:0] cw_opitch   = cmd[384 +: 16];  // 0x30
    wire [31:0] cw_params   = cmd[416 +: 32];  // 0x34

    // VEC_SOFTMAX
    wire [7:0]  cs_in_type  = cmd[192 +: 8];   // 0x18
    wire [7:0]  cs_out_type = cmd[200 +: 8];   // 0x19
    wire [15:0] cs_rows     = cmd[208 +: 16];  // 0x1A
    wire [15:0] cs_len      = cmd[224 +: 16];  // 0x1C
    wire [15:0] cs_ipitch   = cmd[240 +: 16];  // 0x1E
    wire [31:0] cs_in       = cmd[256 +: 32];  // 0x20
    wire [31:0] cs_out      = cmd[288 +: 32];  // 0x24
    wire [15:0] cs_opitch   = cmd[320 +: 16];  // 0x28
    wire [7:0]  cs_in_sh    = cmd[336 +: 8];   // 0x2A
    wire [7:0]  cs_out_sh   = cmd[344 +: 8];   // 0x2B
    wire [31:0] cs_mask     = cmd[352 +: 32];  // 0x2C
    wire [31:0] cs_in_mult  = cmd[384 +: 32];  // 0x30
    wire [31:0] cs_in_zp    = cmd[416 +: 32];  // 0x34
    wire [31:0] cs_out_mult = cmd[448 +: 32];  // 0x38
    wire [7:0]  cs_out_zp   = cmd[480 +: 8];   // 0x3C
    wire [15:0] cs_q_pos    = cmd[496 +: 16];  // 0x3E

    // VEC_LAYERNORM decodes from the VEC_SOFTMAX positions (in_shift = inv_shift,
    // out_shift = c_shift, mask = eps, in_mult = inv_mult, out_mult = c_mult),
    // VEC_LUT from the VEC_ELTWISE ones (params = table).
    function automatic ok_addr(input [31:0] a);
        ok_addr = (a[3:0] == 0) && (a[31:VM_AW+4] == 0);
    endfunction

    wire ce_acc = (ce_mode == M_ACC_I32) || (ce_mode == M_ACC_REQUANT_I8);
    wire ce_bad = (ce_mode > 3) || (ce_rows == 0) || (ce_dpitch[3:0] != 0) ||
                  !ok_addr(ce_dst) || !ok_addr(ce_params) ||
                  (ce_acc && (!ok_addr(ce_part) || ce_ppitch[3:0] != 0));

    wire cw_bad = (cw_op > 5) || (cw_a_type > 1) || (cw_out_type > 1) ||
                  (cw_op != E_PASS_A && (cw_b_type > 1 || cw_bcast > 2)) ||
                  (cw_rows == 0) || (cw_cols == 0) || (cw_cols[ALIGN-1:0] != 0) ||
                  !ok_addr(cw_a) || !ok_addr(cw_o) || !ok_addr(cw_params) ||
                  (cw_op != E_PASS_A && !ok_addr(cw_b)) ||
                  (cw_apitch[3:0] != 0) || (cw_bpitch[3:0] != 0) || (cw_opitch[3:0] != 0);

    wire cs_bad = (cs_in_type > 1) || (cs_out_type != 0) || (cs_rows == 0) ||
                  (cs_len == 0) || (cs_len[ALIGN-1:0] != 0) || (cs_len > MAX_LEN) ||
                  (cs_ipitch[3:0] != 0) || (cs_opitch[3:0] != 0) ||
                  !ok_addr(cs_in) || !ok_addr(cs_out) || (c_flags[0] && !ok_addr(cs_mask)) ||
                  (cs_in_sh > 8'd63) || (cs_out_sh > 8'd63);

    wire cl_bad = (cs_in_type > 1) || (cs_out_type != 0) || (cs_rows == 0) ||
                  (cs_len == 0) || (cs_len[ALIGN-1:0] != 0) || (cs_len > MAX_LEN) ||
                  (cs_ipitch[3:0] != 0) || (cs_opitch[3:0] != 0) ||
                  !ok_addr(cs_in) || !ok_addr(cs_out) || (cs_mask == 0) ||
                  (cs_in_sh < 8'd8) || (cs_in_sh > 8'd63) || (cs_out_sh > 8'd63);

    wire cu_bad = (cw_rows == 0) || (cw_cols == 0) || (cw_cols[ALIGN-1:0] != 0) ||
                  !ok_addr(cw_a) || !ok_addr(cw_o) || !ok_addr(cw_params) ||
                  (cw_apitch[3:0] != 0) || (cw_opitch[3:0] != 0);

    // ------------------------------------------------------------
    // Latched command state
    // ------------------------------------------------------------
    reg [4:0]        state;
    reg              is_epi, is_ln, is_lut;
    // VEC_SOFTMAX
    reg              is_sm, sm_ph, sm_pass, sm_mask_en, sm_causal;
    reg [16:0]       sm_limit;                  // CAUSAL: last visible column of this row
    reg signed [31:0] sm_in_mult, sm_in_zp, sm_out_mult;
    reg [7:0]        sm_in_shift, sm_out_shift;
    reg signed [7:0] sm_out_zp;
    reg [VM_AW-1:0]  mask_base_w;
    reg              f_per_tensor, f_double_round;
    reg [1:0]        mode;
    reg [7:0]        out_zp, cmin, cmax;
    reg [2:0]        eop;
    reg              a_i32, b_i32, o_i32;
    reg [1:0]        bcast;
    reg [15:0]       rows, r;
    reg [NBW-1:0]    nblk, cb;
    reg [VM_AW-1:0]  a_pitch_w, b_pitch_w, o_pitch_w, p_pitch_w;
    reg [VM_AW-1:0]  b_base_w;
    reg [VM_AW-1:0]  row_a_w, row_b_w, row_o_w, row_p_w;

    assign cmd_ready = (state == S_IDLE);
    assign busy      = (state != S_IDLE);
    assign s_ready   = (state == S_EPI_ROW);

    // ------------------------------------------------------------
    // Buffers
    // ------------------------------------------------------------
    reg [127:0]        pbuf [0:N-1];  // params (N >= 16 words, the LUT table's size)
    reg [BUF_W-1:0]    abuf;          // A operand or i32 partial
    reg [BUF_W-1:0]    bbuf;          // B operand
    reg [BUF_W-1:0]    wbuf;          // results to write
    reg [N*32-1:0]     srow;          // matrix stream row

    // block addressing
    wire [4:0]       sub_off  = (cb & (SUB - 1)) * LANES;       // i8: first byte in the word
    wire             sub_last = (cb & (SUB - 1)) == SUB - 1;
    wire [NBW:0]     cb_next  = cb + 1'b1;

    function automatic [VM_AW-1:0] blk_off(input [NBW:0] b, input is_i32);
        blk_off = is_i32 ? b * W32 : (b * W8) >> SB;
    endfunction

    // words to read for block b: a slice of an i8 word comes with the word's first block
    function automatic [RNW-1:0] blk_n(input [NBW:0] b, input is_i32);
        blk_n = is_i32 ? W32 : ((b & (SUB - 1)) == 0) ? W8 : 0;
    endfunction

    // ------------------------------------------------------------
    // Read engine
    // ------------------------------------------------------------
    reg [VM_AW-1:0] rd_base;
    reg [RNW-1:0]   rd_n, rd_iss, rd_rcv;
    reg [1:0]       rd_sel;           // 0 pbuf, 1 abuf, 2 bbuf
    wire            rd_state = (state == S_PARAM) || (state == S_EPI_PART) ||
                               (state == S_ELT_RDA) || (state == S_ELT_RDB) ||
                               (state == S_SM_RDA) || (state == S_SM_RDM) ||
                               (state == S_LN_RD) || (state == S_LUT_RD);
    wire            rd_done  = rd_state && (rd_rcv == rd_n);

    assign vm_rd_req  = rd_state && (rd_iss < rd_n);
    assign vm_rd_addr = rd_base + rd_iss;

    // ------------------------------------------------------------
    // Write engine
    // ------------------------------------------------------------
    reg [VM_AW-1:0] wr_base;
    reg [WNW-1:0]   wr_n, wr_cnt;
    wire            wr_done = (state == S_WR) && (wr_cnt == wr_n);

    assign vm_wr_req  = (state == S_WR) && (wr_cnt < wr_n);
    assign vm_wr_addr = wr_base + wr_cnt;
    wire [WIW-1:0]  wr_i      = wr_cnt[WIW-1:0];
    assign vm_wr_data = wbuf[wr_i*128 +: 128];

    // ------------------------------------------------------------
    // Arithmetic helpers (bit-exact TOSA apply_scale is built in the pipeline below)
    // ------------------------------------------------------------
    function automatic [7:0] clamp8(input signed [31:0] v,
                                    input signed [7:0] lo,
                                    input signed [7:0] hi);
        begin
            if (v < lo)      clamp8 = lo;
            else if (v > hi) clamp8 = hi;
            else             clamp8 = v[7:0];
        end
    endfunction

    function automatic signed [31:0] lane_of(input [BUF_W-1:0] w, input is_i32,
                                             input integer j, input integer off);
        begin
            if (!is_i32)
                lane_of = {{24{w[(off+j)*8+7]}}, w[(off+j)*8 +: 8]};
            else
                lane_of = w[j*32 +: 32];
        end
    endfunction

    // ------------------------------------------------------------
    // Lane compute: feed-forward pipeline
    // ------------------------------------------------------------
    // The operand buffers (srow, abuf, bbuf, pbuf) and the command fields are
    // stable for the whole CALC state, so the pipeline needs no valid bits:
    // CALC waits LAT_EPI / LAT_ELT cycles and then takes the tap.
    //
    //   S1   operands: EPI x = c (+ partial) + bias;  ELT xa = a - a_zp, xb = b - b_zp
    //        rounding terms (depend on the operand sign for DOUBLE_ROUND)
    //   S2-3 xa * ma, xb * b_mult         (32 x 32, two register stages for DSP M/P)
    //   S4   (p + rnd) >>> sh             -> va, vb
    //   S5   EPI tap: c (i32 modes) or clamp(va + out_zp)
    //        ELT op: ADD/SUB/MAX/MIN/PASS -> r; MUL: va * vb (M stage)
    //   S6   MUL P stage
    //   S7   MUL: (p + rnd) >>> mul_sh    (apply_scale with dr = 0, same formula)
    //   S8-9 r * o_mult, rounding term
    //   S10  ELT tap: r (i32 out) or clamp(scaled + o_zp)
    localparam [3:0] LAT_EPI = 4'd5, LAT_ELT = 4'd10;

    function automatic signed [31:0] sat20(input signed [31:0] v);
        sat20 = (v > 32'sd1048575) ? 32'sd1048575 : (v < -32'sd1048575) ? -32'sd1048575 : v;
    endfunction

    function automatic signed [63:0] rnd_term(input signed [31:0] v, input [7:0] sh, input dr);
        reg signed [63:0] r;
        begin
            r = (sh == 0) ? 64'sd0 : (64'sd1 <<< (sh - 1));
            if (dr && sh > 31)
                r = r + (v[31] ? -64'sd1073741824 : 64'sd1073741824);
            rnd_term = r;
        end
    endfunction

    function automatic signed [31:0] shift_round(input signed [63:0] p, input signed [63:0] rnd,
                                                 input [7:0] sh);
        reg signed [63:0] t;
        begin
            t = (p + rnd) >>> sh;
            shift_round = t[31:0];
        end
    endfunction

    // Eltwise params (§9.2)
    wire signed [31:0] p_a_zp   = pbuf[0][31:0];
    wire signed [31:0] p_a_mult = pbuf[0][63:32];
    wire [7:0]         p_a_sh   = pbuf[0][71:64];
    wire signed [31:0] p_b_zp   = pbuf[1][31:0];
    wire signed [31:0] p_b_mult = pbuf[1][63:32];
    wire [7:0]         p_b_sh   = pbuf[1][71:64];
    wire signed [31:0] p_o_mult = pbuf[2][31:0];
    wire [7:0]         p_o_sh   = pbuf[2][39:32];
    wire [7:0]         p_mul_sh = pbuf[2][47:40];
    wire signed [7:0]  p_o_zp   = pbuf[3][7:0];
    wire signed [7:0]  p_cmin   = pbuf[3][15:8];
    wire signed [7:0]  p_cmax   = pbuf[3][23:16];
    wire               p_dr     = pbuf[3][24];

    // output-scale parameters: eltwise params block, or the softmax command
    wire               cmd_out  = is_sm || is_ln;
    wire signed [31:0] o_mult_w = cmd_out ? sm_out_mult  : p_o_mult;
    wire [7:0]         o_sh_w   = cmd_out ? sm_out_shift : p_o_sh;
    wire signed [7:0]  o_zp_w   = cmd_out ? sm_out_zp    : p_o_zp;
    wire signed [7:0]  o_lo_w   = cmd_out ? -8'sd128     : p_cmin;
    wire signed [7:0]  o_hi_w   = cmd_out ? 8'sd127      : p_cmax;
    wire               o_dr_w   = cmd_out ? 1'b0         : p_dr;

    // softmax lane state (see the softmax section below)
    reg signed [31:0] tq   [0:LANES-1];       // t (+ mask) of the current block
    reg [16:0]        ereg [0:LANES-1];       // e = 2^(t - max), Q0.16
    reg [30:0]        sm_R;                   // floor(2^46 / sum)

    // layernorm state (see the layernorm section below)
    reg [1:0]         ln_st;
    reg signed [31:0] ln_s1, ln_mean;
    reg [7:0]         ln_a;                   // pass 2 shift

    wire epi_i32 = (mode == M_STORE_I32) || (mode == M_ACC_I32);
    wire epi_acc = (mode == M_ACC_I32) || (mode == M_ACC_REQUANT_I8);

    // per-lane pipeline registers
    reg signed [31:0] xa1 [0:LANES-1], xb1 [0:LANES-1], ma1 [0:LANES-1], c1 [0:LANES-1];
    reg [7:0]         sa1 [0:LANES-1];
    reg signed [63:0] ra1 [0:LANES-1], rb1 [0:LANES-1];
    reg signed [63:0] pa2 [0:LANES-1], pb2 [0:LANES-1], pa3 [0:LANES-1], pb3 [0:LANES-1];
    reg [7:0]         sa2 [0:LANES-1], sa3 [0:LANES-1];
    reg signed [63:0] ra2 [0:LANES-1], rb2 [0:LANES-1], ra3 [0:LANES-1], rb3 [0:LANES-1];
    reg signed [31:0] c2 [0:LANES-1], c3 [0:LANES-1], c4 [0:LANES-1];
    reg signed [31:0] va4 [0:LANES-1], vb4 [0:LANES-1];
    reg signed [31:0] epi5 [0:LANES-1];
    reg signed [31:0] r5 [0:LANES-1], r6 [0:LANES-1], r7 [0:LANES-1], r8 [0:LANES-1], r9 [0:LANES-1];
    reg signed [63:0] pm5 [0:LANES-1], pm6 [0:LANES-1];
    reg signed [63:0] po8 [0:LANES-1], po9 [0:LANES-1], ro8 [0:LANES-1], ro9 [0:LANES-1];
    reg signed [31:0] elt10 [0:LANES-1];

    integer j, e, ej;
    reg signed [31:0] t_c, t_x, t_b, t_r, t_s;

    wire [31:0] ep_off = (cb & (EP - 1)) * EL;                  // epilogue pass: first lane

    always @(posedge clk) begin
        for (j = 0; j < LANES; j = j + 1) begin
            // ---- S1 ----
            ej = (j < EL) ? ep_off + j : 0;
            e = f_per_tensor ? 0 : ej;
            t_c = srow[ej*32 +: 32];
            if (epi_acc)
                t_c = t_c + lane_of(abuf, 1'b1, j, 0);
            if (is_epi) begin
                t_x     = t_c + $signed(pbuf[e][31:0]);
                ma1[j] <= pbuf[e][63:32];
                sa1[j] <= pbuf[e][71:64];
                ra1[j] <= rnd_term(t_x, pbuf[e][71:64], f_double_round);
            end else if (is_sm && sm_ph) begin                  // softmax pass 2: p = e * R >> 30
                t_x     = {15'd0, ereg[j]};
                ma1[j] <= {1'b0, sm_R};
                sa1[j] <= 8'd30;
                ra1[j] <= rnd_term(t_x, 8'd30, 1'b0);
            end else if (is_sm) begin                           // softmax: t = (in - zp) scaled
                t_x     = lane_of(abuf, a_i32, j, sub_off) - sm_in_zp;
                ma1[j] <= sm_in_mult;
                sa1[j] <= sm_in_shift;
                ra1[j] <= rnd_term(t_x, sm_in_shift, 1'b0);
            end else if (is_ln) begin
                t_s = sat20(lane_of(abuf, a_i32, j, sub_off));
                case (ln_st)
                    LN_SUM: begin                               // x * x
                        t_x     = t_s;
                        ma1[j] <= t_s;
                        sa1[j] <= 8'd0;
                        ra1[j] <= 64'sd0;
                    end
                    LN_ROW: begin                               // lane 0: S1^2, lane 1: Q8 mean
                        t_x     = ln_s1;
                        ma1[j] <= (j == 0) ? ln_s1 : sm_in_mult;
                        sa1[j] <= (j == 0) ? 8'd0 : sm_in_shift - 8'd8;
                        ra1[j] <= (j == 0) ? 64'sd0 : rnd_term(ln_s1, sm_in_shift - 8'd8, 1'b0);
                    end
                    default: begin                              // u = (x << 8) - mean, scaled
                        t_x     = (t_s <<< 8) - ln_mean;
                        ma1[j] <= {1'b0, sm_R};
                        sa1[j] <= ln_a;
                        ra1[j] <= rnd_term(t_x, ln_a, 1'b0);
                    end
                endcase
            end else begin
                t_x     = lane_of(abuf, a_i32, j, sub_off) - p_a_zp;
                ma1[j] <= p_a_mult;
                sa1[j] <= p_a_sh;
                ra1[j] <= rnd_term(t_x, p_a_sh, p_dr);
            end
            t_b = ((bcast == B_SCALAR) ? lane_of(bbuf, b_i32, 0, 0) : lane_of(bbuf, b_i32, j, sub_off)) - p_b_zp;
            xa1[j] <= t_x;
            xb1[j] <= t_b;
            rb1[j] <= rnd_term(t_b, p_b_sh, p_dr);
            c1[j]  <= t_c;

            // ---- S2, S3: multiplies ----
            pa2[j] <= xa1[j] * ma1[j];
            pb2[j] <= xb1[j] * p_b_mult;
            sa2[j] <= sa1[j];  ra2[j] <= ra1[j];  rb2[j] <= rb1[j];  c2[j] <= c1[j];
            pa3[j] <= pa2[j];
            pb3[j] <= pb2[j];
            sa3[j] <= sa2[j];  ra3[j] <= ra2[j];  rb3[j] <= rb2[j];  c3[j] <= c2[j];

            // ---- S4: round + shift ----
            va4[j] <= shift_round(pa3[j], ra3[j], sa3[j]);
            vb4[j] <= shift_round(pb3[j], rb3[j], p_b_sh);
            c4[j]  <= c3[j];

            // ---- S5: EPI tap / ELT op ----
            t_r = va4[j] + $signed(out_zp);
            epi5[j] <= epi_i32 ? c4[j] : {{24{1'b0}}, clamp8(t_r, cmin, cmax)};
            case (eop)
                E_ADD:   r5[j] <= va4[j] + vb4[j];
                E_SUB:   r5[j] <= va4[j] - vb4[j];
                E_MAX:   r5[j] <= (va4[j] > vb4[j]) ? va4[j] : vb4[j];
                E_MIN:   r5[j] <= (va4[j] < vb4[j]) ? va4[j] : vb4[j];
                default: r5[j] <= va4[j];                      // PASS_A (MUL overrides in S7)
            endcase
            pm5[j] <= va4[j] * vb4[j];

            // ---- S6, S7: MUL rounding ----
            pm6[j] <= pm5[j];
            r6[j]  <= r5[j];
            r7[j]  <= (eop == E_MUL) ? shift_round(pm6[j], rnd_term(32'sd0, p_mul_sh, 1'b0), p_mul_sh)
                                     : r6[j];

            // ---- S8, S9: output scale ----
            po8[j] <= r7[j] * o_mult_w;
            ro8[j] <= rnd_term(r7[j], o_sh_w, o_dr_w);
            r8[j]  <= r7[j];
            po9[j] <= po8[j];
            ro9[j] <= ro8[j];
            r9[j]  <= r8[j];

            // ---- S10: ELT tap ----
            t_r = shift_round(po9[j], ro9[j], o_sh_w) + o_zp_w;
            elt10[j] <= o_i32 ? r9[j] : {{24{1'b0}}, clamp8(t_r, o_lo_w, o_hi_w)};
        end
    end

    // pack the tap into write words
    wire out_i32 = is_epi ? epi_i32 : o_i32;
    reg [BUF_W-1:0] calc_w;
    always @(*) begin
        calc_w = (SUB > 1) ? wbuf : {BUF_W{1'b0}};      // i8 slices build up the word
        for (j = 0; j < LANES; j = j + 1) begin
            if (!is_epi || j < EL) begin
                if (out_i32)
                    calc_w[j*32 +: 32] = is_epi ? epi5[j] : elt10[j];
                else
                    calc_w[(sub_off+j)*8 +: 8] = is_epi ? epi5[j][7:0] : elt10[j][7:0];
            end
        end
    end

    reg [3:0] calc_cnt;

    // ------------------------------------------------------------
    // Softmax datapath
    // ------------------------------------------------------------
    localparam [3:0] LAT_T = 4'd4;              // pipeline S1..S4 -> va4

    reg signed [31:0] sm_max, m_new;
    reg [31:0]        sm_sum;                   // Q0.16 sum of e, < 2^27
    reg [16:0]        sm_factor;                // 2^(old max - new max)
    reg [48:0]        sm_prod;
    reg [ESW-1:0]     sm_bsum;
    reg [27:0]        dv_rem;
    reg [30:0]        dv_q;
    reg [5:0]         dv_i;

    // t of lane j: saturating t + mask
    function automatic signed [31:0] sat_add(input signed [31:0] a, input signed [31:0] b);
        reg signed [32:0] t;
        begin
            t = {a[31], a} + {b[31], b};
            if (t > 33'sd2147483647)       sat_add = 32'sh7FFFFFFF;
            else if (t < -33'sd2147483648) sat_add = 32'sh80000000;
            else                           sat_add = t[31:0];
        end
    endfunction

    // block max of t and block sum of e: balanced trees, node i over nodes 2i+1 and 2i+2
    reg signed [31:0] mx_t [0:2*LANES-2];
    reg [ESW-1:0]     sm_t [0:2*LANES-2];
    integer           ti;
    always @(*) begin
        for (ti = 0; ti < LANES; ti = ti + 1) begin
            mx_t[LANES-1+ti] = tq[ti];
            sm_t[LANES-1+ti] = ereg[ti];
        end
        for (ti = LANES - 2; ti >= 0; ti = ti - 1) begin
            mx_t[ti] = (mx_t[2*ti+1] > mx_t[2*ti+2]) ? mx_t[2*ti+1] : mx_t[2*ti+2];
            sm_t[ti] = sm_t[2*ti+1] + sm_t[2*ti+2];
        end
    end
    wire signed [31:0] bmax = mx_t[0];
    wire [ESW-1:0]     esum = sm_t[0];

    // e = 2^(t - m): one ROM per lane (pass 1 uses the new max, pass 2 the row max)
    wire signed [31:0] m_ref = sm_pass ? sm_max : m_new;
    wire [16:0]        e_lane [0:LANES-1];
    genvar gl;
    generate
        for (gl = 0; gl < LANES; gl = gl + 1) begin : g_exp
            wire signed [32:0] zl = {m_ref[31], m_ref} - {tq[gl][31], tq[gl]};
            exp2neg u_exp (.z(zl[32:0]), .e(e_lane[gl]));
        end
    endgenerate

    // running-sum rescale factor 2^(old max - new max)
    wire signed [32:0] z_fac = {m_new[31], m_new} - {sm_max[31], sm_max};
    wire [16:0]        e_fac;
    exp2neg u_exp_fac (.z(z_fac[32:0]), .e(e_fac));

    // divider step: 2^46 / sum, one quotient bit per cycle
    wire [28:0] dv_rem2 = {dv_rem, (dv_i == 6'd46)};
    wire        dv_ge   = (dv_rem2 >= {1'b0, sm_sum[27:0]});

    // ------------------------------------------------------------
    // Layernorm datapath
    // ------------------------------------------------------------
    reg signed [22:0] ln_x1 [0:LG-1];           // block sums of x, level 1
    reg [43:0]        ln_q1 [0:LG-1];           // block sums of x * x, level 1
    reg [63:0]        ln_s2, ln_d, ln_m;
    reg [31:0]        ln_eps;
    reg [$clog2(LN_NB)-1:0] ln_b;
    reg [4:0]         ln_c;                     // pairs of bits D was shifted left

    // q = floor(sqrt(sq_d)), one result bit per cycle
    reg [31:0]  sq_d;
    reg [19:0]  sq_rem;
    reg [15:0]  sq_root;
    reg [3:0]   sq_i;
    wire [19:0] sq_r2 = {sq_rem[17:0], sq_d[31:30]};
    wire [19:0] sq_t  = {2'b00, sq_root, 2'b01};
    wire        sq_ge = (sq_r2 >= sq_t);
    wire [15:0] sq_next = {sq_root[14:0], sq_ge};

    // S1, S2 with the level-1 sums of this block added
    reg signed [31:0] ln_s1_n;
    reg [63:0]        ln_s2_n;
    integer           gi;
    always @(*) begin
        ln_s1_n = ln_s1;
        ln_s2_n = ln_s2;
        for (gi = 0; gi < LG; gi = gi + 1) begin
            ln_s1_n = ln_s1_n + ln_x1[gi];
            ln_s2_n = ln_s2_n + ln_q1[gi];
        end
    end

    // ------------------------------------------------------------
    // LUT datapath: one table copy per lane
    // ------------------------------------------------------------
    reg  [7:0]  lut_i;
    wire        lut_we = (state == S_LUT_FILL);
    wire [7:0]  lut_wd = pbuf[lut_i[7:4]][lut_i[3:0]*8 +: 8];
    wire [7:0]  lut_q [0:LANES-1];
    generate
        for (gl = 0; gl < LANES; gl = gl + 1) begin : g_lut
            lut_ram u_lut (.clk(clk), .we(lut_we), .wa(lut_i), .wd(lut_wd),
                           .ra(abuf[(sub_off+gl)*8 +: 8]), .rd(lut_q[gl]));
        end
    endgenerate

    // ------------------------------------------------------------
    // Sequencer
    // ------------------------------------------------------------
    task automatic start_read(input [VM_AW-1:0] base, input [RNW-1:0] n,
                              input [1:0] sel, input [4:0] st);
        begin
            rd_base <= base;
            rd_n    <= n;
            rd_iss  <= 0;
            rd_rcv  <= 0;
            rd_sel  <= sel;
            state   <= st;
        end
    endtask

    task automatic fail(input [7:0] code);
        begin
            err      <= 1'b1;
            err_code <= code;
            state    <= S_IDLE;
        end
    endtask

    wire [RNW-1:0]   n_a0   = blk_n(0, a_i32);                  // first block of a row
    wire [RNW-1:0]   n_a1   = blk_n(cb_next, a_i32);
    wire [VM_AW-1:0] a_next = row_a_w + blk_off(cb_next, a_i32);
    wire [4:0]       elt_rd = is_lut ? S_LUT_RD : S_ELT_RDA;

    // B block address by broadcast mode
    wire [VM_AW-1:0] b_blk_w = (bcast == B_SCALAR) ? b_base_w :
                               (bcast == B_ROW)    ? b_base_w + blk_off(cb, b_i32) :
                                                     row_b_w  + blk_off(cb, b_i32);

    // result block: an i8 word goes out with its last slice
    wire [WNW-1:0]   wr_blk = out_i32 ? (is_epi ? EW32 : W32) :
                              sub_last ? (is_epi ? EW8 : W8) : 0;

    task automatic write_out;
        begin
            wr_base <= row_o_w + blk_off(cb, out_i32);
            wr_n    <= wr_blk;
            wr_cnt  <= 0;
            state   <= S_WR;
        end
    endtask

    integer k;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            unit_done <= 1'b0;
            err       <= 1'b0;
            err_code  <= 8'h00;
            rd_n      <= 0;
            rd_iss    <= 0;
            rd_rcv    <= 0;
            wr_n      <= 0;
            wr_cnt    <= 0;
            is_epi    <= 1'b0;
            is_sm     <= 1'b0;
            is_ln     <= 1'b0;
            is_lut    <= 1'b0;
            sm_ph     <= 1'b0;
            sm_pass   <= 1'b0;
            calc_cnt  <= 0;
        end else begin
            unit_done <= 1'b0;
            if (state != S_EPI_CALC && state != S_ELT_CALC && state != S_SM_T && state != S_SM_P && state != S_LN_T)
                calc_cnt <= 0;

            // read engine bookkeeping
            if (vm_rd_req && vm_rd_gnt)
                rd_iss <= rd_iss + 1'b1;
            if (rd_state && vm_rd_rvalid) begin
                case (rd_sel)
                    2'd0:    pbuf[rd_rcv[$clog2(N)-1:0]]    <= vm_rd_rdata;
                    2'd1:    abuf[rd_rcv[WIW-1:0]*128 +: 128] <= vm_rd_rdata;
                    default: bbuf[rd_rcv[WIW-1:0]*128 +: 128] <= vm_rd_rdata;
                endcase
                rd_rcv <= rd_rcv + 1'b1;
            end
            // write engine bookkeeping
            if (vm_wr_req && vm_wr_gnt)
                wr_cnt <= wr_cnt + 1'b1;

            case (state)
            // ----------------------------------------------------
            S_IDLE: begin
                if (cmd_valid) begin
                    if (c_opcode == OP_EPILOGUE) begin
                        if (ce_bad) fail(ERR_BAD_FIELD);
                        else begin
                            is_epi         <= 1'b1;
                            is_sm          <= 1'b0;
                            is_ln          <= 1'b0;
                            is_lut         <= 1'b0;
                            f_per_tensor   <= c_flags[0];
                            f_double_round <= c_flags[1];
                            mode      <= ce_mode[1:0];
                            out_zp    <= ce_out_zp;
                            cmin      <= ce_cmin;
                            cmax      <= ce_cmax;
                            rows      <= ce_rows;
                            r         <= 0;
                            o_pitch_w <= ce_dpitch[15:4];
                            p_pitch_w <= ce_ppitch[15:4];
                            row_o_w   <= ce_dst[VM_AW+3:4];
                            row_p_w   <= ce_part[VM_AW+3:4];
                            nblk      <= EP;
                            cb        <= 0;
                            start_read(ce_params[VM_AW+3:4], c_flags[0] ? 1 : N,
                                       2'd0, S_PARAM);
                        end
                    end else if (c_opcode == OP_ELTWISE) begin
                        if (cw_bad) fail(ERR_BAD_FIELD);
                        else begin
                            is_epi    <= 1'b0;
                            is_sm     <= 1'b0;
                            is_ln     <= 1'b0;
                            is_lut    <= 1'b0;
                            eop       <= cw_op[2:0];
                            a_i32     <= cw_a_type[0];
                            b_i32     <= cw_b_type[0];
                            o_i32     <= cw_out_type[0];
                            bcast     <= cw_bcast[1:0];
                            rows      <= cw_rows;
                            r         <= 0;
                            nblk      <= cw_cols[15:LB];
                            cb        <= 0;
                            a_pitch_w <= cw_apitch[15:4];
                            b_pitch_w <= cw_bpitch[15:4];
                            o_pitch_w <= cw_opitch[15:4];
                            row_a_w   <= cw_a[VM_AW+3:4];
                            b_base_w  <= cw_b[VM_AW+3:4];
                            row_b_w   <= cw_b[VM_AW+3:4];
                            row_o_w   <= cw_o[VM_AW+3:4];
                            start_read(cw_params[VM_AW+3:4], 4, 2'd0, S_PARAM);
                        end
                    end else if (c_opcode == OP_SOFTMAX) begin
                        if (cs_bad) fail(ERR_BAD_FIELD);
                        else begin
                            is_epi       <= 1'b0;
                            is_sm        <= 1'b1;
                            is_ln        <= 1'b0;
                            is_lut       <= 1'b0;
                            sm_ph        <= 1'b0;
                            sm_pass      <= 1'b0;
                            sm_mask_en   <= c_flags[0];
                            sm_causal    <= c_flags[1];
                            sm_limit     <= {1'b0, cs_q_pos};
                            sm_in_mult   <= cs_in_mult;
                            sm_in_zp     <= cs_in_zp;
                            sm_in_shift  <= cs_in_sh;
                            sm_out_mult  <= cs_out_mult;
                            sm_out_shift <= cs_out_sh;
                            sm_out_zp    <= cs_out_zp;
                            sm_max       <= 32'sh80000000;
                            sm_sum       <= 32'd0;
                            eop          <= E_PASS_A;
                            a_i32        <= cs_in_type[0];
                            o_i32        <= 1'b0;
                            rows         <= cs_rows;
                            r            <= 0;
                            nblk         <= cs_len[15:LB];
                            cb           <= 0;
                            a_pitch_w    <= cs_ipitch[15:4];
                            o_pitch_w    <= cs_opitch[15:4];
                            row_a_w      <= cs_in[VM_AW+3:4];
                            row_o_w      <= cs_out[VM_AW+3:4];
                            mask_base_w  <= cs_mask[VM_AW+3:4];
                            start_read(cs_in[VM_AW+3:4], cs_in_type[0] ? W32 : W8, 2'd1, S_SM_RDA);
                        end
                    end else if (c_opcode == OP_LAYERNORM) begin
                        if (cl_bad) fail(ERR_BAD_FIELD);
                        else begin
                            is_epi       <= 1'b0;
                            is_sm        <= 1'b0;
                            is_ln        <= 1'b1;
                            is_lut       <= 1'b0;
                            ln_st        <= LN_SUM;
                            ln_s1        <= 32'sd0;
                            ln_s2        <= 64'd0;
                            ln_eps       <= cs_mask;
                            sm_in_mult   <= cs_in_mult;
                            sm_in_shift  <= cs_in_sh;
                            sm_out_mult  <= cs_out_mult;
                            sm_out_shift <= cs_out_sh;
                            sm_out_zp    <= cs_out_zp;
                            eop          <= E_PASS_A;
                            a_i32        <= cs_in_type[0];
                            o_i32        <= 1'b0;
                            rows         <= cs_rows;
                            r            <= 0;
                            nblk         <= cs_len[15:LB];
                            cb           <= 0;
                            a_pitch_w    <= cs_ipitch[15:4];
                            o_pitch_w    <= cs_opitch[15:4];
                            row_a_w      <= cs_in[VM_AW+3:4];
                            row_o_w      <= cs_out[VM_AW+3:4];
                            start_read(cs_in[VM_AW+3:4], cs_in_type[0] ? W32 : W8, 2'd1, S_LN_RD);
                        end
                    end else if (c_opcode == OP_LUT) begin
                        if (cu_bad) fail(ERR_BAD_FIELD);
                        else begin
                            is_epi    <= 1'b0;
                            is_sm     <= 1'b0;
                            is_ln     <= 1'b0;
                            is_lut    <= 1'b1;
                            a_i32     <= 1'b0;
                            o_i32     <= 1'b0;
                            rows      <= cw_rows;
                            r         <= 0;
                            nblk      <= cw_cols[15:LB];
                            cb        <= 0;
                            a_pitch_w <= cw_apitch[15:4];
                            o_pitch_w <= cw_opitch[15:4];
                            row_a_w   <= cw_a[VM_AW+3:4];
                            row_o_w   <= cw_o[VM_AW+3:4];
                            start_read(cw_params[VM_AW+3:4], 16, 2'd0, S_PARAM);
                        end
                    end else begin
                        fail(ERR_BAD_OPCODE);
                    end
                end
            end

            // ----------------------------------------------------
            S_PARAM: begin
                if (rd_done) begin
                    if (is_epi) state <= S_EPI_ROW;
                    else if (is_lut) begin
                        lut_i <= 8'd0;
                        state <= S_LUT_FILL;
                    end else    start_read(row_a_w, n_a0, 2'd1, S_ELT_RDA);
                end
            end

            // ---------------- VEC_EPILOGUE ----------------
            S_EPI_ROW: begin
                if (s_valid) begin
                    srow <= s_data;
                    if (s_last != (r == rows - 1))
                        fail(ERR_STREAM_MISMATCH);
                    else if (mode == M_ACC_I32 || mode == M_ACC_REQUANT_I8)
                        start_read(row_p_w, EW32, 2'd1, S_EPI_PART);
                    else
                        state <= S_EPI_CALC;
                end
            end

            S_EPI_PART: begin
                if (rd_done) state <= S_EPI_CALC;
            end

            S_EPI_CALC: begin
                calc_cnt <= calc_cnt + 1'b1;
                if (calc_cnt == LAT_EPI) begin
                    wbuf <= calc_w;
                    write_out;
                end
            end

            // ---------------- VEC_ELTWISE ----------------
            S_ELT_RDA: begin
                if (rd_done) begin
                    if (eop == E_PASS_A) state <= S_ELT_CALC;
                    else start_read(b_blk_w, (bcast == B_SCALAR) ? 1 : blk_n(cb, b_i32),
                                    2'd2, S_ELT_RDB);
                end
            end

            S_ELT_RDB: begin
                if (rd_done) state <= S_ELT_CALC;
            end

            S_ELT_CALC: begin
                calc_cnt <= calc_cnt + 1'b1;
                if (calc_cnt == LAT_ELT) begin
                    wbuf <= calc_w;
                    write_out;
                end
            end

            // ---------------- VEC_SOFTMAX ----------------
            S_SM_RDA: begin
                if (rd_done) begin
                    if (sm_mask_en) start_read(mask_base_w + blk_off(cb, 1'b1), W32, 2'd2, S_SM_RDM);
                    else            state <= S_SM_T;
                end
            end

            S_SM_RDM: begin
                if (rd_done) state <= S_SM_T;
            end

            S_SM_T: begin
                calc_cnt <= calc_cnt + 1'b1;
                if (calc_cnt == LAT_T) begin
                    for (k = 0; k < LANES; k = k + 1)
                        tq[k] <= (sm_causal && {1'b0, cb, k[LB-1:0]} > sm_limit)
                                 ? 32'sh80000000
                                 : sat_add(va4[k], sm_mask_en ? lane_of(bbuf, 1'b1, k, 0) : 32'sd0);
                    state <= sm_pass ? S_SM_E2 : S_SM_MAX;
                end
            end

            S_SM_MAX: begin
                m_new <= (bmax > sm_max) ? bmax : sm_max;
                state <= S_SM_EXP;
            end

            S_SM_EXP: begin
                for (k = 0; k < LANES; k = k + 1)
                    ereg[k] <= e_lane[k];
                sm_factor <= e_fac;
                state     <= S_SM_SUM1;
            end

            S_SM_SUM1: begin
                sm_bsum <= esum;
                sm_prod <= sm_sum * sm_factor;
                state   <= S_SM_SUM2;
            end

            S_SM_SUM2: begin
                sm_sum <= sm_prod[47:16] + sm_bsum;
                sm_max <= m_new;
                if (cb == nblk - 1) begin
                    dv_rem <= 0;
                    dv_q   <= 0;
                    dv_i   <= 6'd46;
                    state  <= S_SM_DIV;
                end else begin
                    cb <= cb + 1'b1;
                    start_read(a_next, n_a1, 2'd1, S_SM_RDA);
                end
            end

            S_SM_DIV: begin
                dv_rem <= dv_ge ? (dv_rem2 - {1'b0, sm_sum[27:0]}) : dv_rem2[27:0];
                dv_q   <= {dv_q[29:0], dv_ge};
                dv_i   <= dv_i - 1'b1;
                if (dv_i == 0) begin
                    sm_R    <= {dv_q[29:0], dv_ge};
                    sm_pass <= 1'b1;
                    cb      <= 0;
                    if (is_ln) begin
                        ln_st <= LN_OUT;
                        ln_a  <= 8'd45 - {3'd0, ln_c};
                        start_read(row_a_w, n_a0, 2'd1, S_LN_RD);
                    end else
                        start_read(row_a_w, n_a0, 2'd1, S_SM_RDA);
                end
            end

            S_SM_E2: begin
                for (k = 0; k < LANES; k = k + 1)
                    ereg[k] <= e_lane[k];
                sm_ph <= 1'b1;
                state <= S_SM_P;
            end

            S_SM_P: begin
                calc_cnt <= calc_cnt + 1'b1;
                if (calc_cnt == LAT_ELT) begin
                    wbuf  <= calc_w;
                    sm_ph <= 1'b0;
                    write_out;
                end
            end

            // ---------------- VEC_LAYERNORM ----------------
            S_LN_RD: begin
                if (rd_done) state <= S_LN_T;
            end

            S_LN_T: begin
                calc_cnt <= calc_cnt + 1'b1;
                if (ln_st == LN_SUM) begin
                    if (calc_cnt == LAT_T) begin
                        for (k = 0; k < LG; k = k + 1) begin
                            ln_x1[k] <= $signed(xa1[4*k][22:0]) + $signed(xa1[4*k+1][22:0]) +
                                        $signed(xa1[4*k+2][22:0]) + $signed(xa1[4*k+3][22:0]);
                            ln_q1[k] <= pa3[4*k][43:0] + pa3[4*k+1][43:0] + pa3[4*k+2][43:0] + pa3[4*k+3][43:0];
                        end
                        state <= S_LN_ACC;
                    end
                end else if (calc_cnt == LAT_ELT) begin
                    wbuf <= calc_w;
                    write_out;
                end
            end

            S_LN_ACC: begin
                ln_s1 <= ln_s1_n;
                ln_s2 <= ln_s2_n;
                if (cb == nblk - 1) begin
                    ln_st <= LN_ROW;
                    state <= S_LN_R0;
                end else begin
                    cb <= cb + 1'b1;
                    start_read(a_next, n_a1, 2'd1, S_LN_RD);
                end
            end

            S_LN_R0: begin                              // n = LANES * nblk
                ln_m  <= ln_s2 << LB;
                ln_d  <= 64'd0;
                ln_b  <= 0;
                state <= S_LN_MUL;
            end

            S_LN_MUL: begin
                if (nblk[ln_b]) ln_d <= ln_d + ln_m;
                ln_m <= ln_m << 1;
                ln_b <= ln_b + 1'b1;
                if (ln_b == LN_NB - 1) state <= S_LN_D; // the pipeline has S1^2 and the mean by now
            end

            S_LN_D: begin
                ln_d    <= ln_d - pa3[0] + {32'd0, ln_eps};
                ln_mean <= va4[1];
                ln_c    <= 5'd0;
                state   <= S_LN_NORM;
            end

            S_LN_NORM: begin
                if (ln_d[63:62] == 2'b00) begin
                    ln_d <= ln_d << 2;
                    ln_c <= ln_c + 1'b1;
                end else begin
                    sq_d    <= ln_d[63:32];
                    sq_rem  <= 20'd0;
                    sq_root <= 16'd0;
                    sq_i    <= 4'd0;
                    state   <= S_LN_SQRT;
                end
            end

            S_LN_SQRT: begin
                sq_rem  <= sq_ge ? sq_r2 - sq_t : sq_r2;
                sq_root <= sq_next;
                sq_d    <= sq_d << 2;
                sq_i    <= sq_i + 1'b1;
                if (sq_i == 4'd15) begin
                    sm_sum <= {15'd0, sq_next, 1'b0};   // the divider gives 2^46 / 2q
                    dv_rem <= 0;
                    dv_q   <= 0;
                    dv_i   <= 6'd46;
                    state  <= S_SM_DIV;
                end
            end

            // ---------------- VEC_LUT ----------------
            S_LUT_FILL: begin
                lut_i <= lut_i + 1'b1;
                if (lut_i == 8'd255) start_read(row_a_w, n_a0, 2'd1, S_LUT_RD);
            end

            S_LUT_RD: begin
                if (rd_done) state <= S_LUT_C;          // abuf drives the lane RAMs this cycle
            end

            S_LUT_C: begin
                for (k = 0; k < LANES; k = k + 1)
                    wbuf[(sub_off+k)*8 +: 8] <= lut_q[k];
                write_out;
            end

            // ---------------- write + advance ----------------
            S_WR: begin
                if (wr_done) begin
                    if (is_ln) begin
                        if (cb == nblk - 1) begin
                            if (r == rows - 1) begin
                                unit_done <= 1'b1;
                                state     <= S_IDLE;
                            end else begin                  // next row: new sums
                                r       <= r + 1'b1;
                                cb      <= 0;
                                ln_st   <= LN_SUM;
                                ln_s1   <= 32'sd0;
                                ln_s2   <= 64'd0;
                                row_a_w <= row_a_w + a_pitch_w;
                                row_o_w <= row_o_w + o_pitch_w;
                                start_read(row_a_w + a_pitch_w, n_a0, 2'd1, S_LN_RD);
                            end
                        end else begin
                            cb <= cb + 1'b1;
                            start_read(a_next, n_a1, 2'd1, S_LN_RD);
                        end
                    end else if (is_sm) begin
                        if (cb == nblk - 1) begin
                            if (r == rows - 1) begin
                                unit_done <= 1'b1;
                                state     <= S_IDLE;
                            end else begin                  // next row: new max / sum
                                r       <= r + 1'b1;
                                cb      <= 0;
                                sm_pass <= 1'b0;
                                sm_max  <= 32'sh80000000;
                                sm_sum  <= 32'd0;
                                sm_limit <= sm_limit + 1'b1;
                                row_a_w <= row_a_w + a_pitch_w;
                                row_o_w <= row_o_w + o_pitch_w;
                                start_read(row_a_w + a_pitch_w, n_a0, 2'd1, S_SM_RDA);
                            end
                        end else begin
                            cb <= cb + 1'b1;
                            start_read(a_next, n_a1, 2'd1, S_SM_RDA);
                        end
                    end else if (is_epi) begin
                        if (cb != nblk - 1) begin           // next pass over this row
                            cb <= cb + 1'b1;
                            if (epi_acc)
                                start_read(row_p_w + blk_off(cb_next, 1'b1), EW32, 2'd1, S_EPI_PART);
                            else
                                state <= S_EPI_CALC;
                        end else if (r == rows - 1) begin
                            unit_done <= 1'b1;
                            state     <= S_IDLE;
                        end else begin
                            r       <= r + 1'b1;
                            cb      <= 0;
                            row_o_w <= row_o_w + o_pitch_w;
                            row_p_w <= row_p_w + p_pitch_w;
                            state   <= S_EPI_ROW;
                        end
                    end else begin
                        if (cb == nblk - 1) begin
                            if (r == rows - 1) begin
                                unit_done <= 1'b1;
                                state     <= S_IDLE;
                            end else begin
                                r       <= r + 1'b1;
                                cb      <= 0;
                                row_a_w <= row_a_w + a_pitch_w;
                                row_b_w <= row_b_w + b_pitch_w;
                                row_o_w <= row_o_w + o_pitch_w;
                                start_read(row_a_w + a_pitch_w, n_a0, 2'd1, elt_rd);
                            end
                        end else begin
                            cb <= cb + 1'b1;
                            start_read(a_next, n_a1, 2'd1, elt_rd);
                        end
                    end
                end
            end

            default: state <= S_IDLE;
            endcase
        end
    end

endmodule
