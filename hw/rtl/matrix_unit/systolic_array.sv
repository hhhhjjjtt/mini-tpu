`timescale 1ns/1ps

// ============================================================
// systolic_array — the matrix unit's systolic array (instantiated only by
// matrix_unit.sv). Originally FlexNPU's plug-and-play core compute_tile_bram,
// flavor systolic_streaming; renamed and owned here since 2026-11-09. The
// port names (act_addrb, act_doutb, ...) still follow that contract's
// BRAM-style read ports, which the matrix unit's buffers implement.
//
// Streaming output-stationary systolic array. Wavefront kept
// alive across intra-dispatch K-tile chunks; per-PE i32 accumulator
// cleared once on `start` and drained once at S_WRITEBACK.
//
// Contract (FlexNPU docs/PLUGIN_CORE_CONTRACT.md, not in this repository):
//   * tile_done_pulse fires exactly K_total/K_TILE times.
//   * Wait for act_buf_ready && wgt_buf_ready before each K_TILE.
//   * Writeback of the ARRAY_M x ARRAY_N i32 result happens ONCE.
//   * Module name / parameter list / port list are stable across
//     the ARRAY_M / ARRAY_N Stage 3 refactor. Legacy shells that
//     leave ARRAY_M / ARRAY_N unset get a square N x N array
//     (both default to N) so trusted 16x16 remains byte-identical.
//
// Datapath (rectangular ARRAY_M x ARRAY_N):
//   * ACT BRAM word k holds ARRAY_M i8 lanes: lane i = A[i,k].
//   * WGT BRAM word k holds ARRAY_N i8 lanes: lane j = B[k,j].
//   * Triangular skew: act lane i delayed i cycles before entering
//     left edge; wgt lane j delayed j cycles before top edge.
//   * PE[i][j] receives A[i,k] and B[k,j] aligned at cycle
//     k + i + j, accumulating acc += signed(a)*signed(b) on valid.
//   * C[i][j] = sum_k A[i,k]*B[k,j], signed i32, two's-complement wrap.
//
// M/N genuineness:
//   * PE array is [0:ARRAY_M-1][0:ARRAY_N-1] — physical instantiation.
//   * ACT skew chain is ARRAY_M lanes wide, ARRAY_M-1 stages deep.
//   * WGT skew chain is ARRAY_N lanes wide, ARRAY_N-1 stages deep.
//   * v_edge is per-row (ARRAY_M entries) — valid travels with A.
//   * DRAIN_LEN scales with (ARRAY_M-1) + (ARRAY_N-1).
//   * pack_row packs ARRAY_N accumulators per row.
//   * Writeback iterates ARRAY_M rows.
// ============================================================

module systolic_array #(
    parameter integer ACT_WIDTH             = 8,
    parameter integer ACC_WIDTH             = 32,
    parameter integer N                     = 8,

    // EdgeScope Stage 3 (2026-08-28): TRUE M/N split.
    // Default to N so trusted square shells stay byte-identical.
    parameter integer ARRAY_M               = N,
    parameter integer ARRAY_N               = N,

    parameter integer ACT_FIFO_DEPTH        = 8,
    parameter integer OUT_FIFO_DEPTH        = 32,

    parameter integer K_TILE                = 8,
    parameter integer MAX_K                 = 1024,

    parameter integer PIPE_LAT              = 1,
    parameter integer OUTFIFO_RD_LAT        = 1,

    parameter integer IN_BUF_ADDR_WIDTH     = $clog2(MAX_K),
    parameter integer ACT_BUF_DATA_WIDTH    = ARRAY_M * ACT_WIDTH,
    parameter integer WGT_BUF_DATA_WIDTH    = ARRAY_N * ACT_WIDTH,
    // Legacy alias: single "IN_BUF_DATA_WIDTH" instantiations continue
    // to compile at ARRAY_M == ARRAY_N. Prefer ACT_/WGT_ pair.
    parameter integer IN_BUF_DATA_WIDTH     = ACT_BUF_DATA_WIDTH,

    parameter integer OUT_BUF_ADDR_WIDTH    = $clog2(ARRAY_M),
    parameter integer OUT_BUF_DATA_WIDTH    = ARRAY_N * ACC_WIDTH,
    parameter integer OUT_BUF_WE_WIDTH      = OUT_BUF_DATA_WIDTH/8
)(
    input  wire                             clk,
    input  wire                             rst_n,
    input  wire                             start,
    input  wire [$clog2(MAX_K+1)-1:0]       K_total,

    // Chain-flag ports (accepted for shell/TB parity with int_pingpong).
    // Sampled at the start edge.
    //   accumulate_in=1 : do NOT clear acc[] on this dispatch —
    //                     preserve previous dispatch's tile.
    //   writeback_en=0  : skip S_WRITEBACK on this dispatch —
    //                     acc[] retained for the next chained dispatch.
    // Reset defaults (accumulate_in=0, writeback_en=1) reproduce
    // legacy single-dispatch behavior byte-exact.
    input  wire                             i_accumulate_in,
    input  wire                             i_writeback_en,

    output reg                              busy,
    output reg                              done,
    output reg                              tile_done_pulse,

    // ACT read port
    output reg  [IN_BUF_ADDR_WIDTH-1:0]     act_addrb,
    output reg                              act_enb,
    input  wire                             act_buf_ready,
    input  wire [ACT_BUF_DATA_WIDTH-1:0]    act_doutb,

    // WGT read port
    output reg  [IN_BUF_ADDR_WIDTH-1:0]     wgt_addrb,
    output reg                              wgt_enb,
    input  wire                             wgt_buf_ready,
    input  wire [WGT_BUF_DATA_WIDTH-1:0]    wgt_doutb,

    // OUT write port
    output reg  [OUT_BUF_ADDR_WIDTH-1:0]    out_addrb,
    output reg                              out_enb,
    output reg  [OUT_BUF_WE_WIDTH-1:0]      out_web,
    output reg  [OUT_BUF_DATA_WIDTH-1:0]    out_dinb
);

    genvar gi, gj;
    integer ps, pl;

    // ------------------------------------------------------------
    // Input lane views of the BRAM read buses.
    // ACT lanes count = ARRAY_M (A-side row index).
    // WGT lanes count = ARRAY_N (B-side col index).
    // ------------------------------------------------------------
    wire signed [ACT_WIDTH-1:0] act_lane [0:ARRAY_M-1];
    wire signed [ACT_WIDTH-1:0] wgt_lane [0:ARRAY_N-1];

    reg rd_valid;

    // ------------------------------------------------------------
    // Triangular skew shift registers.
    // sk_act: ARRAY_M lanes wide, ARRAY_M-1 stages deep.
    //   stage s holds the act lane vector delayed by (s+1) cycles;
    //   only lane (s+1) of stage s is read as a_edge[s+1].
    // sk_wgt: ARRAY_N lanes wide, ARRAY_N-1 stages deep.
    // sk_v : one bit per stage in the ACT chain (valid rides with A).
    //   depth ARRAY_M-1 (matches ACT skew); valid propagates
    //   rightward through pe_v inside the PE array once past the
    //   left edge.
    // ------------------------------------------------------------
    reg signed [ACT_WIDTH-1:0] sk_act [0:ARRAY_M-2][0:ARRAY_M-1];
    reg signed [ACT_WIDTH-1:0] sk_wgt [0:ARRAY_N-2][0:ARRAY_N-1];
    reg                        sk_v   [0:ARRAY_M-2];

    // Skewed feeds presented to the array edges.
    wire signed [ACT_WIDTH-1:0] a_edge [0:ARRAY_M-1];   // left edge, row i
    wire signed [ACT_WIDTH-1:0] b_edge [0:ARRAY_N-1];   // top edge,  col j
    wire                        v_edge [0:ARRAY_M-1];   // valid, with a_edge

    // ------------------------------------------------------------
    // PE array state.
    // ------------------------------------------------------------
    reg signed [ACT_WIDTH-1:0] pe_a [0:ARRAY_M-1][0:ARRAY_N-1];
    reg signed [ACT_WIDTH-1:0] pe_b [0:ARRAY_M-1][0:ARRAY_N-1];
    reg                        pe_v [0:ARRAY_M-1][0:ARRAY_N-1];
    (* use_dsp = "yes" *)
    reg signed [ACC_WIDTH-1:0] acc  [0:ARRAY_M-1][0:ARRAY_N-1];

    // ------------------------------------------------------------
    // Control FSM.
    // ------------------------------------------------------------
    localparam [2:0] S_IDLE      = 3'd0,
                     S_WAIT_TILE = 3'd1,
                     S_STREAM    = 3'd2,
                     S_DRAIN     = 3'd3,
                     S_WRITEBACK = 3'd4,
                     S_DONE      = 3'd5;
    reg [2:0] state;

    // Cycles for the last K-index to reach PE[ARRAY_M-1][ARRAY_N-1]:
    //   (ARRAY_M-1) skew + (ARRAY_N-1) rightward propagation
    //   + 1 acc register + slack. Legacy 16x16 uses 2*N+2 = 34 which
    //   equals (M-1)+(N-1)+4 at M=N=16 — byte-identical drain length.
    localparam integer DRAIN_LEN = (ARRAY_M - 1) + (ARRAY_N - 1) + 4;

    reg [$clog2(MAX_K+1)-1:0]  k_base, num_pass, pass_cnt;
    reg [$clog2(K_TILE+1)-1:0] k_in_tile;
    reg [$clog2(DRAIN_LEN+1)-1:0] drain_cnt;
    reg [$clog2(ARRAY_M+1)-1:0]   wb_row;

    // Latched chain flags (sampled at start).
    reg latched_writeback_en;

    // Accumulators are cleared the cycle `start` is accepted UNLESS
    // the shell asked us to preserve the previous dispatch's tile.
    // Legacy shells that don't drive i_accumulate_in tie it low so
    // this reduces to (state == S_IDLE) && start — byte-identical.
    wire acc_clear = (state == S_IDLE) && start && !i_accumulate_in;

    // Pack acc[r][*] into the OUT bus. Lane c at [c*ACC_WIDTH +: ACC_WIDTH].
    function [OUT_BUF_DATA_WIDTH-1:0] pack_row;
        input integer r;
        integer c;
        reg [OUT_BUF_DATA_WIDTH-1:0] tmp;
        begin
            tmp = '0;
            for (c = 0; c < ARRAY_N; c = c + 1)
                tmp[c*ACC_WIDTH +: ACC_WIDTH] = acc[r][c];
            pack_row = tmp;
        end
    endfunction

    // ============================================================
    // Lane views + skewed edge feeds.
    //
    // ACT edge: ARRAY_M rows. Row 0 sees the live bus; row i (>=1)
    // sees lane i of sk_act stage i-1 (i.e. delayed by i cycles).
    // WGT edge: ARRAY_N cols. Symmetric.
    // ============================================================
    generate
        for (gi = 0; gi < ARRAY_M; gi = gi + 1) begin : ACT_LANE_GEN
            assign act_lane[gi] = $signed(act_doutb[gi*ACT_WIDTH +: ACT_WIDTH]);
        end
        for (gj = 0; gj < ARRAY_N; gj = gj + 1) begin : WGT_LANE_GEN
            assign wgt_lane[gj] = $signed(wgt_doutb[gj*ACT_WIDTH +: ACT_WIDTH]);
        end

        for (gi = 0; gi < ARRAY_M; gi = gi + 1) begin : A_EDGE_GEN
            if (gi == 0) begin : A_EDGE0
                assign a_edge[gi] = act_lane[0];
                assign v_edge[gi] = rd_valid;
            end else begin : A_EDGEN
                assign a_edge[gi] = sk_act[gi-1][gi];
                assign v_edge[gi] = sk_v[gi-1];
            end
        end

        for (gj = 0; gj < ARRAY_N; gj = gj + 1) begin : B_EDGE_GEN
            if (gj == 0) begin : B_EDGE0
                assign b_edge[gj] = wgt_lane[0];
            end else begin : B_EDGEN
                assign b_edge[gj] = sk_wgt[gj-1][gj];
            end
        end
    endgenerate

    // ============================================================
    // Input pipeline: rd_valid + ACT and WGT skew shift registers.
    // ACT and WGT have independent depths at rectangular.
    // ============================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_valid <= 1'b0;
            for (ps = 0; ps < ARRAY_M-1; ps = ps + 1) begin
                sk_v[ps] <= 1'b0;
                for (pl = 0; pl < ARRAY_M; pl = pl + 1)
                    sk_act[ps][pl] <= '0;
            end
            for (ps = 0; ps < ARRAY_N-1; ps = ps + 1) begin
                for (pl = 0; pl < ARRAY_N; pl = pl + 1)
                    sk_wgt[ps][pl] <= '0;
            end
        end else begin
            rd_valid <= act_enb & wgt_enb;

            // ACT skew chain — depth ARRAY_M-1.
            if (ARRAY_M >= 2) begin
                sk_v[0] <= rd_valid;
                for (pl = 0; pl < ARRAY_M; pl = pl + 1)
                    sk_act[0][pl] <= act_lane[pl];
                for (ps = 1; ps < ARRAY_M-1; ps = ps + 1) begin
                    sk_v[ps] <= sk_v[ps-1];
                    for (pl = 0; pl < ARRAY_M; pl = pl + 1)
                        sk_act[ps][pl] <= sk_act[ps-1][pl];
                end
            end

            // WGT skew chain — depth ARRAY_N-1.
            if (ARRAY_N >= 2) begin
                for (pl = 0; pl < ARRAY_N; pl = pl + 1)
                    sk_wgt[0][pl] <= wgt_lane[pl];
                for (ps = 1; ps < ARRAY_N-1; ps = ps + 1) begin
                    for (pl = 0; pl < ARRAY_N; pl = pl + 1)
                        sk_wgt[ps][pl] <= sk_wgt[ps-1][pl];
                end
            end
        end
    end

    // ============================================================
    // PE array. Each PE registers a (->right), b (->down), v (->right)
    // and accumulates on valid. acc[] cleared only on `start` (when
    // accumulate_in=0); wavefront otherwise never resets between
    // K-tiles.
    // ============================================================
    generate
        for (gi = 0; gi < ARRAY_M; gi = gi + 1) begin : PE_ROW
            for (gj = 0; gj < ARRAY_N; gj = gj + 1) begin : PE_COL
                wire signed [ACT_WIDTH-1:0] a_in;
                wire signed [ACT_WIDTH-1:0] b_in;
                wire                        v_in;

                if (gj == 0) begin : A_EDGE_IN
                    assign a_in = a_edge[gi];
                    assign v_in = v_edge[gi];
                end else begin : A_PROP
                    assign a_in = pe_a[gi][gj-1];
                    assign v_in = pe_v[gi][gj-1];
                end

                if (gi == 0) begin : B_EDGE_IN
                    assign b_in = b_edge[gj];
                end else begin : B_PROP
                    assign b_in = pe_b[gi-1][gj];
                end

                always @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        pe_a[gi][gj] <= '0;
                        pe_b[gi][gj] <= '0;
                        pe_v[gi][gj] <= 1'b0;
                        acc [gi][gj] <= '0;
                    end else begin
                        pe_a[gi][gj] <= a_in;
                        pe_b[gi][gj] <= b_in;
                        pe_v[gi][gj] <= v_in;
                        if (acc_clear)
                            acc[gi][gj] <= '0;
                        else if (v_in)
                            acc[gi][gj] <= acc[gi][gj]
                                         + $signed(a_in) * $signed(b_in);
                    end
                end
            end
        end
    endgenerate

    // ============================================================
    // Control FSM.
    // ============================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_IDLE;
            busy            <= 1'b0;
            done            <= 1'b0;
            tile_done_pulse <= 1'b0;
            act_enb         <= 1'b0;  wgt_enb   <= 1'b0;
            act_addrb       <= '0;    wgt_addrb <= '0;
            out_enb         <= 1'b0;  out_web   <= '0;
            out_addrb       <= '0;    out_dinb  <= '0;
            k_base          <= '0;    num_pass  <= '0;  pass_cnt <= '0;
            k_in_tile       <= '0;    drain_cnt <= '0;  wb_row   <= '0;
            latched_writeback_en <= 1'b1;
        end else begin
            done            <= 1'b0;
            tile_done_pulse <= 1'b0;
            act_enb         <= 1'b0;  wgt_enb <= 1'b0;
            out_enb         <= 1'b0;  out_web <= '0;

            case (state)
            S_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    busy     <= 1'b1;
                    k_base   <= '0;
                    pass_cnt <= '0;
                    num_pass <= K_total / K_TILE;
                    latched_writeback_en <= i_writeback_en;
                    state    <= S_WAIT_TILE;
                end
            end

            S_WAIT_TILE: begin
                if (act_buf_ready && wgt_buf_ready) begin
                    k_in_tile <= '0;
                    state     <= S_STREAM;
                end
            end

            S_STREAM: begin
                act_enb   <= 1'b1;
                wgt_enb   <= 1'b1;
                act_addrb <= k_base + k_in_tile;
                wgt_addrb <= k_base + k_in_tile;
                if (k_in_tile == K_TILE-1) begin
                    tile_done_pulse <= 1'b1;
                    if (pass_cnt == num_pass - 1) begin
                        drain_cnt <= '0;
                        state     <= S_DRAIN;
                    end else begin
                        pass_cnt <= pass_cnt + 1'b1;
                        k_base   <= k_base + K_TILE;
                        state    <= S_WAIT_TILE;
                    end
                end else begin
                    k_in_tile <= k_in_tile + 1'b1;
                end
            end

            S_DRAIN: begin
                if (drain_cnt == DRAIN_LEN-1) begin
                    wb_row <= '0;
                    // writeback_en=0 : skip S_WRITEBACK; retain acc[]
                    //                  for a subsequent chained dispatch.
                    if (latched_writeback_en) begin
                        state  <= S_WRITEBACK;
                    end else begin
                        state  <= S_DONE;
                    end
                end else begin
                    drain_cnt <= drain_cnt + 1'b1;
                end
            end

            // Write ARRAY_M rows of ARRAY_N i32 lanes to OUT buffer.
            S_WRITEBACK: begin
                out_enb   <= 1'b1;
                out_web   <= {OUT_BUF_WE_WIDTH{1'b1}};
                out_addrb <= wb_row;
                out_dinb  <= pack_row(wb_row);
                if (wb_row == ARRAY_M-1) state <= S_DONE;
                else                     wb_row <= wb_row + 1'b1;
            end

            S_DONE: begin
                done  <= 1'b1;
                busy  <= 1'b0;
                state <= S_IDLE;
            end

            default: state <= S_IDLE;
            endcase
        end
    end

endmodule
