`timescale 1ns/1ps

// =============================================================================
// load_unit — LOAD (DDR -> vmem, 2D strided) and LOAD_WGT (DDR -> weight buffer)
// =============================================================================
//
// INTERFACE_SPEC.md §5.1, §5.3. Unit 0.
//
//   AR side : dma_burst_gen walks the source rows; up to MAX_OUT bursts in
//             flight, across row boundaries (single ARID -> in-order data).
//   R side  : beats go into an RFIFO (rready = !full), so the AXI port never
//             waits on a vmem bank grant combinationally.
//   Writer  : LOAD     -> vmem write client, destination rows vmem_stride apart
//             LOAD_WGT -> weight buffer port, rows 0..k-1 of the selected half,
//                         N/16 words per row, read as N/16 DMA rows of k words
//
// Done (unit_done pulse) when the last beat is written. An AXI error response
// (RRESP[1]) still drains the transfer, then raises ERR_UNIT_AXI instead of done.
// Field errors are detected before any AXI traffic and drop the command.
// Errors are sticky (err / err_code) until reset or abort.
//
// abort (CORE_RESET): AXI masters must not be reset mid-transaction, so the
// unit stops issuing, keeps a pending AR valid until accepted, discards the
// read data still owed to it, then returns to idle. rst_n is the global reset.
//
// =============================================================================

module load_unit #(
    parameter integer AXI_AW      = 40,
    parameter integer VM_AW       = 17,         // vmem word address bits (2 MB)
    parameter integer K_MAX       = 512,
    parameter integer N           = 16,         // matrix width: weight row bytes
    parameter integer MAX_OUT     = 8,          // outstanding AR bursts
    parameter integer RFIFO_DEPTH = 16,
    parameter [3:0]   AXI_CACHE   = 4'b0000
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     abort,      // level: drain and go idle

    // ---- Command (front end) ----
    input  wire                     cmd_valid,
    input  wire [511:0]             cmd,
    output wire                     cmd_ready,
    output reg                      unit_done,
    output reg                      err,
    output reg  [7:0]               err_code,
    output wire                     busy,

    // ---- AXI4 read master ----
    output wire [0:0]               m_axi_arid,
    output wire [AXI_AW-1:0]        m_axi_araddr,
    output wire [7:0]               m_axi_arlen,
    output wire [2:0]               m_axi_arsize,
    output wire [1:0]               m_axi_arburst,
    output wire [3:0]               m_axi_arcache,
    output wire [2:0]               m_axi_arprot,
    output wire                     m_axi_arvalid,
    input  wire                     m_axi_arready,
    input  wire [0:0]               m_axi_rid,
    input  wire [127:0]             m_axi_rdata,
    input  wire [1:0]               m_axi_rresp,
    input  wire                     m_axi_rlast,
    input  wire                     m_axi_rvalid,
    output wire                     m_axi_rready,

    // ---- vmem write client ----
    output wire                     vm_wr_req,
    output wire [VM_AW-1:0]         vm_wr_addr,
    output wire [127:0]             vm_wr_data,
    input  wire                     vm_wr_gnt,

    // ---- Weight buffer write port (matrix unit) ----
    output reg                      wgt_wr_en,
    output reg                      wgt_wr_half,
    output reg  [$clog2(K_MAX*N/16)-1:0] wgt_wr_addr,
    output reg  [127:0]             wgt_wr_data
);

    localparam [7:0] OP_LOAD = 8'h01, OP_LOAD_WGT = 8'h02;
    localparam [7:0] ERR_BAD_OPCODE = 8'h10, ERR_BAD_FIELD = 8'h11,
                     ERR_UNIT_AXI = 8'h21, ERR_VMEM_RANGE = 8'h22;
    localparam [47:0] VMEM_BYTES = 48'd1 << (VM_AW + 4);

    localparam [2:0] S_IDLE = 3'd0, S_CHECK = 3'd1, S_RUN = 3'd2, S_END = 3'd3, S_DRAIN = 3'd4;
    reg [2:0] state;
    reg       ar_wait;                      // AR valid but not yet accepted

    assign cmd_ready = (state == S_IDLE);
    assign busy      = (state != S_IDLE);

    // ---- latched command ----
    reg         is_wgt, half;
    reg [7:0]   op_q, half_byte;
    reg [63:0]  ddr;
    reg [31:0]  vaddr, dstride, vstride;
    reg [15:0]  rows, row_bytes, kk;

    // ---- field checks (S_CHECK) ----
    wire [47:0] vm_last  = {16'd0, vaddr} + (rows - 1'b1) * {16'd0, vstride} + {32'd0, row_bytes};
    wire load_bad = (rows == 0) || (row_bytes == 0) || (row_bytes[3:0] != 0) || (row_bytes > 16'd4096) ||
                    (ddr[3:0] != 0) || (dstride[3:0] != 0) || (vaddr[3:0] != 0) || (vstride[3:0] != 0);
    wire wgt_bad  = (kk == 0) || (kk[3:0] != 0) || (kk > K_MAX) || (half_byte > 8'd1) || (ddr[3:0] != 0);

    // ---- AR side ----
    reg                       gen_start;
    wire                      gen_active;
    wire [$clog2(MAX_OUT+1)-1:0] out_cnt_w;
    reg  [$clog2(MAX_OUT+1)-1:0] out_cnt;
    wire                      ar_fire = m_axi_arvalid && m_axi_arready;

    dma_burst_gen #(.AW(AXI_AW)) u_ar (
        .clk(clk), .rst_n(rst_n),
        .start(gen_start), .base(ddr), .rows(rows), .row_bytes(row_bytes), .stride(dstride),
        .active(gen_active), .addr(m_axi_araddr), .len(m_axi_arlen), .fire(ar_fire)
    );

    assign m_axi_arid    = 1'b0;
    assign m_axi_arsize  = 3'd4;            // 16 B
    assign m_axi_arburst = 2'b01;           // INCR
    assign m_axi_arcache = AXI_CACHE;
    assign m_axi_arprot  = 3'b000;
    assign m_axi_arvalid = ((state == S_RUN) && !gen_start && gen_active && (out_cnt < MAX_OUT)) ||
                           ((state == S_DRAIN) && ar_wait);

    // ---- R side -> RFIFO ----
    wire         rf_empty, rf_full;
    wire [127:0] rf_dout;
    wire         rf_pop;
    wire         r_fire = m_axi_rvalid && m_axi_rready;
    reg          axi_err;

    assign m_axi_rready = ((state == S_RUN) && !rf_full) || (state == S_DRAIN);

    sfifo #(.W(128), .DEPTH(RFIFO_DEPTH)) u_rf (
        .clk(clk), .rst_n(rst_n),
        .push(r_fire && state == S_RUN), .din(m_axi_rdata), .pop(rf_pop),
        .dout(rf_dout), .empty(rf_empty), .full(rf_full), .count()
    );

    // ---- writer ----
    reg [31:0] wr_row_base;
    reg [15:0] wr_off;                      // bytes within the row
    reg [15:0] wr_rows_left;
    reg [15:0] wgt_idx;
    wire       wr_row_end = (wr_off + 16'd16 == row_bytes);

    assign vm_wr_req  = (state == S_RUN) && !is_wgt && !rf_empty && (wr_rows_left != 0);
    assign vm_wr_addr = (wr_row_base + {16'd0, wr_off}) >> 4;
    assign vm_wr_data = rf_dout;

    wire wgt_take = (state == S_RUN) && is_wgt && !rf_empty && (wr_rows_left != 0);
    assign rf_pop = (vm_wr_req && vm_wr_gnt) || wgt_take || (state == S_DRAIN);

    // ---- sequencer ----
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; unit_done <= 1'b0; err <= 1'b0; err_code <= 0;
            is_wgt <= 0; half <= 0; op_q <= 0; half_byte <= 0; ddr <= 0;
            vaddr <= 0; dstride <= 0; vstride <= 0; rows <= 0; row_bytes <= 0; kk <= 0;
            gen_start <= 1'b0; out_cnt <= 0; axi_err <= 1'b0; ar_wait <= 1'b0;
            wr_row_base <= 0; wr_off <= 0; wr_rows_left <= 0; wgt_idx <= 0;
            wgt_wr_en <= 1'b0; wgt_wr_half <= 1'b0; wgt_wr_addr <= 0; wgt_wr_data <= 0;
        end else begin
            unit_done <= 1'b0;
            gen_start <= 1'b0;
            wgt_wr_en <= 1'b0;

            out_cnt <= out_cnt + ar_fire - (r_fire && m_axi_rlast);
            ar_wait <= m_axi_arvalid && !m_axi_arready;
            if (r_fire && m_axi_rresp[1]) axi_err <= 1'b1;

            if (abort && state != S_IDLE && state != S_DRAIN) begin
                state <= S_DRAIN;
                err <= 1'b0; err_code <= 0;
            end else if (abort && state == S_IDLE) begin
                err <= 1'b0; err_code <= 0;
            end else case (state)
                S_IDLE: if (cmd_valid) begin
                    op_q      <= cmd[7:0];
                    is_wgt    <= (cmd[7:0] == OP_LOAD_WGT);
                    ddr       <= cmd[128 +: 64];
                    vaddr     <= cmd[192 +: 32];
                    half_byte <= cmd[192 +: 8];
                    half      <= cmd[192];
                    kk        <= cmd[224 +: 16];
                    dstride   <= cmd[256 +: 32];
                    vstride   <= cmd[288 +: 32];
                    if (cmd[7:0] == OP_LOAD_WGT) begin
                        rows      <= N / 16;                     // k * N bytes in all
                        row_bytes <= {cmd[224 +: 12], 4'd0};     // 16 * k (k <= 4095 here)
                        dstride   <= {cmd[224 +: 12], 4'd0};
                    end else begin
                        rows      <= cmd[224 +: 16];
                        row_bytes <= cmd[240 +: 16];
                    end
                    state <= S_CHECK;
                end

                S_CHECK: begin
                    if (op_q != OP_LOAD && op_q != OP_LOAD_WGT) begin
                        err <= 1'b1; err_code <= ERR_BAD_OPCODE; state <= S_IDLE;
                    end else if (is_wgt ? wgt_bad : load_bad) begin
                        err <= 1'b1; err_code <= ERR_BAD_FIELD; state <= S_IDLE;
                    end else if (!is_wgt && vm_last > VMEM_BYTES) begin
                        err <= 1'b1; err_code <= ERR_VMEM_RANGE; state <= S_IDLE;
                    end else begin
                        gen_start    <= 1'b1;
                        axi_err      <= 1'b0;
                        wr_row_base  <= vaddr;
                        wr_off       <= 0;
                        wr_rows_left <= rows;
                        wgt_idx      <= 0;
                        state        <= S_RUN;
                    end
                end

                S_RUN: begin
                    if (rf_pop) begin
                        if (is_wgt) begin
                            wgt_wr_en   <= 1'b1;
                            wgt_wr_half <= half;
                            wgt_wr_addr <= wgt_idx[$clog2(K_MAX*N/16)-1:0];
                            wgt_wr_data <= rf_dout;
                            wgt_idx     <= wgt_idx + 1'b1;
                        end
                        if (wr_row_end) begin
                            wr_off       <= 0;
                            wr_row_base  <= wr_row_base + vstride;
                            wr_rows_left <= wr_rows_left - 1'b1;
                            if (wr_rows_left == 16'd1) state <= S_END;
                        end else
                            wr_off <= wr_off + 16'd16;
                    end
                end

                S_END: begin                        // last beat written; all bursts returned
                    if (axi_err) begin err <= 1'b1; err_code <= ERR_UNIT_AXI; end
                    else         unit_done <= 1'b1;
                    state <= S_IDLE;
                end

                S_DRAIN: if (out_cnt == 0 && !m_axi_arvalid && !r_fire && rf_empty)
                    state <= S_IDLE;

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
