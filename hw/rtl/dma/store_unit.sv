`timescale 1ns/1ps

// =============================================================================
// store_unit — STORE (vmem -> DDR, 2D strided)
// =============================================================================
//
// INTERFACE_SPEC.md §5.2. Unit 3.
//
//   vmem reader : walks the source rows (vmem_stride apart) on vmem read
//                 client 2 into a WFIFO; a request is made only when the
//                 FIFO has room for it and the read in flight.
//   AW side     : dma_burst_gen walks the destination rows; up to MAX_OUT
//                 bursts without a B response. Each AW pushes its beat count
//                 into a small LEN FIFO for the W side.
//   W side      : sends WFIFO beats for bursts whose AW has been accepted,
//                 WLAST on the last beat of each burst. WSTRB all ones.
//   B side      : counts responses.
//
// Done only after every burst's B response (so data is in DDR when the program
// reports DONE). BRESP[1] on any burst -> ERR_UNIT_AXI instead of done.
//
// abort (CORE_RESET): every accepted AW still needs its W beats and B, so the
// unit stops reading vmem and issuing new bursts, keeps a pending AW valid
// until accepted, finishes the accepted bursts (remaining FIFO data, then
// WSTRB = 0 beats), waits for all B responses, and returns to idle.
//
// =============================================================================

module store_unit #(
    parameter integer AXI_AW      = 40,
    parameter integer VM_AW       = 17,
    parameter integer MAX_OUT     = 8,          // bursts awaiting B
    parameter integer WFIFO_DEPTH = 32,
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

    // ---- AXI4 write master ----
    output wire [0:0]               m_axi_awid,
    output wire [AXI_AW-1:0]        m_axi_awaddr,
    output wire [7:0]               m_axi_awlen,
    output wire [2:0]               m_axi_awsize,
    output wire [1:0]               m_axi_awburst,
    output wire [3:0]               m_axi_awcache,
    output wire [2:0]               m_axi_awprot,
    output wire                     m_axi_awvalid,
    input  wire                     m_axi_awready,
    output wire [127:0]             m_axi_wdata,
    output wire [15:0]              m_axi_wstrb,
    output wire                     m_axi_wlast,
    output wire                     m_axi_wvalid,
    input  wire                     m_axi_wready,
    input  wire [0:0]               m_axi_bid,
    input  wire [1:0]               m_axi_bresp,
    input  wire                     m_axi_bvalid,
    output wire                     m_axi_bready,

    // ---- vmem read client ----
    output wire                     vm_rd_req,
    output wire [VM_AW-1:0]         vm_rd_addr,
    input  wire                     vm_rd_gnt,
    input  wire                     vm_rd_rvalid,
    input  wire [127:0]             vm_rd_rdata
);

    localparam [7:0] OP_STORE = 8'h03;
    localparam [7:0] ERR_BAD_OPCODE = 8'h10, ERR_BAD_FIELD = 8'h11,
                     ERR_UNIT_AXI = 8'h21, ERR_VMEM_RANGE = 8'h22;
    localparam [47:0] VMEM_BYTES = 48'd1 << (VM_AW + 4);
    localparam integer FCW = $clog2(WFIFO_DEPTH + 1);
    localparam integer OCW = $clog2(MAX_OUT + 1);

    localparam [1:0] S_IDLE = 2'd0, S_CHECK = 2'd1, S_RUN = 2'd2, S_DRAIN = 2'd3;
    reg [1:0] state;
    reg       aw_wait;                      // AW valid but not yet accepted

    assign cmd_ready = (state == S_IDLE);
    assign busy      = (state != S_IDLE);

    // ---- latched command ----
    reg [7:0]   op_q;
    reg [63:0]  ddr;
    reg [31:0]  vaddr, dstride, vstride;
    reg [15:0]  rows, row_bytes;

    wire [47:0] vm_last = {16'd0, vaddr} + (rows - 1'b1) * {16'd0, vstride} + {32'd0, row_bytes};
    wire bad = (rows == 0) || (row_bytes == 0) || (row_bytes[3:0] != 0) || (row_bytes > 16'd4096) ||
               (ddr[3:0] != 0) || (dstride[3:0] != 0) || (vaddr[3:0] != 0) || (vstride[3:0] != 0);

    // =========================================================================
    // vmem reader -> WFIFO
    // =========================================================================
    wire         wf_empty, wf_full;
    wire [127:0] wf_dout;
    wire [FCW-1:0] wf_count;
    wire         wf_pop;
    reg          rd_inflight;               // granted last cycle, data arrives now
    reg  [31:0]  rd_row_base;
    reg  [15:0]  rd_off, rd_rows_left;

    sfifo #(.W(128), .DEPTH(WFIFO_DEPTH)) u_wf (
        .clk(clk), .rst_n(rst_n),
        .push(vm_rd_rvalid && state == S_RUN), .din(vm_rd_rdata), .pop(wf_pop),
        .dout(wf_dout), .empty(wf_empty), .full(wf_full), .count(wf_count)
    );

    assign vm_rd_req  = (state == S_RUN) && (rd_rows_left != 0) &&
                        ({1'b0, wf_count} + rd_inflight < WFIFO_DEPTH);
    assign vm_rd_addr = (rd_row_base + {16'd0, rd_off}) >> 4;
    wire   rd_fire    = vm_rd_req && vm_rd_gnt;

    // =========================================================================
    // AW side
    // =========================================================================
    reg            gen_start;
    wire           gen_active;
    reg  [OCW-1:0] out_cnt;
    wire           aw_fire = m_axi_awvalid && m_axi_awready;
    wire           b_fire  = m_axi_bvalid && m_axi_bready;

    dma_burst_gen #(.AW(AXI_AW)) u_aw (
        .clk(clk), .rst_n(rst_n),
        .start(gen_start), .base(ddr), .rows(rows), .row_bytes(row_bytes), .stride(dstride),
        .active(gen_active), .addr(m_axi_awaddr), .len(m_axi_awlen), .fire(aw_fire)
    );

    assign m_axi_awid    = 1'b0;
    assign m_axi_awsize  = 3'd4;
    assign m_axi_awburst = 2'b01;
    assign m_axi_awcache = AXI_CACHE;
    assign m_axi_awprot  = 3'b000;
    assign m_axi_awvalid = ((state == S_RUN) && !gen_start && gen_active && (out_cnt < MAX_OUT)) ||
                           ((state == S_DRAIN) && aw_wait);

    // ---- LEN FIFO: burst lengths whose AW was accepted ----
    wire       lf_empty;
    wire [7:0] lf_len;
    wire       lf_pop;

    sfifo #(.W(8), .DEPTH(MAX_OUT)) u_lf (
        .clk(clk), .rst_n(rst_n),
        .push(aw_fire), .din(m_axi_awlen), .pop(lf_pop),
        .dout(lf_len), .empty(lf_empty), .full(), .count()
    );

    // =========================================================================
    // W side
    // =========================================================================
    reg  [7:0] w_beat;
    wire       w_fire = m_axi_wvalid && m_axi_wready;

    // in S_DRAIN, beats without data carry WSTRB = 0 (the FIFO cannot refill there)
    wire   drain_null   = (state == S_DRAIN) && wf_empty;
    assign m_axi_wvalid = ((state == S_RUN) && !wf_empty && !lf_empty) ||
                          ((state == S_DRAIN) && !lf_empty);
    assign m_axi_wdata  = drain_null ? 128'd0 : wf_dout;
    assign m_axi_wstrb  = drain_null ? 16'h0000 : 16'hFFFF;
    assign m_axi_wlast  = (w_beat == lf_len);
    assign wf_pop       = (w_fire && !drain_null) || ((state == S_DRAIN) && lf_empty);
    assign lf_pop       = w_fire && m_axi_wlast;

    assign m_axi_bready = 1'b1;

    // =========================================================================
    // Sequencer
    // =========================================================================
    reg axi_err;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; unit_done <= 1'b0; err <= 1'b0; err_code <= 0;
            op_q <= 0; ddr <= 0; vaddr <= 0; dstride <= 0; vstride <= 0; rows <= 0; row_bytes <= 0;
            rd_inflight <= 1'b0; rd_row_base <= 0; rd_off <= 0; rd_rows_left <= 0;
            gen_start <= 1'b0; out_cnt <= 0; w_beat <= 0; axi_err <= 1'b0; aw_wait <= 1'b0;
        end else begin
            unit_done   <= 1'b0;
            gen_start   <= 1'b0;
            rd_inflight <= rd_fire;

            out_cnt <= out_cnt + aw_fire - b_fire;
            aw_wait <= m_axi_awvalid && !m_axi_awready;
            if (b_fire && m_axi_bresp[1]) axi_err <= 1'b1;
            if (w_fire) w_beat <= m_axi_wlast ? 8'd0 : w_beat + 1'b1;

            if (rd_fire) begin
                if (rd_off + 16'd16 == row_bytes) begin
                    rd_off       <= 0;
                    rd_row_base  <= rd_row_base + vstride;
                    rd_rows_left <= rd_rows_left - 1'b1;
                end else
                    rd_off <= rd_off + 16'd16;
            end

            if (abort && state != S_IDLE && state != S_DRAIN) begin
                state <= S_DRAIN;
                err <= 1'b0; err_code <= 0;
            end else if (abort && state == S_IDLE) begin
                err <= 1'b0; err_code <= 0;
            end else case (state)
                S_IDLE: if (cmd_valid) begin
                    op_q      <= cmd[7:0];
                    ddr       <= cmd[128 +: 64];
                    vaddr     <= cmd[192 +: 32];
                    rows      <= cmd[224 +: 16];
                    row_bytes <= cmd[240 +: 16];
                    dstride   <= cmd[256 +: 32];
                    vstride   <= cmd[288 +: 32];
                    state     <= S_CHECK;
                end

                S_CHECK: begin
                    if (op_q != OP_STORE) begin
                        err <= 1'b1; err_code <= ERR_BAD_OPCODE; state <= S_IDLE;
                    end else if (bad) begin
                        err <= 1'b1; err_code <= ERR_BAD_FIELD; state <= S_IDLE;
                    end else if (vm_last > VMEM_BYTES) begin
                        err <= 1'b1; err_code <= ERR_VMEM_RANGE; state <= S_IDLE;
                    end else begin
                        gen_start    <= 1'b1;
                        axi_err      <= 1'b0;
                        rd_row_base  <= vaddr;
                        rd_off       <= 0;
                        rd_rows_left <= rows;
                        state        <= S_RUN;
                    end
                end

                // all bursts issued and acknowledged (W precedes B, so all data is out)
                S_RUN: if (!gen_start && !gen_active && out_cnt == 0 && !aw_fire) begin
                    if (axi_err) begin err <= 1'b1; err_code <= ERR_UNIT_AXI; end
                    else         unit_done <= 1'b1;
                    state <= S_IDLE;
                end

                S_DRAIN: if (lf_empty && out_cnt == 0 && !m_axi_awvalid && !aw_fire && wf_empty)
                    state <= S_IDLE;
            endcase
        end
    end

endmodule
