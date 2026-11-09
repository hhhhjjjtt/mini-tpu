`timescale 1ns/1ps
`include "mini_tpu_config.svh"

// =============================================================================
// mini_tpu_top — mini-tpu accelerator top level (structural, Verilog-2001)
// =============================================================================
//
// TENSOR_CORE_UPDATE_SPEC.md §3, INTERFACE_SPEC.md §7-§8.
//
//   s_axil_*       AXI-Lite slave, 64 KB window (ARM: control, mailbox, I-mem)
//   m_axi_fetch_*  AXI4 read master  (scalar unit descriptor fetch)
//   m_axi_load_*   AXI4 read master  (LOAD / LOAD_WGT)
//   m_axi_store_*  AXI4 write master (STORE)
//   irq            level, PL -> PS (CORE_CTRL.IRQ_EN && IRQ_STATUS != 0)
//
// One clock (clk, 100 MHz target) and one active-low reset (rst_n). Masters
// are 128-bit, ID width 1 (always 0), 40-bit addresses.
//
// Verilog-2001 so the file can be a block-design module reference; the
// modules below it are SystemVerilog. The X_INTERFACE attributes tell the
// block design which clock / reset / interrupt the AXI interfaces belong to
// (ignored by simulators).
//
// =============================================================================

module mini_tpu_top #(
    parameter integer AXI_AW = 40,
    parameter integer VM_AW  = `MTPU_VM_AW,         // vmem word address bits
    parameter integer N      = `MTPU_ARRAY_N
)(
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axil:m_axi_fetch:m_axi_load:m_axi_store, ASSOCIATED_RESET rst_n" *)
    input  wire                 clk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 rst_n RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire                 rst_n,

    // ---- AXI-Lite slave ----
    input  wire [15:0]          s_axil_awaddr,
    input  wire [2:0]           s_axil_awprot,
    input  wire                 s_axil_awvalid,
    output wire                 s_axil_awready,
    input  wire [31:0]          s_axil_wdata,
    input  wire [3:0]           s_axil_wstrb,
    input  wire                 s_axil_wvalid,
    output wire                 s_axil_wready,
    output wire [1:0]           s_axil_bresp,
    output wire                 s_axil_bvalid,
    input  wire                 s_axil_bready,
    input  wire [15:0]          s_axil_araddr,
    input  wire [2:0]           s_axil_arprot,
    input  wire                 s_axil_arvalid,
    output wire                 s_axil_arready,
    output wire [31:0]          s_axil_rdata,
    output wire [1:0]           s_axil_rresp,
    output wire                 s_axil_rvalid,
    input  wire                 s_axil_rready,

    // ---- AXI4 read master: descriptor fetch ----
    output wire [0:0]           m_axi_fetch_arid,
    output wire [AXI_AW-1:0]    m_axi_fetch_araddr,
    output wire [7:0]           m_axi_fetch_arlen,
    output wire [2:0]           m_axi_fetch_arsize,
    output wire [1:0]           m_axi_fetch_arburst,
    output wire [3:0]           m_axi_fetch_arcache,
    output wire [2:0]           m_axi_fetch_arprot,
    output wire                 m_axi_fetch_arvalid,
    input  wire                 m_axi_fetch_arready,
    input  wire [0:0]           m_axi_fetch_rid,
    input  wire [127:0]         m_axi_fetch_rdata,
    input  wire [1:0]           m_axi_fetch_rresp,
    input  wire                 m_axi_fetch_rlast,
    input  wire                 m_axi_fetch_rvalid,
    output wire                 m_axi_fetch_rready,

    // ---- AXI4 read master: LOAD ----
    output wire [0:0]           m_axi_load_arid,
    output wire [AXI_AW-1:0]    m_axi_load_araddr,
    output wire [7:0]           m_axi_load_arlen,
    output wire [2:0]           m_axi_load_arsize,
    output wire [1:0]           m_axi_load_arburst,
    output wire [3:0]           m_axi_load_arcache,
    output wire [2:0]           m_axi_load_arprot,
    output wire                 m_axi_load_arvalid,
    input  wire                 m_axi_load_arready,
    input  wire [0:0]           m_axi_load_rid,
    input  wire [127:0]         m_axi_load_rdata,
    input  wire [1:0]           m_axi_load_rresp,
    input  wire                 m_axi_load_rlast,
    input  wire                 m_axi_load_rvalid,
    output wire                 m_axi_load_rready,

    // ---- AXI4 write master: STORE ----
    output wire [0:0]           m_axi_store_awid,
    output wire [AXI_AW-1:0]    m_axi_store_awaddr,
    output wire [7:0]           m_axi_store_awlen,
    output wire [2:0]           m_axi_store_awsize,
    output wire [1:0]           m_axi_store_awburst,
    output wire [3:0]           m_axi_store_awcache,
    output wire [2:0]           m_axi_store_awprot,
    output wire                 m_axi_store_awvalid,
    input  wire                 m_axi_store_awready,
    output wire [127:0]         m_axi_store_wdata,
    output wire [15:0]          m_axi_store_wstrb,
    output wire                 m_axi_store_wlast,
    output wire                 m_axi_store_wvalid,
    input  wire                 m_axi_store_wready,
    input  wire [0:0]           m_axi_store_bid,
    input  wire [1:0]           m_axi_store_bresp,
    input  wire                 m_axi_store_bvalid,
    output wire                 m_axi_store_bready,

    (* X_INTERFACE_INFO = "xilinx.com:signal:interrupt:1.0 irq INTERRUPT" *)
    (* X_INTERFACE_PARAMETER = "SENSITIVITY LEVEL_HIGH" *)
    output wire                 irq
);

    // =========================================================================
    // Scalar unit <-> unit front ends
    // =========================================================================
    wire         units_rst_n, clear_done;
    wire [3:0]   push;
    wire [511:0] push_cmd;
    localparam integer FREE_W = `MTPU_FREE_W;

    wire [4*FREE_W-1:0] free;
    wire [95:0]  done_counts;
    wire [3:0]   unit_done, unit_busy, unit_err, wait_stall, starved;
    wire [31:0]  unit_err_code;
    wire [3:0]   f_valid, f_ready;
    wire [2047:0] f_cmd;

    assign m_axi_fetch_arprot = 3'b000;

    scalar_unit #(.AXI_AW(AXI_AW), .FREE_W(FREE_W), .FIFO_DEPTH(`MTPU_FIFO_DEPTH), .HW_ARCH(`MTPU_HW_ARCH)) u_scalar (
        .clk(clk), .rst_n(rst_n),
        .s_axil_awaddr(s_axil_awaddr), .s_axil_awvalid(s_axil_awvalid), .s_axil_awready(s_axil_awready),
        .s_axil_wdata(s_axil_wdata), .s_axil_wstrb(s_axil_wstrb), .s_axil_wvalid(s_axil_wvalid),
        .s_axil_wready(s_axil_wready), .s_axil_bresp(s_axil_bresp), .s_axil_bvalid(s_axil_bvalid),
        .s_axil_bready(s_axil_bready), .s_axil_araddr(s_axil_araddr), .s_axil_arvalid(s_axil_arvalid),
        .s_axil_arready(s_axil_arready), .s_axil_rdata(s_axil_rdata), .s_axil_rresp(s_axil_rresp),
        .s_axil_rvalid(s_axil_rvalid), .s_axil_rready(s_axil_rready),
        .irq(irq),
        .m_axi_arid(m_axi_fetch_arid), .m_axi_araddr(m_axi_fetch_araddr), .m_axi_arlen(m_axi_fetch_arlen),
        .m_axi_arsize(m_axi_fetch_arsize), .m_axi_arburst(m_axi_fetch_arburst),
        .m_axi_arcache(m_axi_fetch_arcache), .m_axi_arvalid(m_axi_fetch_arvalid),
        .m_axi_arready(m_axi_fetch_arready), .m_axi_rid(m_axi_fetch_rid), .m_axi_rdata(m_axi_fetch_rdata),
        .m_axi_rresp(m_axi_fetch_rresp), .m_axi_rlast(m_axi_fetch_rlast), .m_axi_rvalid(m_axi_fetch_rvalid),
        .m_axi_rready(m_axi_fetch_rready),
        .units_rst_n(units_rst_n), .clear_done(clear_done), .push(push), .push_cmd(push_cmd),
        .free(free), .done_counts(done_counts), .unit_done(unit_done), .unit_busy(unit_busy),
        .unit_err(unit_err), .unit_err_code(unit_err_code), .wait_stall(wait_stall), .starved(starved)
    );

    genvar g;
    generate
        for (g = 0; g < 4; g = g + 1) begin : g_fe
            unit_frontend #(.DEPTH(`MTPU_FIFO_DEPTH), .FREE_W(FREE_W)) u_fe (
                .clk(clk), .rst_n(units_rst_n), .clear(clear_done),
                .push(push[g]), .push_cmd(push_cmd), .free(free[g*FREE_W +: FREE_W]), .overflow(),
                .done_counts(done_counts),
                .cmd_valid(f_valid[g]), .cmd(f_cmd[g*512 +: 512]), .cmd_ready(f_ready[g]),
                .unit_done(unit_done[g]), .done_count(done_counts[g*24 +: 24]),
                .wait_stall(wait_stall[g]), .starved(starved[g])
            );
        end
    endgenerate

    // =========================================================================
    // vmem: read clients 0 MATRIX, 1 VECTOR, 2 STORE; write clients 0 LOAD, 1 VECTOR
    // =========================================================================
    wire [2:0]         rd_req, rd_gnt, rd_rvalid;
    wire [3*VM_AW-1:0] rd_addr;
    wire [3*128-1:0]   rd_rdata;
    wire [1:0]         wr_req, wr_gnt;
    wire [2*VM_AW-1:0] wr_addr;
    wire [2*128-1:0]   wr_data;

    vmem #(.NB(`MTPU_VMEM_BANKS)) u_vmem (
        .clk(clk), .rst_n(rst_n),
        .rd_req(rd_req), .rd_addr(rd_addr), .rd_gnt(rd_gnt),
        .rd_rvalid(rd_rvalid), .rd_rdata(rd_rdata),
        .wr_req(wr_req), .wr_addr(wr_addr), .wr_data(wr_data), .wr_gnt(wr_gnt)
    );

    // =========================================================================
    // Units. LOAD / STORE keep the global reset and drain on CORE_RESET (abort).
    // =========================================================================
    wire         wgt_wr_en, wgt_wr_half;
    wire [`MTPU_WGT_AW-1:0] wgt_wr_addr;
    wire [127:0] wgt_wr_data;
    wire         s_valid, s_last, s_ready;
    localparam integer T = (N < 16) ? 16 : N;           // tile width (matrix_unit.sv)
    wire [T*32-1:0] s_data;

    load_unit #(.AXI_AW(AXI_AW), .VM_AW(VM_AW), .K_MAX(`MTPU_MAX_K), .N(T)) u_load (
        .clk(clk), .rst_n(rst_n), .abort(!units_rst_n),
        .cmd_valid(f_valid[0]), .cmd(f_cmd[0 +: 512]), .cmd_ready(f_ready[0]),
        .unit_done(unit_done[0]), .err(unit_err[0]), .err_code(unit_err_code[0 +: 8]), .busy(unit_busy[0]),
        .m_axi_arid(m_axi_load_arid), .m_axi_araddr(m_axi_load_araddr), .m_axi_arlen(m_axi_load_arlen),
        .m_axi_arsize(m_axi_load_arsize), .m_axi_arburst(m_axi_load_arburst), .m_axi_arcache(m_axi_load_arcache),
        .m_axi_arprot(m_axi_load_arprot), .m_axi_arvalid(m_axi_load_arvalid), .m_axi_arready(m_axi_load_arready),
        .m_axi_rid(m_axi_load_rid), .m_axi_rdata(m_axi_load_rdata), .m_axi_rresp(m_axi_load_rresp),
        .m_axi_rlast(m_axi_load_rlast), .m_axi_rvalid(m_axi_load_rvalid), .m_axi_rready(m_axi_load_rready),
        .vm_wr_req(wr_req[0]), .vm_wr_addr(wr_addr[0 +: VM_AW]), .vm_wr_data(wr_data[0 +: 128]),
        .vm_wr_gnt(wr_gnt[0]),
        .wgt_wr_en(wgt_wr_en), .wgt_wr_half(wgt_wr_half), .wgt_wr_addr(wgt_wr_addr), .wgt_wr_data(wgt_wr_data)
    );

    matrix_unit #(.N(N), .VM_AW(VM_AW), .K_MAX(`MTPU_MAX_K)) u_mat (
        .clk(clk), .rst_n(units_rst_n),
        .cmd_valid(f_valid[1]), .cmd(f_cmd[512 +: 512]), .cmd_ready(f_ready[1]),
        .unit_done(unit_done[1]), .err(unit_err[1]), .err_code(unit_err_code[8 +: 8]), .busy(unit_busy[1]),
        .vm_rd_req(rd_req[0]), .vm_rd_addr(rd_addr[0 +: VM_AW]), .vm_rd_gnt(rd_gnt[0]),
        .vm_rd_rvalid(rd_rvalid[0]), .vm_rd_rdata(rd_rdata[0 +: 128]),
        .wgt_wr_en(wgt_wr_en), .wgt_wr_half(wgt_wr_half),
        .wgt_wr_addr(wgt_wr_addr), .wgt_wr_data(wgt_wr_data),
        .m_valid(s_valid), .m_data(s_data), .m_last(s_last), .m_ready(s_ready)
    );

    vector_unit #(.N(T), .LANES(`MTPU_VECTOR_LANES), .VM_AW(VM_AW), .MAX_LEN(`MTPU_MAX_LEN)) u_vec (
        .clk(clk), .rst_n(units_rst_n),
        .cmd_valid(f_valid[2]), .cmd(f_cmd[1024 +: 512]), .cmd_ready(f_ready[2]),
        .unit_done(unit_done[2]), .err(unit_err[2]), .err_code(unit_err_code[16 +: 8]), .busy(unit_busy[2]),
        .s_valid(s_valid), .s_data(s_data), .s_last(s_last), .s_ready(s_ready),
        .vm_rd_req(rd_req[1]), .vm_rd_addr(rd_addr[VM_AW +: VM_AW]), .vm_rd_gnt(rd_gnt[1]),
        .vm_rd_rvalid(rd_rvalid[1]), .vm_rd_rdata(rd_rdata[128 +: 128]),
        .vm_wr_req(wr_req[1]), .vm_wr_addr(wr_addr[VM_AW +: VM_AW]),
        .vm_wr_data(wr_data[128 +: 128]), .vm_wr_gnt(wr_gnt[1])
    );

    store_unit #(.AXI_AW(AXI_AW), .VM_AW(VM_AW)) u_store (
        .clk(clk), .rst_n(rst_n), .abort(!units_rst_n),
        .cmd_valid(f_valid[3]), .cmd(f_cmd[1536 +: 512]), .cmd_ready(f_ready[3]),
        .unit_done(unit_done[3]), .err(unit_err[3]), .err_code(unit_err_code[24 +: 8]), .busy(unit_busy[3]),
        .m_axi_awid(m_axi_store_awid), .m_axi_awaddr(m_axi_store_awaddr), .m_axi_awlen(m_axi_store_awlen),
        .m_axi_awsize(m_axi_store_awsize), .m_axi_awburst(m_axi_store_awburst),
        .m_axi_awcache(m_axi_store_awcache), .m_axi_awprot(m_axi_store_awprot),
        .m_axi_awvalid(m_axi_store_awvalid), .m_axi_awready(m_axi_store_awready),
        .m_axi_wdata(m_axi_store_wdata), .m_axi_wstrb(m_axi_store_wstrb), .m_axi_wlast(m_axi_store_wlast),
        .m_axi_wvalid(m_axi_store_wvalid), .m_axi_wready(m_axi_store_wready),
        .m_axi_bid(m_axi_store_bid), .m_axi_bresp(m_axi_store_bresp),
        .m_axi_bvalid(m_axi_store_bvalid), .m_axi_bready(m_axi_store_bready),
        .vm_rd_req(rd_req[2]), .vm_rd_addr(rd_addr[2*VM_AW +: VM_AW]), .vm_rd_gnt(rd_gnt[2]),
        .vm_rd_rvalid(rd_rvalid[2]), .vm_rd_rdata(rd_rdata[256 +: 128])
    );

endmodule
