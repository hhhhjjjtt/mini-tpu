`timescale 1ns/1ps
`include "mini_tpu_config.svh"

// =============================================================================
// tb_blob — runs compiler output on the RTL and dumps DDR
// =============================================================================
//
//   make sim_blob BLOB=<dir>
//
// <dir>/ddr.hex   DDR image: 65536 lines, one 128-bit word each ($readmemh)
// <dir>/run.txt   one command per line, in order:
//                   R <region> <hex address>      set REGION_BASE
//                   P <hex address> 0             run the program at this address
//
// Prints "RESULT <addr> <status> <cycles> <descriptors>" and "PERF <addr>" followed by
// busy, wait and starved cycles of LOAD, MATRIX, VECTOR and STORE per program, then
// writes <dir>/ddr_out.hex. The DUT and the AXI-Lite master match tb_system.
// =============================================================================

module tb_blob;

    localparam integer N         = `MTPU_ARRAY_N;
    localparam integer VM_AW     = `MTPU_VM_AW;
    localparam integer AXI_AW    = 40;
    localparam integer DDR_WORDS = 65536;          // 1 MB

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    // =========================================================================
    // DUT: mini_tpu_top (the synthesized top level)
    // =========================================================================
    reg  [15:0] awaddr = 0, araddr = 0;
    reg         awvalid = 0, wvalid = 0, bready = 0, arvalid = 0, rready = 0;
    reg  [31:0] wdata = 0;
    wire        awready, wready, bvalid, arready, rvalid;
    wire [1:0]  bresp, rresp;
    wire [31:0] rdata;
    wire        irq;

    // DDR model ports: read 0 fetch, read 1 LOAD, write 0 STORE
    wire [2*AXI_AW-1:0] d_araddr;
    wire [15:0]         d_arlen;
    wire [5:0]          d_arsize;
    wire [3:0]          d_arburst;
    wire [1:0]          d_arvalid, d_arready, d_rlast, d_rvalid, d_rready;
    wire [255:0]        d_rdata;
    wire [3:0]          d_rresp;
    wire [AXI_AW-1:0]   d_awaddr;
    wire [7:0]          d_awlen;
    wire [2:0]          d_awsize;
    wire [1:0]          d_awburst, d_bresp;
    wire                d_awvalid, d_awready, d_wlast, d_wvalid, d_wready, d_bvalid, d_bready;
    wire [127:0]        d_wdata;
    wire [15:0]         d_wstrb;

    mini_tpu_top #(.AXI_AW(AXI_AW), .VM_AW(VM_AW), .N(N)) u_top (
        .clk(clk), .rst_n(rst_n),
        .s_axil_awaddr(awaddr), .s_axil_awprot(3'b000), .s_axil_awvalid(awvalid), .s_axil_awready(awready),
        .s_axil_wdata(wdata), .s_axil_wstrb(4'hF), .s_axil_wvalid(wvalid), .s_axil_wready(wready),
        .s_axil_bresp(bresp), .s_axil_bvalid(bvalid), .s_axil_bready(bready),
        .s_axil_araddr(araddr), .s_axil_arprot(3'b000), .s_axil_arvalid(arvalid), .s_axil_arready(arready),
        .s_axil_rdata(rdata), .s_axil_rresp(rresp), .s_axil_rvalid(rvalid), .s_axil_rready(rready),

        .m_axi_fetch_arid(), .m_axi_fetch_araddr(d_araddr[0 +: AXI_AW]), .m_axi_fetch_arlen(d_arlen[0 +: 8]),
        .m_axi_fetch_arsize(d_arsize[0 +: 3]), .m_axi_fetch_arburst(d_arburst[0 +: 2]),
        .m_axi_fetch_arcache(), .m_axi_fetch_arprot(),
        .m_axi_fetch_arvalid(d_arvalid[0]), .m_axi_fetch_arready(d_arready[0]),
        .m_axi_fetch_rid(1'b0), .m_axi_fetch_rdata(d_rdata[0 +: 128]), .m_axi_fetch_rresp(d_rresp[0 +: 2]),
        .m_axi_fetch_rlast(d_rlast[0]), .m_axi_fetch_rvalid(d_rvalid[0]), .m_axi_fetch_rready(d_rready[0]),

        .m_axi_load_arid(), .m_axi_load_araddr(d_araddr[AXI_AW +: AXI_AW]), .m_axi_load_arlen(d_arlen[8 +: 8]),
        .m_axi_load_arsize(d_arsize[3 +: 3]), .m_axi_load_arburst(d_arburst[2 +: 2]),
        .m_axi_load_arcache(), .m_axi_load_arprot(),
        .m_axi_load_arvalid(d_arvalid[1]), .m_axi_load_arready(d_arready[1]),
        .m_axi_load_rid(1'b0), .m_axi_load_rdata(d_rdata[128 +: 128]), .m_axi_load_rresp(d_rresp[2 +: 2]),
        .m_axi_load_rlast(d_rlast[1]), .m_axi_load_rvalid(d_rvalid[1]), .m_axi_load_rready(d_rready[1]),

        .m_axi_store_awid(), .m_axi_store_awaddr(d_awaddr), .m_axi_store_awlen(d_awlen),
        .m_axi_store_awsize(d_awsize), .m_axi_store_awburst(d_awburst),
        .m_axi_store_awcache(), .m_axi_store_awprot(),
        .m_axi_store_awvalid(d_awvalid), .m_axi_store_awready(d_awready),
        .m_axi_store_wdata(d_wdata), .m_axi_store_wstrb(d_wstrb), .m_axi_store_wlast(d_wlast),
        .m_axi_store_wvalid(d_wvalid), .m_axi_store_wready(d_wready),
        .m_axi_store_bid(1'b0), .m_axi_store_bresp(d_bresp), .m_axi_store_bvalid(d_bvalid),
        .m_axi_store_bready(d_bready),

        .irq(irq)
    );

    // internal signals used by the checks below
    wire [3:0]  push     = u_top.push;
    wire [15:0] free     = u_top.free;
    wire [3:0]  overflow = {u_top.g_fe[3].u_fe.overflow, u_top.g_fe[2].u_fe.overflow,
                            u_top.g_fe[1].u_fe.overflow, u_top.g_fe[0].u_fe.overflow};

    // ---------------- DDR ----------------
    axi_ddr_model #(.AW(AXI_AW), .NR(2), .NW(1), .WORDS(DDR_WORDS), .ERR_BASE(64'hF_0000)) u_ddr (
        .clk(clk),
        .araddr(d_araddr), .arlen(d_arlen), .arsize(d_arsize), .arburst(d_arburst),
        .arvalid(d_arvalid), .arready(d_arready),
        .rdata(d_rdata), .rresp(d_rresp), .rlast(d_rlast), .rvalid(d_rvalid), .rready(d_rready),
        .awaddr(d_awaddr), .awlen(d_awlen), .awsize(d_awsize), .awburst(d_awburst),
        .awvalid(d_awvalid), .awready(d_awready),
        .wdata(d_wdata), .wstrb(d_wstrb), .wlast(d_wlast), .wvalid(d_wvalid), .wready(d_wready),
        .bresp(d_bresp), .bvalid(d_bvalid), .bready(d_bready)
    );

    // =========================================================================
    // AXI-Lite master
    // =========================================================================
    task automatic axil_write(input [15:0] a, input [31:0] d);
        bit aw_ok, w_ok;
        begin
            @(negedge clk);
            awaddr = a; awvalid = 1; wdata = d; wvalid = 1;
            aw_ok = 0; w_ok = 0;
            while (!(aw_ok && w_ok)) begin
                @(posedge clk);
                if (awvalid && awready) aw_ok = 1;
                if (wvalid && wready)   w_ok  = 1;
                @(negedge clk);
                if (aw_ok) awvalid = 0;
                if (w_ok)  wvalid  = 0;
            end
            bready = 1;
            do @(posedge clk); while (!(bvalid));
            @(negedge clk); bready = 0;
        end
    endtask

    task automatic axil_read(input [15:0] a, output [31:0] d);
        begin
            @(negedge clk);
            araddr = a; arvalid = 1;
            do @(posedge clk); while (!(arready));
            @(negedge clk); arvalid = 0; rready = 1;
            do @(posedge clk); while (!(rvalid));
            d = rdata;
            @(negedge clk); rready = 0;
        end
    endtask

    localparam [15:0] R_CORE_CTRL = 16'h010, R_DOORBELL = 16'h014, R_STATUS = 16'h018,
                      R_PROG_LO = 16'h020, R_PROG_HI = 16'h024, R_REGION = 16'h100,
                      R_FW_VERSION = 16'h200, R_CYCLES = 16'h300, R_DESC_COUNT = 16'h340,
                      R_PERF = 16'h310,                 // UNIT_BUSY, then UNIT_WAIT and UNIT_STARVED 16 B on

                      W_IMEM = 16'h8000;

    reg [31:0] fw [0:4095];
    string     dir;

    task automatic boot();
        reg [31:0] v;
        int n, last;
        begin
            $readmemh("../fw/build/fw.hex", fw);
            last = 0;
            for (int i = 0; i < 4096; i++) if (fw[i] != 0) last = i;
            axil_write(R_CORE_CTRL, 32'h1);
            for (int i = 0; i <= last; i++) axil_write(W_IMEM + 4 * i, fw[i]);
            axil_write(R_CORE_CTRL, 32'h0);
            n = 0;
            do begin axil_read(R_FW_VERSION, v); n++; end while (v != 32'h0001_0000 && n < 2000);
            if (v != 32'h0001_0000) begin $display("TB FAIL: firmware did not boot"); $finish; end
        end
    endtask

    // +trace: one line per event of the units, for timing analysis
    //   P<u> push to unit u   A<u> unit u takes a command   D<u> unit u done
    //   S core starts a tile  F core finishes a tile   R matrix running 0/1
    reg tracing = 0;
    initial tracing = $test$plusargs("trace");
    always @(posedge clk) if (tracing) begin
        for (int u = 0; u < 4; u++) begin
            if (u_top.push[u])                         $display("T %0d P%0d", u_ddr.cyc, u);
            if (u_top.f_valid[u] && u_top.f_ready[u])  $display("T %0d A%0d", u_ddr.cyc, u);
            if (u_top.unit_done[u])                    $display("T %0d D%0d", u_ddr.cyc, u);
        end
        if (u_top.u_mat.c_start)  $display("T %0d S", u_ddr.cyc);
        if (u_top.u_mat.core_fin) $display("T %0d F", u_ddr.cyc);
    end

    task automatic run_prog(input [31:0] pb);
        reg [31:0] st, cycles, dc;
        begin
            axil_write(R_PROG_LO, pb);
            axil_write(R_PROG_HI, 0);
            axil_write(R_DOORBELL, 1);
            do begin
                repeat (50) @(negedge clk);
                axil_read(R_STATUS, st);
            end while (st[3:0] == 4'd1);
            axil_read(R_CYCLES, cycles);
            axil_read(R_DESC_COUNT, dc);
            $display("RESULT %h %h %0d %0d", pb, st, cycles, dc);
            $write("PERF %h", pb);                          // busy, wait, starved per unit
            for (int u = 0; u < 4; u++) begin
                axil_read(R_PERF + 4 * u, cycles);  $write(" %0d", cycles);
                axil_read(R_PERF + 16 + 4 * u, cycles);  $write(" %0d", cycles);
                axil_read(R_PERF + 32 + 4 * u, cycles);  $write(" %0d", cycles);
            end
            $write("\n");
        end
    endtask

    initial begin
        int fd, n;
        string kind;
        reg [31:0] a, b;
        if (!$value$plusargs("blob=%s", dir)) begin $display("TB FAIL: no +blob=<dir>"); $finish; end
        $readmemh({dir, "/ddr.hex"}, u_ddr.mem);
        repeat (5) @(negedge clk);
        rst_n = 1;
        repeat (5) @(negedge clk);
        boot();
        fd = $fopen({dir, "/run.txt"}, "r");
        if (fd == 0) begin $display("TB FAIL: no run.txt"); $finish; end
        while ($fscanf(fd, "%s %h %h", kind, a, b) == 3) begin
            if (kind == "R") begin
                axil_write(R_REGION + 8 * a, b);
                axil_write(R_REGION + 8 * a + 4, 0);
            end else run_prog(a);
        end
        $fclose(fd);
        $writememh({dir, "/ddr_out.hex"}, u_ddr.mem);
        if (u_ddr.violations != 0) $display("TB FAIL: %0d AXI protocol violations", u_ddr.violations);
        else $display("TB DONE");
        $finish;
    end

endmodule
