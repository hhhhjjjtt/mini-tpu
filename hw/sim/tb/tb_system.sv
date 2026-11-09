`timescale 1ns/1ps
`include "mini_tpu_config.svh"

// =============================================================================
// tb_system — program-driven system test: ARM -> scalar unit -> units -> DDR
// =============================================================================
//
// DUT: mini_tpu_top, the synthesized top level (scalar unit with PicoRV32 + issue
// engine + real firmware hw/fw/build/fw.hex, 4 unit front ends,
// load_unit, matrix_unit, vector_unit, store_unit, vmem).
//
// Testbench models:
//   * ARM: AXI-Lite master tasks. Loads firmware through the I-mem window,
//     sets regions, writes PROG_BASE, rings DOORBELL, polls STATUS.
//   * DDR: axi_ddr_model, 1 MB, three AXI4 ports (fetch read, LOAD read,
//     STORE write) with random latency / stalls, protocol checks, and an
//     SLVERR window at 0xF0000.
//
// Programs (built here as 64 B descriptors in DDR):
//   P1  LOAD x3 -> GEMM -> VEC_EPILOGUE -> STORE -> HALT           (data checked)
//   P2  2 chunks, NEXT_CHUNK prefetch, LOOP x2 over a body unrolled for weight
//       ping-pong, region strides, WAR/RAW waits across all 4 units, FENCE
//                                                                  (data checked)
//   P3  dispatch rate: LOOP x64 of 4 tiny ops; reports cycles per descriptor
//   P4  BREAK / RESUME, IRQ_STATUS, PL->PS irq
//   P5  errors + CORE_RESET recovery: bad magic, bad opcode, unset region,
//       unit error (VEC_SOFTMAX), watchdog (GEMM without consumer), ABORT,
//       bad ddr_region (issue engine stop), ABORT while the issue engine is
//       blocked on a full FIFO
//   P6  DMA: strided LOAD/STORE round trip with 4 KB-crossing rows, LOAD/STORE
//       field errors, AXI SLVERR on read and write, CORE_RESET in the middle of
//       a large LOAD and a large STORE (drain), bandwidth
//   P7  attention head on-chip: LOAD_WGT_VM transposed (K^T) -> GEMM Q.K^T ->
//       requant -> S in vmem -> LOAD_WGT_VM straight (V) -> GEMM S.V (waits on
//       VECTOR) -> i32 -> STORE                                      (data checked)
//   P8  GEMM throughput: A 256 x 512 . W 512 x 64 (4 weight tiles, prefetch into
//       the other half), same program as sw/runtime/tests/mtpu_test.c "tput"  (rows sampled)
//   P9  attention head with softmax, same program as sw/runtime/tests/mtpu_test.c "head":
//       2 x (LOAD_WGT_VM K^T + GEMM + requant) -> VEC_SOFTMAX -> LOAD_WGT_VM V ->
//       GEMM P.V (waits on VECTOR) -> i32 -> STORE              (data checked)
//   P1  again after all resets
// =============================================================================

module tb_system;

    localparam integer N         = `MTPU_ARRAY_N;
    localparam integer VM_AW     = `MTPU_VM_AW;
    localparam integer AXI_AW    = 40;
    localparam integer DDR_WORDS = 65536;          // 1 MB

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    longint cyc = 0;
    always @(posedge clk) cyc <= cyc + 1;

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
    // Push monitor (dispatch rate)
    // =========================================================================
    longint push_t [0:4095];
    int     npush = 0;
    bit     log_en = 0;
    always @(posedge clk)
        if (log_en && push != 0 && npush < 4096) begin
            push_t[npush] = cyc;
            npush++;
        end

    always @(posedge clk)
        if (overflow != 0) $display("FAIL: front-end overflow %b", overflow);

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

    // =========================================================================
    // Register offsets (INTERFACE_SPEC §7)
    // =========================================================================
    localparam [15:0] R_HW_ID = 16'h000, R_HW_ARCH = 16'h008, R_CORE_CTRL = 16'h010,
                      R_DOORBELL = 16'h014, R_STATUS = 16'h018, R_IRQ_STATUS = 16'h01C,
                      R_PROG_LO = 16'h020, R_PROG_HI = 16'h024, R_WATCHDOG = 16'h030,
                      R_REGION = 16'h100, R_FW_VERSION = 16'h200, R_FW_HEARTBEAT = 16'h208,
                      R_ERR_DESC = 16'h210, R_ERR_UNIT = 16'h214, R_ERR_INFO0 = 16'h218,
                      R_CYCLES = 16'h300, R_BUSY = 16'h310, R_WAIT = 16'h320, R_STARVED = 16'h330,
                      R_DESC_COUNT = 16'h340, R_TRACE_COUNT = 16'h344,
                      W_TRACE = 16'h2000, W_DMEM = 16'h4000, W_IMEM = 16'h8000;

    // DDR map (byte addresses)
    localparam int RG0 = 32'h40000, RG1 = 32'h60000, RG2 = 32'h70000, RG3 = 32'h80000;
    function automatic int VB(input int n); VB = n * 32'h40000; endfunction   // vmem bank base

    integer errors = 0, checked = 0;

    task automatic expect32(input [31:0] got, input [31:0] exp, input string what);
        if (got !== exp) begin
            $display("FAIL %s: got %h exp %h", what, got, exp);
            errors++;
        end
    endtask

    // =========================================================================
    // Firmware boot
    // =========================================================================
    reg [31:0] fw [0:4095];

    task automatic boot_core();
        reg [31:0] v; int n;
        begin
            axil_write(R_CORE_CTRL, 32'h1);
            axil_write(R_CORE_CTRL, 32'h0);
            n = 0;
            do begin axil_read(R_FW_VERSION, v); n++; end while (v != 32'h0001_0000 && n < 2000);
            if (v != 32'h0001_0000) begin $display("FAIL: firmware did not boot"); errors++; end
        end
    endtask

    // =========================================================================
    // Descriptor builders (INTERFACE_SPEC §4-§5); byte offset o -> bits [8o +: n]
    // =========================================================================
    function automatic [511:0] dw(input [511:0] d, input int u, input int dd);
        begin d[16 + u] = 1'b1; d[32 + 16*u +: 16] = dd; dw = d; end
    endfunction

    function automatic [511:0] dLOAD(input bit st, input int rg, input int off, input int va,
                                     input int rows, input int rb, input int ds, input int vs);
        reg [511:0] d;
        begin
            d = 0; d[7:0] = st ? 8'h03 : 8'h01; d[128 +: 8] = rg; d[160 +: 32] = off;
            d[192 +: 32] = va; d[224 +: 16] = rows; d[240 +: 16] = rb;
            d[256 +: 32] = ds; d[288 +: 32] = vs;
            dLOAD = d;
        end
    endfunction

    function automatic [511:0] dLOADW(input int rg, input int off, input int h, input int k);
        reg [511:0] d;
        begin d = 0; d[7:0] = 8'h02; d[128 +: 8] = rg; d[160 +: 32] = off; d[192 +: 8] = h; d[224 +: 16] = k; dLOADW = d; end
    endfunction

    function automatic [511:0] dGEMM(input int act, input int pitch, input int mt, input int kk, input int h);
        reg [511:0] d;
        begin
            d = 0; d[7:0] = 8'h10;
            d[192 +: 32] = act; d[224 +: 16] = pitch; d[240 +: 16] = mt; d[256 +: 16] = kk; d[272 +: 8] = h;
            dGEMM = d;
        end
    endfunction

    function automatic [511:0] dWL(input int src, input int pitch, input int kk, input int h, input bit trans);
        reg [511:0] d;
        begin
            d = 0; d[7:0] = 8'h11; d[8] = trans;
            d[192 +: 32] = src; d[224 +: 16] = pitch; d[256 +: 16] = kk; d[272 +: 8] = h;
            dWL = d;
        end
    endfunction

    function automatic [511:0] dEPI(input int mode, input int flags, input int zp, input int lo, input int hi,
                                    input int rows, input int dpitch, input int dst, input int params);
        reg [511:0] d;
        begin
            d = 0; d[7:0] = 8'h20; d[15:8] = flags;
            d[192 +: 8] = mode; d[200 +: 8] = zp; d[208 +: 8] = lo; d[216 +: 8] = hi;
            d[224 +: 16] = rows; d[240 +: 16] = dpitch; d[256 +: 32] = dst; d[352 +: 32] = params;
            dEPI = d;
        end
    endfunction

    function automatic [511:0] dPASS(input int a, input int o, input int params);   // ELTWISE PASS_A 1x16 i8
        reg [511:0] d;
        begin
            d = 0; d[7:0] = 8'h21; d[192 +: 8] = 5; d[224 +: 16] = 1; d[240 +: 16] = 16;
            d[256 +: 32] = a; d[288 +: 32] = a; d[320 +: 32] = o;
            d[352 +: 16] = 16; d[368 +: 16] = 16; d[384 +: 16] = 16; d[416 +: 32] = params;
            dPASS = d;
        end
    endfunction

    function automatic [511:0] dOP(input [7:0] op, input int p0, input int p1);  // control ops
        reg [511:0] d;
        begin d = 0; d[7:0] = op; d[192 +: 32] = p0; d[224 +: 16] = p1; dOP = d; end
    endfunction

    function automatic [511:0] dLOOP(input int count, input int body, input int s0, input int s3);
        reg [511:0] d;
        begin
            d = dOP(8'hF0, count, body);
            d[256 + 32*0 +: 32] = s0;                // region_stride[R0]
            d[256 + 32*3 +: 32] = s3;                // region_stride[R3]
            dLOOP = d;
        end
    endfunction

    task automatic put_desc(input int addr, input [511:0] d);
        int i;
        for (i = 0; i < 4; i++) u_ddr.mem[addr / 16 + i] = d[128*i +: 128];
    endtask

    task automatic put_header(input int pb, input int off, input int len, input int rmask, input int magic);
        reg [511:0] h;
        begin
            h = 0;
            h[0 +: 32] = magic; h[32 +: 16] = 2; h[48 +: 16] = 64;
            h[64 +: 8] = 16; h[72 +: 8] = 8'h0F; h[80 +: 16] = 2048; h[96 +: 8] = rmask;
            h[128 +: 32] = off; h[160 +: 16] = len; h[224 +: 32] = 32'hC0DE;
            put_desc(pb, h);
        end
    endtask

    // Simple program builder: one chunk right after the header
    reg [511:0] pd [0:255];
    int         pn;

    task automatic emit_single(input int pb, input int rmask);
        int i;
        begin
            put_header(pb, 64, pn, rmask, 32'h4750544D);
            for (i = 0; i < pn; i++) put_desc(pb + 64 + 64 * i, pd[i]);
        end
    endtask

    // Run: PROG_BASE, DOORBELL = RUN, poll STATUS until it leaves RUNNING
    task automatic wait_not_running(output [31:0] st, input int max_polls);
        int n;
        begin
            n = 0;
            do begin
                repeat (50) @(negedge clk);
                axil_read(R_STATUS, st);
                n++;
            end while (st[3:0] == 4'd1 && n < max_polls);
        end
    endtask

    task automatic run_prog(input int pb, output [31:0] st, input int max_polls);
        begin
            axil_write(R_PROG_LO, pb);
            axil_write(R_PROG_HI, 0);
            axil_write(R_DOORBELL, 1);
            wait_not_running(st, max_polls);
        end
    endtask

    // =========================================================================
    // Golden models (same as tb_vector_path)
    // =========================================================================
    function automatic int apply_scale_ref(input int v, input int m, input int sh, input bit dr);
        longint prod, rnd;
        begin
            prod = longint'(v) * longint'(m);
            rnd  = (sh == 0) ? 0 : (longint'(1) << (sh - 1));
            if (dr && sh > 31)
                rnd += (v >= 0) ? (longint'(1) << 30) : -(longint'(1) << 30);
            apply_scale_ref = int'((prod + rnd) >>> sh);
        end
    endfunction

    function automatic int clamp_ref(input int v, input int lo, input int hi);
        clamp_ref = (v < lo) ? lo : (v > hi) ? hi : v;
    endfunction

    task automatic ddr_wb(input int addr, input int v);
        u_ddr.mem[addr / 16][(addr % 16) * 8 +: 8] = v;
    endtask

    task automatic ddr_ww(input int addr, input int v);      // 32-bit, aligned
        u_ddr.mem[addr / 16][(addr % 16) * 8 +: 32] = v;
    endtask

    function automatic int ddr_rb(input int addr);
        reg [7:0] b;
        begin b = u_ddr.mem[addr / 16][(addr % 16) * 8 +: 8]; ddr_rb = $signed(b); end
    endfunction

    localparam integer MAXR = 32, MAXK = 128, MAXT = 4;
    int A    [0:MAXR-1][0:MAXK-1];
    int W    [0:MAXT-1][0:MAXK-1][0:N-1];
    int bias [0:N-1], mult [0:N-1], shf [0:N-1];

    task automatic put_A(input int rows, input int kk, input int addr, input int pitch);
        int r, t;
        for (r = 0; r < rows; r++)
            for (t = 0; t < kk; t++) begin
                A[r][t] = $signed(8'($urandom));
                ddr_wb(addr + r * pitch + t, A[r][t]);
            end
    endtask

    task automatic put_W(input int tile, input int kk, input int addr);     // §2.5 layout
        int t, j;
        for (t = 0; t < kk; t++)
            for (j = 0; j < N; j++) begin
                W[tile][t][j] = $signed(8'($urandom));
                ddr_wb(addr + t * 16 + j, W[tile][t][j]);
            end
    endtask

    task automatic put_table(input int addr, input int shmin, input int shmax);       // §9.1
        int j;
        for (j = 0; j < N; j++) begin
            bias[j] = int'($urandom % 20001) - 10000;
            mult[j] = 32'h4000_0000 | ($urandom & 32'h3FFF_FFFF);
            shf[j]  = shmin + ($urandom % (shmax - shmin + 1));
            u_ddr.mem[(addr + j * 16) / 16] = 0;
            ddr_ww(addr + j * 16, bias[j]);
            ddr_ww(addr + j * 16 + 4, mult[j]);
            ddr_wb(addr + j * 16 + 8, shf[j]);
        end
    endtask

    // check an i8 output tile (rows x 16) in DDR against GEMM + requant golden
    task automatic check_out(input int tile, input int rows, input int kk, input int addr, input int pitch,
                             input int zp, input bit dr, input string what);
        int r, j, t, c, v;
        for (r = 0; r < rows; r++)
            for (j = 0; j < N; j++) begin
                c = 0;
                for (t = 0; t < kk; t++) c += A[r][t] * W[tile][t][j];
                v = clamp_ref(apply_scale_ref(c + bias[j], mult[j], shf[j], dr) + zp, -128, 127);
                if (ddr_rb(addr + r * pitch + j) != v) begin
                    if (errors < 12)
                        $display("MISMATCH %s r%0d j%0d: got %0d exp %0d", what, r, j, ddr_rb(addr + r * pitch + j), v);
                    errors++;
                end
                checked++;
            end
    endtask

    // =========================================================================
    // Programs
    // =========================================================================
    localparam int PB1 = 32'h1000, PB2 = 32'h3000, PB3 = 32'h6000, PB4 = 32'hA000, PB5 = 32'hC000;

    task automatic build_p1();
        begin
            put_A(32, 64, RG2, 64);
            put_W(0, 64, RG0);
            put_table(RG1, 30, 40);
            pn = 0;
            pd[pn++] = dLOAD(0, 1, 0, VB(1), 1, 256, 256, 256);              // params
            pd[pn++] = dLOAD(0, 2, 0, VB(0), 32, 64, 64, 64);                // A
            pd[pn++] = dLOADW(0, 0, 0, 64);                                  // W -> half 0
            pd[pn++] = dw(dGEMM(VB(0), 64, 2, 64, 0), 0, 0);                 // wait LOAD all
            pd[pn++] = dw(dEPI(0, 8'b10, 5, -128, 127, 32, 16, VB(3), VB(1)), 0, 0);
            pd[pn++] = dw(dLOAD(1, 3, 0, VB(3), 32, 16, 16, 16), 2, 0);      // STORE, wait VECTOR all
            pd[pn++] = dOP(8'hFF, 0, 0);
            emit_single(PB1, 8'h0F);
        end
    endtask

    task automatic run_p1(input string tag);
        reg [31:0] st, v;
        begin
            for (int i = 0; i < 32; i++) u_ddr.mem[(RG3 + i * 16) / 16] = 128'hx;
            run_prog(PB1, st, 400);
            expect32(st, 32'h3, {tag, " STATUS"});
            axil_read(R_DESC_COUNT, v);
            expect32(v, 7, {tag, " DESC_COUNT"});
            check_out(0, 32, 64, RG3, 16, 5, 1, tag);
        end
    endtask

    // P2: chunk 0 = prologue + NEXT_CHUNK; chunk 1 = LOOP x2 over 2 weight tiles each
    task automatic build_p2();
        int t, c1;
        reg [511:0] d;
        begin
            put_A(32, 128, RG2 + 32'h1000, 128);
            for (t = 0; t < 4; t++) put_W(t, 128, RG0 + 32'h2000 + t * 2048);
            put_table(RG1 + 32'h100, 26, 34);
            for (t = 0; t < 32 * 4; t++) u_ddr.mem[(RG3 + 32'h1000) / 16 + t] = 128'hx;

            c1 = PB2 + 32'h400;
            put_header(PB2, 64, 4, 8'h0F, 32'h4750544D);
            put_desc(PB2 + 64 + 64*0, dLOAD(0, 1, 32'h100, VB(1) + 32'h100, 1, 256, 256, 256));
            put_desc(PB2 + 64 + 64*1, dLOAD(0, 2, 32'h1000, VB(0) + 32'h8000, 32, 128, 128, 128));
            put_desc(PB2 + 64 + 64*2, dOP(8'h00, 0, 0));                                     // NOP
            put_desc(PB2 + 64 + 64*3, dOP(8'hF1, c1 - PB2, 11));                             // NEXT_CHUNK

            put_desc(c1 + 64*0,  dLOOP(2, 8, 4096, 32));
            put_desc(c1 + 64*1,  dw(dLOADW(0, 32'h2000, 0, 128), 1, 1));                       // WAR on half 0
            put_desc(c1 + 64*2,  dw(dLOADW(0, 32'h2000 + 2048, 1, 128), 1, 0));                // WAR on half 1
            put_desc(c1 + 64*3,  dw(dGEMM(VB(0) + 32'h8000, 128, 2, 128, 0), 0, 1));           // RAW: all but newest LOAD
            d = dw(dEPI(0, 0, -7, -128, 127, 32, 16, VB(3) + 32'h4000, VB(1) + 32'h100), 0, 3); // params loaded
            put_desc(c1 + 64*4,  dw(d, 3, 1));                                                 // WAR on X0
            put_desc(c1 + 64*5,  dw(dGEMM(VB(0) + 32'h8000, 128, 2, 128, 1), 0, 0));
            put_desc(c1 + 64*6,  dw(dEPI(0, 0, -7, -128, 127, 32, 16, VB(3) + 32'h5000, VB(1) + 32'h100), 3, 0));
            put_desc(c1 + 64*7,  dw(dLOAD(1, 3, 32'h1000,      VB(3) + 32'h4000, 32, 16, 64, 16), 2, 1));
            put_desc(c1 + 64*8,  dw(dLOAD(1, 3, 32'h1000 + 16, VB(3) + 32'h5000, 32, 16, 64, 16), 2, 0));
            put_desc(c1 + 64*9,  dOP(8'hF3, 0, 0));                                            // FENCE
            put_desc(c1 + 64*10, dOP(8'hFF, 0, 0));                                            // HALT
        end
    endtask

    task automatic run_p2();
        reg [31:0] st, v, b, w, s;
        int t, u;
        begin
            run_prog(PB2, st, 2000);
            expect32(st, 32'h3, "P2 STATUS");
            axil_read(R_DESC_COUNT, v);
            expect32(v, 4 + 1 + 2 * 8 + 2, "P2 DESC_COUNT");    // chunk0 + LOOP + 2x body + FENCE + HALT
            for (t = 0; t < 4; t++) check_out(t, 32, 128, RG3 + 32'h1000 + 16 * t, 64, -7, 0, "P2");
            axil_read(R_CYCLES, v);
            $display("P2: %0d cycles RUN->DONE", v);
            for (u = 0; u < 4; u++) begin
                axil_read(R_BUSY + 4*u, b); axil_read(R_WAIT + 4*u, w); axil_read(R_STARVED + 4*u, s);
                $display("    unit %0d: busy %0d  wait %0d  starved %0d", u, b, w, s);
            end
        end
    endtask

    // P3: dispatch rate — LOOP x64 over {PASS, LOAD, PASS, LOAD} (each op is tiny)
    task automatic build_p3();
        begin
            pn = 0;
            pd[pn++] = dLOOP(64, 4, 0, 0);
            pd[pn++] = dPASS(VB(4), VB(6), VB(4) + 32'h100);
            pd[pn++] = dLOAD(0, 2, 0, VB(7), 1, 16, 16, 16);
            pd[pn++] = dPASS(VB(4) + 16, VB(6) + 16, VB(4) + 32'h100);
            pd[pn++] = dLOAD(0, 2, 16, VB(7) + 16, 1, 16, 16, 16);
            pd[pn++] = dOP(8'hFF, 0, 0);
            emit_single(PB3, 8'h0F);
        end
    endtask

    task automatic run_p3();
        reg [31:0] st, cycles, dc;
        longint iv [0:4095];
        longint tmp;
        int i, j, n;
        begin
            npush = 0; log_en = 1;
            run_prog(PB3, st, 20000);
            log_en = 0;
            expect32(st, 32'h3, "P3 STATUS");
            axil_read(R_CYCLES, cycles);
            axil_read(R_DESC_COUNT, dc);
            expect32(dc, 1 + 64 * 4 + 1, "P3 DESC_COUNT");
            expect32(npush, 256, "P3 pushes");
            // median push-to-push interval
            n = npush - 1;
            for (i = 0; i < n; i++) iv[i] = push_t[i + 1] - push_t[i];
            for (i = 1; i < n; i++) begin
                tmp = iv[i]; j = i - 1;
                while (j >= 0 && iv[j] > tmp) begin iv[j + 1] = iv[j]; j--; end
                iv[j + 1] = tmp;
            end
            $display("P3 dispatch: %0d descriptors in %0d cycles RUN->DONE (%0.1f cycles/descriptor)",
                     dc, cycles, real'(cycles) / real'(dc));
            $display("    push-to-push interval: min %0d, median %0d, p90 %0d, max %0d cycles",
                     iv[0], iv[n / 2], iv[(n * 9) / 10], iv[n - 1]);
        end
    endtask

    // =========================================================================
    // P6: DMA
    // =========================================================================
    localparam int PB6 = 32'hE000;

    // Round trip: strided LOAD DDR -> vmem, strided STORE vmem -> DDR, rows crossing 4 KB
    localparam int RT_ROWS = 5, RT_RB = 4096 - 48;               // 253 beats per row
    localparam int RT_SRC = 32'h2F00, RT_SSTR = RT_RB + 32;      // R2-relative
    localparam int RT_DST = 32'h10100, RT_DSTR = RT_RB + 80;     // R3-relative
    localparam int RT_VM = 32'h40000 * 5, RT_VSTR = RT_RB + 16;

    task automatic run_roundtrip(input string tag);
        reg [31:0] st;
        int r, w;
        begin
            for (r = 0; r < RT_ROWS; r++)
                for (w = 0; w < RT_DSTR / 16; w++)
                    u_ddr.mem[(RG3 + RT_DST + r * RT_DSTR) / 16 + w] = {4{32'h5E47_1E00 + w}};
            for (r = 0; r < RT_ROWS; r++)
                for (w = 0; w < RT_RB / 16; w++)
                    u_ddr.mem[(RG2 + RT_SRC + r * RT_SSTR) / 16 + w] = {$urandom, $urandom, $urandom, $urandom};
            pn = 0;
            pd[pn++] = dLOAD(0, 2, RT_SRC, RT_VM, RT_ROWS, RT_RB, RT_SSTR, RT_VSTR);
            pd[pn++] = dw(dLOAD(1, 3, RT_DST, RT_VM, RT_ROWS, RT_RB, RT_DSTR, RT_VSTR), 0, 0);
            pd[pn++] = dOP(8'hFF, 0, 0);
            emit_single(PB6, 8'h0F);
            run_prog(PB6, st, 2000);
            expect32(st, 32'h3, {tag, " STATUS"});
            for (r = 0; r < RT_ROWS; r++)
                for (w = 0; w < RT_DSTR / 16; w++) begin
                    if (w < RT_RB / 16) begin
                        if (u_ddr.mem[(RG3 + RT_DST + r * RT_DSTR) / 16 + w] !== u_ddr.mem[(RG2 + RT_SRC + r * RT_SSTR) / 16 + w]) begin
                            if (errors < 12) $display("MISMATCH %s row %0d word %0d", tag, r, w);
                            errors++;
                        end
                    end else if (u_ddr.mem[(RG3 + RT_DST + r * RT_DSTR) / 16 + w] !== {4{32'h5E47_1E00 + w}}) begin
                        if (errors < 12) $display("FAIL %s: gap word %0d after row %0d overwritten", tag, w, r);
                        errors++;
                    end
                    checked++;
                end
        end
    endtask

    // Run a one-descriptor program expected to fail in a unit
    task automatic expect_unit_error(input [511:0] d, input int unit, input [7:0] code, input string tag);
        reg [31:0] st, v;
        begin
            pn = 0;
            pd[pn++] = d;
            pd[pn++] = dOP(8'hFF, 0, 0);
            emit_single(PB6, 8'h0F);
            run_prog(PB6, st, 400);
            expect32(st, {16'd0, code, 8'h04}, tag);
            axil_read(R_ERR_UNIT, v); expect32(v, unit, {tag, " ERR_UNIT"});
            boot_core();
        end
    endtask

    // Start a long transfer, then pulse CORE_RESET while the unit is mid-burst
    task automatic reset_mid_transfer(input bit st_, input string tag);
        reg [31:0] v;
        begin
            pn = 0;
            if (st_) pd[pn++] = dLOAD(1, 3, 32'h20000, VB(6), 16, 4096, 4096, 4096);
            else     pd[pn++] = dLOAD(0, 2, 0,         VB(6), 16, 4096, 4096, 4096);
            pd[pn++] = dOP(8'hFF, 0, 0);
            emit_single(PB6, 8'h0F);
            axil_write(R_PROG_LO, PB6);
            axil_write(R_DOORBELL, 1);
            do @(negedge clk); while (!(st_ ? u_top.u_store.busy : u_top.u_load.busy));
            repeat (400) @(negedge clk);
            if (st_ ? !u_top.u_store.busy : !u_top.u_load.busy) begin $display("FAIL %s: transfer already finished", tag); errors++; end
            axil_write(R_CORE_CTRL, 32'h1);
            axil_write(R_CORE_CTRL, 32'h0);
            do @(negedge clk); while (st_ ? u_top.u_store.busy : u_top.u_load.busy);     // drain
            boot_core();
            axil_read(R_STATUS, v); expect32(v, 0, {tag, " STATUS idle after reset"});
        end
    endtask

    task automatic run_p6();
        reg [31:0] st, lb, sb;
        begin
            axil_write(R_REGION + 8*4, 32'hF_0000);              // R4 -> SLVERR window
            // (a) round trip with strides and 4 KB crossings
            run_roundtrip("P6a");
            // (b) bandwidth: 64 KB LOAD, then 64 KB STORE
            pn = 0;
            pd[pn++] = dLOAD(0, 2, 0, VB(6), 16, 4096, 4096, 4096);
            pd[pn++] = dw(dLOAD(1, 3, 32'h20000, VB(6), 16, 4096, 4096, 4096), 0, 0);
            pd[pn++] = dOP(8'hFF, 0, 0);
            emit_single(PB6, 8'h0F);
            run_prog(PB6, st, 2000);
            expect32(st, 32'h3, "P6b STATUS");
            axil_read(R_BUSY + 0, lb); axil_read(R_BUSY + 12, sb);
            $display("P6b DMA: LOAD 64 KB in %0d cycles (%0.2f B/cycle), STORE 64 KB in %0d cycles (%0.2f B/cycle)",
                     lb, 65536.0 / lb, sb, 65536.0 / sb);
            for (int i = 0; i < 4096; i++)
                if (u_ddr.mem[(RG3 + 32'h20000) / 16 + i] !== u_ddr.mem[RG2 / 16 + i]) begin
                    if (errors < 12) $display("MISMATCH P6b word %0d", i);
                    errors++;
                end
            checked += 4096;
            // (c) field errors and (d) AXI error responses
            expect_unit_error(dLOAD(0, 2, 0, VB(7), 1, 24, 32, 32), 0, 8'h11, "P6c LOAD row_bytes=24");
            expect_unit_error(dLOADW(0, 0, 0, 520), 0, 8'h11, "P6c LOAD_WGT k=520");
            expect_unit_error(dLOAD(1, 3, 0, VB(7) + 32'h3FFF0, 2, 32, 32, 32), 3, 8'h22, "P6c STORE past vmem end");
            expect_unit_error(dOP(8'h23, 0, 0), 2, 8'h11, "P6c VEC_LAYERNORM rows=0");
            expect_unit_error(dOP(8'h24, 0, 0), 2, 8'h11, "P6c VEC_LUT rows=0");
            expect_unit_error(dLOAD(0, 4, 0, VB(7), 4, 256, 256, 256), 0, 8'h21, "P6d LOAD SLVERR");
            expect_unit_error(dLOAD(1, 4, 0, VB(7), 4, 256, 256, 256), 3, 8'h21, "P6d STORE SLVERR");
            // (e) CORE_RESET in the middle of transfers; the next programs must be clean
            reset_mid_transfer(0, "P6e LOAD");
            reset_mid_transfer(1, "P6e STORE");
            run_roundtrip("P6e round trip after resets");
        end
    endtask

    // =========================================================================
    // P7: one attention head, Q.K^T and S.V, weights from vmem
    // =========================================================================
    localparam int PB7 = 32'hD000;
    localparam int AT_M = 32, AT_D = 32, AT_S = 16;          // query rows, head dim, keys
    int Qm [0:AT_M-1][0:AT_D-1];
    int Km [0:AT_S-1][0:AT_D-1];
    int Vm [0:AT_S-1][0:N-1];
    int Sm [0:AT_M-1][0:AT_S-1];

    task automatic run_p7();
        reg [31:0] st;
        int r, j, t, acc, v, qv, kv, vv, sv, ov, pv;
        begin
            // DDR inputs (R2) and per-tensor requant table (R1)
            for (r = 0; r < AT_M; r++) for (t = 0; t < AT_D; t++) begin
                Qm[r][t] = $signed(8'($urandom)); ddr_wb(RG2 + 32'h8000 + r * AT_D + t, Qm[r][t]);
            end
            for (j = 0; j < AT_S; j++) for (t = 0; t < AT_D; t++) begin
                Km[j][t] = $signed(8'($urandom)); ddr_wb(RG2 + 32'h9000 + j * AT_D + t, Km[j][t]);
            end
            for (t = 0; t < AT_S; t++) for (j = 0; j < N; j++) begin
                Vm[t][j] = $signed(8'($urandom)); ddr_wb(RG2 + 32'hA000 + t * N + j, Vm[t][j]);
            end
            put_table(RG1 + 32'h300, 37, 39);
            for (r = 0; r < AT_M * 4; r++) u_ddr.mem[(RG3 + 32'h30000) / 16 + r] = 128'hx;

            qv = VB(0) + 32'h10000; kv = VB(1) + 32'h10000; vv = VB(2) + 32'h10000;
            sv = VB(3) + 32'h10000; ov = VB(4) + 32'h10000; pv = VB(5) + 32'h10000;
            pn = 0;
            pd[pn++] = dLOAD(0, 1, 32'h300,  pv, 1, 256, 256, 256);
            pd[pn++] = dLOAD(0, 2, 32'h8000, qv, AT_M, AT_D, AT_D, AT_D);
            pd[pn++] = dLOAD(0, 2, 32'h9000, kv, AT_S, AT_D, AT_D, AT_D);
            pd[pn++] = dLOAD(0, 2, 32'hA000, vv, AT_S, N, N, N);
            pd[pn++] = dw(dWL(kv, AT_D, AT_D, 0, 1'b1), 0, 0);                          // K^T -> half 0
            pd[pn++] = dw(dGEMM(qv, AT_D, AT_M / 16, AT_D, 0), 0, 0);                  // Q . K^T
            pd[pn++] = dw(dEPI(0, 8'b11, 3, -128, 127, AT_M, AT_S, sv, pv), 0, 0);     // S = requant, per tensor
            pd[pn++] = dw(dWL(vv, N, AT_S, 1, 1'b0), 0, 0);                            // V -> half 1
            pd[pn++] = dw(dGEMM(sv, AT_S, AT_M / 16, AT_S, 1), 2, 0);                  // S . V, waits on VECTOR
            pd[pn++] = dEPI(1, 0, 0, 0, 0, AT_M, 64, ov, pv);                          // STORE_I32
            pd[pn++] = dw(dLOAD(1, 3, 32'h30000, ov, AT_M, 64, 64, 64), 2, 0);
            pd[pn++] = dOP(8'hFF, 0, 0);
            emit_single(PB7, 8'h0F);
            run_prog(PB7, st, 2000);
            expect32(st, 32'h3, "P7 STATUS");

            // golden: S = clamp(apply_scale(Q.K^T + bias0) + 3), O = S.V
            for (r = 0; r < AT_M; r++)
                for (j = 0; j < AT_S; j++) begin
                    acc = 0;
                    for (t = 0; t < AT_D; t++) acc += Qm[r][t] * Km[j][t];
                    Sm[r][j] = clamp_ref(apply_scale_ref(acc + bias[0], mult[0], shf[0], 1) + 3, -128, 127);
                end
            v = 0;
            for (r = 0; r < AT_M; r++) for (j = 0; j < AT_S; j++) if (Sm[r][j] > -128 && Sm[r][j] < 127) v++;
            $display("P7: %0d of %0d scores unsaturated", v, AT_M * AT_S);
            for (r = 0; r < AT_M; r++)
                for (j = 0; j < N; j++) begin
                    acc = 0;
                    for (t = 0; t < AT_S; t++) acc += Sm[r][t] * Vm[t][j];
                    v = u_ddr.mem[(RG3 + 32'h30000 + r * 64 + (j / 4) * 16) / 16][(j % 4) * 32 +: 32];
                    if (v !== acc) begin
                        if (errors < 12) $display("MISMATCH P7 O[%0d][%0d]: got %0d exp %0d", r, j, v, acc);
                        errors++;
                    end
                    checked++;
                end
        end
    endtask

    // =========================================================================
    // P8: GEMM throughput (mirrors sw/runtime/tests/mtpu_test.c t_tput)
    // =========================================================================
    localparam int PB8 = 32'hB000;
    localparam int TP_M = 256, TP_K = 512, TP_T = 4;
    localparam int TP_A = 32'h40000, TP_W = 32'h8000, TP_O = 32'h60000;    // R3, R0, R3 offsets
    int Ab [0:TP_M-1][0:TP_K-1];
    int Wb [0:TP_T-1][0:TP_K-1][0:N-1];

    task automatic run_p8();
        reg [31:0] st, cyc32, mb, lb, vb;
        int r, t, j, q, c, exp, got;
        begin
            for (r = 0; r < TP_M; r++) for (t = 0; t < TP_K; t++) begin
                Ab[r][t] = $signed(8'($urandom)); ddr_wb(RG3 + TP_A + r * TP_K + t, Ab[r][t]);
            end
            for (q = 0; q < TP_T; q++) for (t = 0; t < TP_K; t++) for (j = 0; j < N; j++) begin
                Wb[q][t][j] = $signed(8'($urandom)); ddr_wb(RG0 + TP_W + q * TP_K * 16 + t * 16 + j, Wb[q][t][j]);
            end
            put_table(RG1 + 32'h400, 36, 40);
            pn = 0;
            pd[pn++] = dLOAD(0, 1, 32'h400, VB(1) + 32'h400, 1, 256, 256, 256);
            pd[pn++] = dLOAD(0, 3, TP_A, VB(0), TP_M, TP_K, TP_K, TP_K);
            for (q = 0; q < TP_T; q++) begin
                pd[pn++] = dw(dLOADW(0, TP_W + q * TP_K * 16, q % 2, TP_K), 1, 1);
                pd[pn++] = dw(dGEMM(VB(0), TP_K, TP_M / 16, TP_K, q % 2), 0, 0);
                pd[pn++] = dw(dEPI(0, 8'h2, 0, -128, 127, TP_M, TP_T * 16, VB(3) + 16 * q, VB(1) + 32'h400), 0, 0);
            end
            pd[pn++] = dw(dLOAD(1, 3, TP_O, VB(3), TP_M, TP_T * 16, TP_T * 16, TP_T * 16), 2, 0);
            pd[pn++] = dOP(8'hFF, 0, 0);
            emit_single(PB8, 8'h0F);
            run_prog(PB8, st, 20000);
            expect32(st, 32'h3, "P8 STATUS");
            axil_read(R_CYCLES, cyc32); axil_read(R_BUSY + 4, mb); axil_read(R_BUSY + 0, lb); axil_read(R_BUSY + 8, vb);
            $display("P8: %0d cycles, %0.1f MAC/cycle; busy LOAD %0d MATRIX %0d VECTOR %0d (%0.1f MAC/cycle while MATRIX busy)",
                     cyc32, (256.0 * 512 * 64) / cyc32, lb, mb, vb, (256.0 * 512 * 64) / mb);
            for (r = 0; r < TP_M; r += 17)                                  // sampled rows, all columns
                for (q = 0; q < TP_T; q++)
                    for (j = 0; j < N; j++) begin
                        c = 0;
                        for (t = 0; t < TP_K; t++) c += Ab[r][t] * Wb[q][t][j];
                        exp = clamp_ref(apply_scale_ref(c + bias[j], mult[j], shf[j], 1), -128, 127);
                        got = ddr_rb(RG3 + TP_O + r * TP_T * 16 + q * 16 + j);
                        if (got != exp) begin
                            if (errors < 12) $display("MISMATCH P8 [%0d][%0d]: got %0d exp %0d", r, q * 16 + j, got, exp);
                            errors++;
                        end
                        checked++;
                    end
        end
    endtask

    // =========================================================================
    // P9: attention head with softmax (mirrors sw/runtime/tests/mtpu_test.c t_head)
    // =========================================================================
    localparam int PB9 = 32'h9000;
    int EXP2_LUT [0:255];
    initial for (int i = 0; i < 256; i++) EXP2_LUT[i] = $rtoi($floor(65536.0 * $pow(2.0, -i / 256.0) + 0.5));

    function automatic longint exp2neg_ref(input longint z);
        longint zr, ip;
        begin
            zr = z + 128; ip = zr >>> 16;
            exp2neg_ref = (ip > 16) ? 0 : (EXP2_LUT[(zr >>> 8) & 255] >>> ip);
        end
    endfunction

    task automatic real_to_ms(input real r, output int m, output int sh);
        begin
            sh = 30 - $rtoi($floor($ln(r) / $ln(2.0)));
            m  = $rtoi($floor(r * $pow(2.0, sh) + 0.5));
            if (m < 0) begin m = $rtoi($floor(r * $pow(2.0, sh - 1) + 0.5)); sh = sh - 1; end
        end
    endtask

    function automatic [511:0] dSM(input int rows, input int len, input int ip, input int in_a, input int out_a,
                                   input int op, input int in_sh, input int out_sh, input int in_m, input int out_m);
        reg [511:0] d;
        begin
            d = 0; d[7:0] = 8'h22;
            d[208 +: 16] = rows; d[224 +: 16] = len; d[240 +: 16] = ip; d[256 +: 32] = in_a;
            d[288 +: 32] = out_a; d[320 +: 16] = op; d[336 +: 8] = in_sh; d[344 +: 8] = out_sh;
            d[384 +: 32] = in_m; d[448 +: 32] = out_m;            // in_zp = 0, out_zp = 0, i8 in / out
            dSM = d;
        end
    endfunction

    localparam int HD_M = 32, HD_D = 32, HD_S = 32;
    int Qh [0:HD_M-1][0:HD_D-1];
    int Kh [0:HD_S-1][0:HD_D-1];
    int Vh [0:HD_S-1][0:N-1];
    int Sh [0:HD_M-1][0:HD_S-1];
    int Ph [0:HD_M-1][0:HD_S-1];

    task automatic run_p9();
        reg [31:0] st;
        int r, j, t, b, c, in_m, in_sh, out_m, out_sh, got;
        int qv, kv, vv, sv, ppv, ov, tv;
        longint tt [0:HD_S-1];
        longint mx, mnew, bm, sum, bsum, R, pq;
        begin
            for (r = 0; r < HD_M; r++) for (t = 0; t < HD_D; t++) begin Qh[r][t] = $signed(8'($urandom)); ddr_wb(RG2 + 32'hC000 + r * HD_D + t, Qh[r][t]); end
            for (j = 0; j < HD_S; j++) for (t = 0; t < HD_D; t++) begin Kh[j][t] = $signed(8'($urandom)); ddr_wb(RG2 + 32'hD000 + j * HD_D + t, Kh[j][t]); end
            for (t = 0; t < HD_S; t++) for (j = 0; j < N; j++)    begin Vh[t][j] = $signed(8'($urandom)); ddr_wb(RG2 + 32'hE000 + t * N + j, Vh[t][j]); end
            put_table(RG1 + 32'h500, 36, 38);
            for (r = 0; r < HD_M * 4; r++) u_ddr.mem[(RG3 + 32'h68000) / 16 + r] = 128'hx;
            real_to_ms(0.05 * 1.4426950408889634 * 65536.0, in_m, in_sh);
            real_to_ms(127.0 / 65536.0, out_m, out_sh);

            qv = VB(0) + 32'h20000; kv = VB(1) + 32'h20000; vv = VB(2) + 32'h20000; sv = VB(3) + 32'h20000;
            ppv = VB(5) + 32'h20000; ov = VB(4) + 32'h20000; tv = VB(6) + 32'h20000;
            pn = 0;
            pd[pn++] = dLOAD(0, 1, 32'h500, tv, 1, 256, 256, 256);
            pd[pn++] = dLOAD(0, 2, 32'hC000, qv, HD_M, HD_D, HD_D, HD_D);
            pd[pn++] = dLOAD(0, 2, 32'hD000, kv, HD_S, HD_D, HD_D, HD_D);
            pd[pn++] = dLOAD(0, 2, 32'hE000, vv, HD_S, N, N, N);
            for (b = 0; b < 2; b++) begin
                pd[pn++] = dw(dWL(kv + 16 * b * HD_D, HD_D, HD_D, b, 1'b1), 0, 0);
                pd[pn++] = dw(dGEMM(qv, HD_D, HD_M / 16, HD_D, b), 0, 0);
                pd[pn++] = dw(dEPI(0, 8'h3, 0, -128, 127, HD_M, HD_S, sv + 16 * b, tv), 0, 0);
            end
            pd[pn++] = dSM(HD_M, HD_S, HD_S, sv, ppv, HD_S, in_sh, out_sh, in_m, out_m);
            pd[pn++] = dWL(vv, N, HD_S, 0, 1'b0);
            pd[pn++] = dw(dGEMM(ppv, HD_S, HD_M / 16, HD_S, 0), 2, 0);
            pd[pn++] = dEPI(1, 0, 0, 0, 0, HD_M, 64, ov, tv);
            pd[pn++] = dw(dLOAD(1, 3, 32'h68000, ov, HD_M, 64, 64, 64), 2, 0);
            pd[pn++] = dOP(8'hFF, 0, 0);
            emit_single(PB9, 8'h0F);
            run_prog(PB9, st, 4000);
            expect32(st, 32'h3, "P9 STATUS");

            // reference: S = requant(Q.K^T), P = softmax(S) (§5.7 model), O = P.V
            for (r = 0; r < HD_M; r++)
                for (j = 0; j < HD_S; j++) begin
                    c = 0;
                    for (t = 0; t < HD_D; t++) c += Qh[r][t] * Kh[j][t];
                    Sh[r][j] = clamp_ref(apply_scale_ref(c + bias[0], mult[0], shf[0], 1), -128, 127);
                end
            for (r = 0; r < HD_M; r++) begin
                for (j = 0; j < HD_S; j++) tt[j] = apply_scale_ref(Sh[r][j], in_m, in_sh, 0);
                mx = -64'sd2147483648; sum = 0;
                for (b = 0; b < HD_S; b += 16) begin
                    bm = tt[b];
                    for (j = b + 1; j < b + 16; j++) if (tt[j] > bm) bm = tt[j];
                    mnew = (bm > mx) ? bm : mx;
                    bsum = 0;
                    for (j = b; j < b + 16; j++) bsum += exp2neg_ref(mnew - tt[j]);
                    sum = ((sum * exp2neg_ref(mnew - mx)) >>> 16) + bsum;
                    mx = mnew;
                end
                R = (longint'(1) <<< 46) / sum;
                for (j = 0; j < HD_S; j++) begin
                    pq = apply_scale_ref(int'(exp2neg_ref(mx - tt[j])), int'(R), 30, 0);
                    Ph[r][j] = clamp_ref(apply_scale_ref(int'(pq), out_m, out_sh, 0), -128, 127);
                end
            end
            for (r = 0; r < HD_M; r++)
                for (j = 0; j < N; j++) begin
                    c = 0;
                    for (t = 0; t < HD_S; t++) c += Ph[r][t] * Vh[t][j];
                    got = u_ddr.mem[(RG3 + 32'h68000 + r * 64 + (j / 4) * 16) / 16][(j % 4) * 32 +: 32];
                    if (got !== c) begin
                        if (errors < 12) $display("MISMATCH P9 O[%0d][%0d]: got %0d exp %0d", r, j, got, c);
                        errors++;
                    end
                    checked++;
                end
        end
    endtask

    // =========================================================================
    // Test sequence
    // =========================================================================
    reg [31:0] v, st;

    initial begin
        $readmemh("../fw/build/fw.hex", fw);
        repeat (5) @(negedge clk);
        rst_n = 1;
        repeat (5) @(negedge clk);

        // ---- identity, I-mem load, boot ----
        axil_read(R_HW_ID, v);   expect32(v, 32'h5550_544D, "HW_ID");
        axil_read(R_HW_ARCH, v); expect32(v, 32'h0800_0F10, "HW_ARCH");
        axil_read(R_CORE_CTRL, v); expect32(v, 32'h1, "CORE_CTRL reset value");
        for (int i = 0; i < 4096; i++) axil_write(W_IMEM + 4 * i, fw[i]);
        axil_read(W_IMEM + 4 * 17, v); expect32(v, fw[17], "I-mem readback");
        axil_write(R_REGION + 8*0, RG0);
        axil_write(R_REGION + 8*1, RG1);
        axil_write(R_REGION + 8*2, RG2);
        axil_write(R_REGION + 8*3, RG3);
        boot_core();
        // I-mem is read-only to the host while the core runs
        axil_write(W_IMEM + 4 * 17, 32'hDEAD_BEEF);
        axil_read(W_IMEM + 4 * 17, v); expect32(v, fw[17], "I-mem write-protect");
        axil_read(R_FW_HEARTBEAT, v);
        repeat (200) @(negedge clk);
        axil_read(R_FW_HEARTBEAT, st);
        if (st == v) begin $display("FAIL: heartbeat not advancing"); errors++; end
        $display("boot ok, %0d errors", errors);

        // ---- P1 ----
        build_p1();
        run_p1("P1");
        $display("P1 done, %0d checked, %0d errors", checked, errors);

        // ---- P2 ----
        build_p2();
        run_p2();
        $display("P2 done, %0d checked, %0d errors", checked, errors);

        // ---- P3 ----
        build_p3();
        for (int i = 0; i < 32; i++) u_top.u_vmem.bd_write((VB(4) >> 4) + i, 128'd0);   // PASS inputs + params
        run_p3();

        // ---- P4: BREAK / RESUME + IRQ ----
        axil_write(R_IRQ_STATUS, 32'h7);
        axil_write(R_CORE_CTRL, 32'h2);                    // IRQ_EN
        pn = 0;
        pd[pn++] = dLOAD(0, 2, 0, VB(7), 1, 16, 16, 16);
        pd[pn++] = dOP(8'hF2, 16'h55, 0);                  // BREAK 0x55
        pd[pn++] = dOP(8'h00, 0, 0);
        pd[pn++] = dOP(8'hFF, 0, 0);
        emit_single(PB4, 8'h0F);
        run_prog(PB4, st, 400);
        expect32(st, 32'h0055_0002, "P4 STATUS=BREAK");
        if (!irq) begin $display("FAIL: irq not raised on BREAK"); errors++; end
        axil_read(R_IRQ_STATUS, v); expect32(v, 32'h2, "P4 IRQ_STATUS BREAK");
        axil_write(R_IRQ_STATUS, 32'h2);
        axil_write(R_DOORBELL, 2);                          // RESUME
        wait_not_running(st, 400);
        expect32(st, 32'h3, "P4 STATUS=DONE");
        axil_read(R_IRQ_STATUS, v); expect32(v, 32'h1, "P4 IRQ_STATUS DONE");
        axil_read(R_DESC_COUNT, v); expect32(v, 4, "P4 DESC_COUNT");
        axil_read(R_TRACE_COUNT, v); expect32(v, 1, "P4 TRACE_COUNT");
        axil_read(W_TRACE + 4, v);  expect32(v, 32'h0001_0000, "P4 trace chunk/desc");
        axil_read(W_TRACE + 8, v);  expect32(v, 32'h0000_01F2, "P4 trace op/event");
        axil_read(W_TRACE + 12, v); expect32(v, 32'h55, "P4 trace info");
        axil_write(R_IRQ_STATUS, 32'h7);
        if (irq) begin $display("FAIL: irq stuck after W1C"); errors++; end
        axil_write(R_CORE_CTRL, 32'h0);
        $display("P4 done, %0d errors", errors);

        // ---- P5: errors + recovery ----
        // (a) bad magic
        pn = 1; pd[0] = dOP(8'hFF, 0, 0);
        emit_single(PB5, 8'h0F);
        put_header(PB5, 64, 1, 8'h0F, 32'h1234_5678);
        run_prog(PB5, st, 100);
        expect32(st, 32'h0000_0104, "P5a bad magic");
        axil_read(R_ERR_INFO0, v); expect32(v, 32'h1234_5678, "P5a ERR_INFO0");
        boot_core();

        // (b) bad opcode at descriptor 2
        pn = 0;
        pd[pn++] = dOP(8'h00, 0, 0);
        pd[pn++] = dOP(8'h00, 0, 0);
        pd[pn++] = dOP(8'h77, 0, 0);
        pd[pn++] = dOP(8'hFF, 0, 0);
        emit_single(PB5, 8'h0F);
        run_prog(PB5, st, 100);
        expect32(st, 32'h0000_1004, "P5b bad opcode");
        axil_read(R_ERR_DESC, v);  expect32(v, 32'h0000_0002, "P5b ERR_DESC");
        axil_read(R_ERR_INFO0, v); expect32(v, 32'h77, "P5b ERR_INFO0");
        boot_core();

        // (c) required region unset (R5)
        pn = 1; pd[0] = dOP(8'hFF, 0, 0);
        emit_single(PB5, 8'h2F);
        run_prog(PB5, st, 100);
        expect32(st, 32'h0000_0404, "P5c region unset");
        axil_read(R_ERR_INFO0, v); expect32(v, 32'h20, "P5c ERR_INFO0");
        boot_core();

        // (d) unit error: VEC_SOFTMAX with rows = 0 -> VECTOR ERR_BAD_FIELD
        pn = 0;
        pd[pn++] = dOP(8'h22, 0, 0);
        pd[pn++] = dOP(8'hF3, 0, 0);
        pd[pn++] = dOP(8'hFF, 0, 0);
        emit_single(PB5, 8'h0F);
        run_prog(PB5, st, 100);
        expect32(st, 32'h0000_1104, "P5d unit error");
        axil_read(R_ERR_UNIT, v); expect32(v, 2, "P5d ERR_UNIT");
        boot_core();

        // (e) watchdog: GEMM with no consuming epilogue never completes
        axil_write(R_WATCHDOG, 3000);
        pn = 0;
        pd[pn++] = dGEMM(VB(0), 64, 1, 16, 0);
        pd[pn++] = dOP(8'hFF, 0, 0);
        emit_single(PB5, 8'h0F);
        run_prog(PB5, st, 400);
        expect32(st, 32'h0000_3004, "P5e watchdog");
        axil_read(R_ERR_INFO0, v); expect32(v, 32'h0000_0000, "P5e FIFO occupancy (GEMM popped)");
        axil_write(R_WATCHDOG, 0);
        boot_core();                                        // also resets the stuck matrix unit

        // (f) ABORT while waiting at a FENCE behind a long LOAD
        pn = 0;
        pd[pn++] = dLOAD(0, 2, 0, VB(7), 64, 4096, 4096, 4096);   // ~16 K vmem writes
        pd[pn++] = dOP(8'hF3, 0, 0);
        pd[pn++] = dOP(8'hFF, 0, 0);
        emit_single(PB5, 8'h0F);
        axil_write(R_PROG_LO, PB5);
        axil_write(R_DOORBELL, 1);
        repeat (2000) @(negedge clk);
        axil_write(R_DOORBELL, 3);
        wait_not_running(st, 400);
        expect32(st, 32'h0000_3104, "P5f ABORT");
        boot_core();

        // (g) memory op with ddr_region > 7: issue engine stops, firmware reports ERR_BAD_FIELD
        pn = 0;
        pd[pn++] = dLOAD(0, 2, 0, VB(7), 1, 16, 16, 16);
        pd[pn] = dLOAD(0, 2, 0, VB(7), 1, 16, 16, 16); pd[pn][128 +: 8] = 9; pn++;
        pd[pn++] = dOP(8'hFF, 0, 0);
        emit_single(PB5, 8'h0F);
        run_prog(PB5, st, 100);
        expect32(st, 32'h0000_1104, "P5g bad region");
        axil_read(R_ERR_DESC, v);  expect32(v, 32'h0000_0001, "P5g ERR_DESC");
        axil_read(R_ERR_INFO0, v); expect32(v, 32'h10, "P5g ERR_INFO0");
        boot_core();

        // (h) ABORT while the issue engine is blocked on a full VECTOR FIFO:
        //     a long LOAD, then 12 VECTOR commands that wait on it (8 fit in the FIFO)
        pn = 0;
        pd[pn++] = dLOAD(0, 2, 0, VB(7), 64, 4096, 4096, 4096);
        for (int i = 0; i < 12; i++) pd[pn++] = dw(dPASS(VB(4), VB(6), VB(4) + 32'h100), 0, 0);
        pd[pn++] = dOP(8'hFF, 0, 0);
        emit_single(PB5, 8'h0F);
        axil_write(R_PROG_LO, PB5);
        axil_write(R_DOORBELL, 1);
        repeat (3000) @(negedge clk);
        if (free[2*4 +: 4] != 0 || !u_top.u_scalar.ise_busy) begin
            $display("FAIL P5h: issue engine not blocked (VECTOR free=%0d, busy=%0b)", free[2*4 +: 4], u_top.u_scalar.ise_busy);
            errors++;
        end
        axil_write(R_DOORBELL, 3);
        wait_not_running(st, 400);
        expect32(st, 32'h0000_3104, "P5h ABORT during issue run");
        if (u_top.u_scalar.ise_busy) begin $display("FAIL P5h: issue engine still busy"); errors++; end
        boot_core();
        $display("P5 done, %0d errors", errors);

        // ---- P6: DMA ----
        run_p6();
        $display("P6 done, %0d checked, %0d errors", checked, errors);

        // ---- P7: attention head with weights from vmem ----
        run_p7();
        $display("P7 done, %0d checked, %0d errors", checked, errors);

        // ---- P8: GEMM throughput ----
        run_p8();
        $display("P8 done, %0d checked, %0d errors", checked, errors);

        // ---- P9: attention head with softmax ----
        run_p9();
        $display("P9 done, %0d checked, %0d errors", checked, errors);

        // ---- P1 again after all the resets ----
        build_p1();
        run_p1("P1 after recovery");

        axil_read(R_HW_ID, v);
        if (u_ddr.violations != 0) begin
            $display("FAIL: %0d AXI protocol violations seen by the DDR model", u_ddr.violations);
            errors++;
        end
        if (errors == 0) $display("TB PASS: %0d values checked", checked);
        else             $display("TB FAIL: %0d errors (%0d checked)", errors, checked);
        $finish;
    end

    initial begin
        #200_000_000;
        $display("TB FAIL: global timeout");
        $finish;
    end

endmodule
