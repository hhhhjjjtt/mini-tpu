`timescale 1ns/1ps

// =============================================================================
// scalar_unit — PicoRV32 control unit + issue engine + host interface
// =============================================================================
//
// INTERFACE_SPEC.md §7 (ARM side) and §8 (RISC-V side). Contents:
//
//   * PicoRV32 (RV32I) running the thin-router firmware (hw/fw/).
//   * I-mem 16 KB, D-mem 8 KB, descriptor buffer 2 x 16 KB, trace ring 4 KB.
//   * Unit ports: command staging + PUSH into the unit front ends, FREE / DONE /
//     UNIT_STATUS readback.
//   * Fetch DMA: AXI4 read master (common/queue_axi4_master_r.sv) filling a
//     descriptor bank.
//   * Mailbox: registers shared with the ARM (doorbell, status, regions, ...).
//   * Timer, trace ring, perf counters, watchdog, PL->PS IRQ.
//
// Core bus: zero-wait for memories using the PicoRV32 look-ahead interface
// (BRAM read issued on mem_la_read, data valid when mem_valid rises); IO
// registers answer one cycle after mem_valid.
//
// Host bus: AXI-Lite slave, one transaction at a time, 3-cycle read latency.
//
// CORE_CTRL.CORE_RESET (reset value 1) holds the core, the unit front ends and
// the units (units_rst_n) in reset, and clears core-owned mailbox state.
// I-mem is host-writable only while CORE_RESET = 1.
//
// =============================================================================

module scalar_unit #(
    parameter integer NUM_UNITS  = 4,           // fixed 4 (register layouts)
    parameter integer CNT_W      = 24,
    parameter integer FREE_W     = 4,
    parameter integer FIFO_DEPTH = 8,
    parameter integer CMD_W      = 512,
    parameter integer AXI_AW     = 40,
    parameter [31:0]  HW_VERSION = 32'h0002_0000,
    parameter [31:0]  HW_ARCH    = 32'h0800_0F10   // vmem 2048 KB, units 0x0F, N 16
)(
    input  wire                         clk,
    input  wire                         rst_n,

    // ---- AXI-Lite slave (ARM), 64 KB window ----
    input  wire [15:0]                  s_axil_awaddr,
    input  wire                         s_axil_awvalid,
    output wire                         s_axil_awready,
    input  wire [31:0]                  s_axil_wdata,
    input  wire [3:0]                   s_axil_wstrb,
    input  wire                         s_axil_wvalid,
    output wire                         s_axil_wready,
    output wire [1:0]                   s_axil_bresp,
    output reg                          s_axil_bvalid,
    input  wire                         s_axil_bready,
    input  wire [15:0]                  s_axil_araddr,
    input  wire                         s_axil_arvalid,
    output wire                         s_axil_arready,
    output reg  [31:0]                  s_axil_rdata,
    output wire [1:0]                   s_axil_rresp,
    output reg                          s_axil_rvalid,
    input  wire                         s_axil_rready,

    output wire                         irq,            // PL -> PS

    // ---- AXI4 read master (descriptor fetch) ----
    output wire [0:0]                   m_axi_arid,
    output wire [AXI_AW-1:0]            m_axi_araddr,
    output wire [7:0]                   m_axi_arlen,
    output wire [2:0]                   m_axi_arsize,
    output wire [1:0]                   m_axi_arburst,
    output wire [3:0]                   m_axi_arcache,
    output wire                         m_axi_arvalid,
    input  wire                         m_axi_arready,
    input  wire [0:0]                   m_axi_rid,
    input  wire [127:0]                 m_axi_rdata,
    input  wire [1:0]                   m_axi_rresp,
    input  wire                         m_axi_rlast,
    input  wire                         m_axi_rvalid,
    output wire                         m_axi_rready,

    // ---- Units ----
    output reg                          units_rst_n,    // front ends + units
    output reg                          clear_done,     // DONE counters -> 0
    output wire [NUM_UNITS-1:0]         push,
    output wire [CMD_W-1:0]             push_cmd,
    input  wire [NUM_UNITS*FREE_W-1:0]  free,
    input  wire [NUM_UNITS*CNT_W-1:0]   done_counts,
    input  wire [NUM_UNITS-1:0]         unit_done,
    input  wire [NUM_UNITS-1:0]         unit_busy,
    input  wire [NUM_UNITS-1:0]         unit_err,
    input  wire [NUM_UNITS*8-1:0]       unit_err_code,
    input  wire [NUM_UNITS-1:0]         wait_stall,
    input  wire [NUM_UNITS-1:0]         starved
);

    // =========================================================================
    // Constants
    // =========================================================================
    localparam [31:0] HW_ID = 32'h5550_544D;            // "MTPU"

    localparam [3:0] ST_IDLE = 4'd0, ST_RUNNING = 4'd1, ST_BREAK = 4'd2,
                     ST_DONE = 4'd3, ST_ERROR   = 4'd4;

    localparam [1:0] DB_RUN = 2'd1, DB_RESUME = 2'd2, DB_ABORT = 2'd3;

    localparam [7:0] ERR_WATCHDOG  = 8'h30;
    localparam [7:0] ERR_CORE_TRAP = 8'h32;

    // =========================================================================
    // Mailbox / control registers
    // =========================================================================
    reg         core_reset;             // CORE_CTRL[0]
    reg         irq_en;                 // CORE_CTRL[1]
    reg  [1:0]  doorbell;               // pending command, 0 = none
    reg  [31:0] status;
    reg  [2:0]  irq_status;
    reg  [63:0] prog_base;
    reg  [31:0] watchdog;
    reg  [63:0] region [0:7];
    reg  [31:0] fw_version, fw_arch, fw_heartbeat;
    reg  [31:0] err_desc, err_unit, err_info0, err_info1;
    reg  [31:0] desc_count;
    reg  [63:0] perf_cycles;
    reg  [31:0] perf_busy [0:3], perf_wait [0:3], perf_starved [0:3];
    reg  [31:0] trace_count;
    reg         wd_fired, trap_seen;

    wire running = (status[3:0] == ST_RUNNING);

    assign irq = irq_en && |irq_status;

    // =========================================================================
    // Host AXI-Lite slave -> host bus (hb_*)
    // =========================================================================
    reg         aw_held, w_held;
    reg  [15:0] aw_addr_q;
    reg  [31:0] w_data_q;
    reg  [3:0]  w_strb_q;
    reg         rd_s1, rd_s2;           // read pipeline stages
    reg  [15:0] ar_addr_q;

    assign s_axil_awready = !aw_held && !s_axil_bvalid;
    assign s_axil_wready  = !w_held  && !s_axil_bvalid;
    assign s_axil_arready = !rd_s1 && !rd_s2 && !s_axil_rvalid;
    assign s_axil_bresp   = 2'b00;
    assign s_axil_rresp   = 2'b00;

    wire        hb_we    = aw_held && w_held && !s_axil_bvalid;
    wire [15:0] hb_waddr = aw_addr_q;
    wire [31:0] hb_wdata = w_data_q;
    wire [3:0]  hb_wstrb = w_strb_q;
    wire        hb_re    = rd_s1;       // address valid on ar_addr_q
    wire [15:0] hb_raddr = ar_addr_q;

    // Host windows (§7.5)
    wire hb_w_regs  = (hb_waddr[15:12] == 4'h0);
    wire hb_w_dmem  = (hb_waddr[15:13] == 3'b010);         // 0x4000-0x5FFF
    wire hb_w_imem  = (hb_waddr[15:14] == 2'b10);          // 0x8000-0xBFFF
    wire hb_r_regs  = (hb_raddr[15:12] == 4'h0);
    wire hb_r_trace = (hb_raddr[15:12] == 4'h2);
    wire hb_r_dmem  = (hb_raddr[15:13] == 3'b010);
    wire hb_r_imem  = (hb_raddr[15:14] == 2'b10);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            aw_held <= 1'b0; w_held <= 1'b0; s_axil_bvalid <= 1'b0;
            aw_addr_q <= 0; w_data_q <= 0; w_strb_q <= 0;
            rd_s1 <= 1'b0; rd_s2 <= 1'b0; ar_addr_q <= 0;
        end else begin
            if (s_axil_awvalid && s_axil_awready) begin aw_held <= 1'b1; aw_addr_q <= s_axil_awaddr; end
            if (s_axil_wvalid  && s_axil_wready)  begin w_held  <= 1'b1; w_data_q <= s_axil_wdata; w_strb_q <= s_axil_wstrb; end
            if (hb_we) begin aw_held <= 1'b0; w_held <= 1'b0; s_axil_bvalid <= 1'b1; end
            else if (s_axil_bvalid && s_axil_bready) s_axil_bvalid <= 1'b0;

            if (s_axil_arvalid && s_axil_arready) begin rd_s1 <= 1'b1; ar_addr_q <= s_axil_araddr; end
            else rd_s1 <= 1'b0;
            rd_s2 <= rd_s1;
        end
    end

    // =========================================================================
    // PicoRV32
    // =========================================================================
    wire        core_resetn = rst_n && !core_reset;
    wire        trap;
    wire        mem_valid, mem_instr;
    wire        mem_ready;
    wire [31:0] mem_addr, mem_wdata;
    wire [3:0]  mem_wstrb;
    wire [31:0] mem_rdata;
    wire        mem_la_read, mem_la_write;
    wire [31:0] mem_la_addr;
    wire [31:0] core_irq;

    picorv32 #(
        .ENABLE_COUNTERS     (0),
        .ENABLE_COUNTERS64   (0),
        .ENABLE_REGS_DUALPORT(1),
        .BARREL_SHIFTER      (1),
        .COMPRESSED_ISA      (0),
        .CATCH_MISALIGN      (1),
        .CATCH_ILLINSN       (1),
        .ENABLE_MUL          (0),
        .ENABLE_DIV          (0),
        .ENABLE_IRQ          (1),
        .ENABLE_IRQ_QREGS    (0),
        .ENABLE_IRQ_TIMER    (0),
        .PROGADDR_RESET      (32'h0000_0000),
        .PROGADDR_IRQ        (32'h0000_0010),
        .STACKADDR           (32'h0001_2000)
    ) u_core (
        .clk(clk), .resetn(core_resetn), .trap(trap),
        .mem_valid(mem_valid), .mem_instr(mem_instr), .mem_ready(mem_ready),
        .mem_addr(mem_addr), .mem_wdata(mem_wdata), .mem_wstrb(mem_wstrb), .mem_rdata(mem_rdata),
        .mem_la_read(mem_la_read), .mem_la_write(mem_la_write), .mem_la_addr(mem_la_addr),
        .mem_la_wdata(), .mem_la_wstrb(),
        .pcpi_valid(), .pcpi_insn(), .pcpi_rs1(), .pcpi_rs2(),
        .pcpi_wr(1'b0), .pcpi_rd(32'd0), .pcpi_wait(1'b0), .pcpi_ready(1'b0),
        .irq(core_irq), .eoi(),
        .trace_valid(), .trace_data()
    );

    // ---- Core address decode (§8) ----
    function automatic [2:0] region_of(input [31:0] a);
        // 0 none, 1 I-mem, 2 D-mem, 3 descriptor banks, 4 IO
        if      (a[31:14] == 18'h0)               region_of = 3'd1;   // 0x0000_0000 16 KB
        else if (a[31:13] == 19'h8)               region_of = 3'd2;   // 0x0001_0000 8 KB
        else if (a[31:15] == 17'h4)               region_of = 3'd3;   // 0x0002_0000 32 KB
        else if (a[31:16] == 16'h1000)            region_of = 3'd4;   // 0x1000_xxxx
        else                                      region_of = 3'd0;
    endfunction

    wire [2:0] la_reg  = region_of(mem_la_addr);
    wire [2:0] cur_reg = region_of(mem_addr);
    wire       cur_mem = (cur_reg == 3'd1) || (cur_reg == 3'd2) || (cur_reg == 3'd3);
    wire       core_we = mem_valid && (|mem_wstrb);

    reg        io_ack;
    reg [31:0] io_rdata;

    // memories answer in the mem_valid cycle; IO and unmapped one cycle later
    assign mem_ready = mem_valid && (cur_mem ? 1'b1 : io_ack);
    wire   io_svc    = mem_valid && !cur_mem && !io_ack;     // one service edge per IO access

    // =========================================================================
    // Memories
    // =========================================================================
    // I-mem: 4096 x 32. Port A core read (look-ahead), port B host RW.
    reg [31:0] imem [0:4095];
    reg [31:0] imem_qa, imem_qb;
    always @(posedge clk) begin
        if (mem_la_read && la_reg == 3'd1)
            imem_qa <= imem[mem_la_addr[13:2]];
    end
    integer bi;
    always @(posedge clk) begin
        if (hb_we && hb_w_imem && core_reset)
            for (bi = 0; bi < 4; bi = bi + 1)
                if (hb_wstrb[bi]) imem[hb_waddr[13:2]][bi*8 +: 8] <= hb_wdata[bi*8 +: 8];
        if (hb_re && hb_r_imem)
            imem_qb <= imem[hb_raddr[13:2]];
    end

    // D-mem: 2048 x 32. Port A core RW, port B host RW (debug).
    reg [31:0] dmem [0:2047];
    reg [31:0] dmem_qa, dmem_qb;
    integer bj;
    always @(posedge clk) begin
        if (mem_la_read && la_reg == 3'd2)
            dmem_qa <= dmem[mem_la_addr[12:2]];
        if (core_we && cur_reg == 3'd2)
            for (bj = 0; bj < 4; bj = bj + 1)
                if (mem_wstrb[bj]) dmem[mem_addr[12:2]][bj*8 +: 8] <= mem_wdata[bj*8 +: 8];
    end
    integer bk;
    always @(posedge clk) begin
        if (hb_we && hb_w_dmem)
            for (bk = 0; bk < 4; bk = bk + 1)
                if (hb_wstrb[bk]) dmem[hb_waddr[12:2]][bk*8 +: 8] <= hb_wdata[bk*8 +: 8];
        if (hb_re && hb_r_dmem)
            dmem_qb <= dmem[hb_raddr[12:2]];
    end

    // Descriptor banks: 2048 x 128 (bank = index[10]). Write: fetch DMA.
    // Read: issue engine (§8.7) or core.
    wire         ise_rd_en;
    wire [10:0]  ise_rd_addr;
    reg  [127:0] desc_mem [0:2047];
    reg  [127:0] desc_q;
    wire         fd_beat;
    wire [9:0]   fd_beat_idx;
    wire [127:0] fd_beat_data;
    reg          f_bank;
    // one read port shared by the issue engine (priority) and the core look-ahead;
    // a single address/enable keeps it a simple dual-port BRAM
    wire         desc_re    = ise_rd_en || (mem_la_read && la_reg == 3'd3);
    wire [10:0]  desc_raddr = ise_rd_en ? ise_rd_addr : mem_la_addr[14:4];
    always @(posedge clk) begin
        if (fd_beat)
            desc_mem[{f_bank, fd_beat_idx}] <= fd_beat_data;
        if (desc_re)
            desc_q <= desc_mem[desc_raddr];
    end

    // Trace ring: 256 x 128. Write: core TRACE_PUSH. Read: host.
    reg  [127:0] trace_mem [0:255];
    reg  [127:0] trace_q;
    reg  [31:0]  trace_w [0:3];
    reg  [7:0]   trace_ptr;
    reg          trace_push;
    always @(posedge clk) begin
        if (trace_push)
            trace_mem[trace_ptr] <= {trace_w[3], trace_w[2], trace_w[1], trace_w[0]};
        if (hb_re && hb_r_trace)
            trace_q <= trace_mem[hb_raddr[11:4]];
    end

    // ---- Core read mux ----
    assign mem_rdata = (cur_reg == 3'd1) ? imem_qa :
                       (cur_reg == 3'd2) ? dmem_qa :
                       (cur_reg == 3'd3) ? desc_q[mem_addr[3:2]*32 +: 32] :
                                           io_rdata;

    // =========================================================================
    // IO space (§8.1-§8.4)
    // =========================================================================
    wire [3:0]  io_blk  = mem_addr[15:12];      // 0 unit ports, 2 fetch, 3 mailbox, 4 timer
    wire [11:0] io_off  = mem_addr[11:0];
    wire [1:0]  io_unit = mem_addr[9:8];
    wire [7:0]  io_uoff = mem_addr[7:0];

    // ---- Unit ports: shared staging (W0..W15) ----
    reg [31:0]      staging [0:15];
    reg [3:0]       fw_push;
    wire [CMD_W-1:0] staging_cmd;
    genvar gs;
    generate
        for (gs = 0; gs < 16; gs = gs + 1) begin : g_stage
            assign staging_cmd[gs*32 +: 32] = staging[gs];
        end
    endgenerate

    // ---- Issue engine (§8.7) ----
    wire         ise_busy;
    wire [2:0]   ise_reason;
    wire [7:0]   ise_stop_op;
    wire [8:0]   ise_n, ise_pos;
    wire [3:0]   ise_push;
    wire [511:0] ise_cmd;
    reg          ise_go, ise_stop;
    reg  [8:0]   ise_start, ise_count;
    reg  [31:0]  loop_add [0:7];
    reg  [CNT_W-1:0] issued [0:3];
    wire [8*64-1:0]  region_flat;
    wire [8*32-1:0]  loop_add_flat;
    wire [4*CNT_W-1:0] issued_flat;

    genvar gr;
    generate
        for (gr = 0; gr < 8; gr = gr + 1) begin : g_rflat
            assign region_flat[gr*64 +: 64]   = region[gr];
            assign loop_add_flat[gr*32 +: 32] = loop_add[gr];
        end
        for (gr = 0; gr < 4; gr = gr + 1) begin : g_iflat
            assign issued_flat[gr*CNT_W +: CNT_W] = issued[gr];
        end
    endgenerate

    issue_engine #(.FREE_W(FREE_W), .CNT_W(CNT_W)) u_ise (
        .clk(clk), .rst_n(core_resetn),
        .go(ise_go), .stop(ise_stop), .start_pos(ise_start), .count(ise_count),
        .busy(ise_busy), .reason(ise_reason), .stop_op(ise_stop_op), .n_issued(ise_n), .pos(ise_pos),
        .rd_en(ise_rd_en), .rd_addr(ise_rd_addr), .rd_data(desc_q),
        .region_base(region_flat), .loop_add(loop_add_flat),
        .issued(issued_flat), .free(free),
        .push(ise_push), .cmd(ise_cmd)
    );

    assign push     = fw_push | ise_push;
    assign push_cmd = (ise_push != 0) ? ise_cmd : staging_cmd;

    // issued[u]: every command pushed to unit u since the last clear (§4.2)
    integer iu;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (iu = 0; iu < 4; iu = iu + 1) issued[iu] <= 0;
        end else if (clear_done || core_reset) begin
            for (iu = 0; iu < 4; iu = iu + 1) issued[iu] <= 0;
        end else begin
            for (iu = 0; iu < 4; iu = iu + 1)
                if (push[iu]) issued[iu] <= issued[iu] + 1'b1;
        end
    end

    // ---- Fetch DMA ----
    reg  [63:0] f_src;
    reg  [14:0] f_len;
    reg         f_start;
    reg         f_done, f_axi_err;
    wire        f_busy, f_done_pulse;

    queue_axi4_master_r #(
        .ADDR_W(AXI_AW), .DATA_W(128), .LENGTH_W(15), .BEAT_BYTES(16),
        .MAX_OUTSTANDING(4), .BRAM_ADDR_W(10)
    ) u_fetch (
        .clk(clk), .rst_n(rst_n),
        .start(f_start), .src_addr(f_src[AXI_AW-1:0]), .length_bytes(f_len),
        .busy(f_busy), .done(f_done_pulse),
        .m_axi_arid(m_axi_arid), .m_axi_araddr(m_axi_araddr), .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize), .m_axi_arburst(m_axi_arburst), .m_axi_arcache(m_axi_arcache),
        .m_axi_ardomain(), .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
        .m_axi_rid(m_axi_rid), .m_axi_rdata(m_axi_rdata), .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid(m_axi_rvalid), .m_axi_rready(m_axi_rready),
        .slot_writer_busy(1'b1), .slot_beat_wr_valid(fd_beat),
        .bram_wr_addr(fd_beat_idx), .bram_wr_data(fd_beat_data)
    );

    wire f_rerr = m_axi_rvalid && m_axi_rready && m_axi_rresp[1];

    // ---- Timer ----
    reg [63:0] cycle_ctr;
    reg [31:0] cycle_hi_latched;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) cycle_ctr <= 0;
        else        cycle_ctr <= cycle_ctr + 1'b1;
    end

    // ---- Summary word (mailbox 0x404) ----
    wire [31:0] unit_summary = {15'd0, (doorbell != 2'd0),
                                3'd0, wd_fired, starved, unit_busy, unit_err};

    // ---- Core IRQs (§8.6): masked at reset, firmware polls ----
    assign core_irq = {26'd0, |unit_err, (f_done || f_axi_err), (doorbell != 2'd0), 3'd0};

    // =========================================================================
    // Register read helpers
    // =========================================================================
    function automatic [31:0] unit_status_word(input integer u);
        unit_status_word = {16'd0, unit_err_code[u*8 +: 8], 6'd0, unit_err[u], unit_busy[u]};
    endfunction

    // shared register file read (host and mailbox use the same offsets, §8.3)
    function automatic [31:0] shared_rd(input [11:0] off);
        begin
            shared_rd = 32'd0;
            casez (off)
                12'h000: shared_rd = HW_ID;
                12'h004: shared_rd = HW_VERSION;
                12'h008: shared_rd = HW_ARCH;
                12'h010: shared_rd = {30'd0, irq_en, core_reset};
                12'h018: shared_rd = status;
                12'h01C: shared_rd = {29'd0, irq_status};
                12'h020: shared_rd = prog_base[31:0];
                12'h024: shared_rd = prog_base[63:32];
                12'h030: shared_rd = watchdog;
                12'b0001_00??_?000: shared_rd = region[off[5:3]][31:0];
                12'b0001_00??_?100: shared_rd = region[off[5:3]][63:32];
                12'h200: shared_rd = fw_version;
                12'h204: shared_rd = fw_arch;
                12'h208: shared_rd = fw_heartbeat;
                12'h210: shared_rd = err_desc;
                12'h214: shared_rd = err_unit;
                12'h218: shared_rd = err_info0;
                12'h21C: shared_rd = err_info1;
                12'h300: shared_rd = perf_cycles[31:0];
                12'h304: shared_rd = perf_cycles[63:32];
                12'h310, 12'h314, 12'h318, 12'h31C: shared_rd = perf_busy[off[3:2]];
                12'h320, 12'h324, 12'h328, 12'h32C: shared_rd = perf_wait[off[3:2]];
                12'h330, 12'h334, 12'h338, 12'h33C: shared_rd = perf_starved[off[3:2]];
                12'h340: shared_rd = desc_count;
                12'h344: shared_rd = trace_count;
                12'h404: shared_rd = unit_summary;
                default: shared_rd = 32'd0;
            endcase
        end
    endfunction

    // =========================================================================
    // Main register process: host writes, core IO, hardware events
    // =========================================================================
    reg [31:0] host_reg_q;
    integer r, u;

    wire host_db_wr = hb_we && hb_w_regs && (hb_waddr[11:0] == 12'h014);
    wire core_db_rd = io_svc && cur_reg == 3'd4 && !core_we && io_blk == 4'h3 && io_off == 12'h014;

    // watchdog: cycles with outstanding work and no unit completing
    reg  [31:0] wd_ctr;
    wire        work_pending = (~starved != 0) || (unit_busy != 0);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            core_reset <= 1'b1; irq_en <= 1'b0; doorbell <= 2'd0;
            status <= 0; irq_status <= 0; prog_base <= 0; watchdog <= 0;
            for (r = 0; r < 8; r = r + 1) region[r] <= 0;
            fw_version <= 0; fw_arch <= 0; fw_heartbeat <= 0;
            err_desc <= 0; err_unit <= 0; err_info0 <= 0; err_info1 <= 0;
            desc_count <= 0; perf_cycles <= 0; trace_count <= 0;
            for (u = 0; u < 4; u = u + 1) begin perf_busy[u] <= 0; perf_wait[u] <= 0; perf_starved[u] <= 0; end
            wd_fired <= 1'b0; trap_seen <= 1'b0; wd_ctr <= 0;
            units_rst_n <= 1'b0; clear_done <= 1'b0; fw_push <= 0;
            for (r = 0; r < 16; r = r + 1) staging[r] <= 0;
            for (r = 0; r < 4; r = r + 1) trace_w[r] <= 0;
            trace_ptr <= 0; trace_push <= 1'b0;
            f_src <= 0; f_len <= 0; f_bank <= 1'b0; f_start <= 1'b0; f_done <= 1'b0; f_axi_err <= 1'b0;
            ise_go <= 1'b0; ise_stop <= 1'b0; ise_start <= 0; ise_count <= 0;
            for (r = 0; r < 8; r = r + 1) loop_add[r] <= 0;
            io_ack <= 1'b0; io_rdata <= 0; cycle_hi_latched <= 0; host_reg_q <= 0;
        end else begin
            units_rst_n <= !core_reset;
            clear_done  <= 1'b0;
            fw_push     <= 0;
            trace_push  <= 1'b0;
            f_start     <= 1'b0;
            ise_go      <= 1'b0;
            ise_stop    <= 1'b0;
            io_ack      <= io_svc;

            // ---------------- perf counters ----------------
            if (running) begin
                perf_cycles <= perf_cycles + 1'b1;
                for (u = 0; u < 4; u = u + 1) begin
                    if (unit_busy[u])  perf_busy[u]    <= perf_busy[u] + 1'b1;
                    if (wait_stall[u]) perf_wait[u]    <= perf_wait[u] + 1'b1;
                    if (starved[u] && !unit_busy[u]) perf_starved[u] <= perf_starved[u] + 1'b1;
                end
            end

            // ---------------- fetch DMA status ----------------
            if (f_done_pulse) f_done    <= 1'b1;
            if (f_rerr)       f_axi_err <= 1'b1;

            // ---------------- core IO ----------------
            if (io_svc) begin
                io_rdata <= 32'd0;
                if (cur_reg == 3'd4) case (io_blk)
                    // unit ports
                    4'h0: if (mem_addr[11:10] == 2'b00) begin
                        if (core_we) begin
                            if (io_uoff < 8'h40)       staging[io_uoff[5:2]] <= mem_wdata;
                            else if (io_uoff == 8'h40) fw_push[io_unit] <= 1'b1;
                        end else begin
                            case (io_uoff)
                                8'h44: io_rdata <= {{(32-FREE_W){1'b0}}, free[io_unit*FREE_W +: FREE_W]};
                                8'h48: io_rdata <= {{(32-CNT_W){1'b0}}, done_counts[io_unit*CNT_W +: CNT_W]};
                                8'h4C: io_rdata <= unit_status_word(io_unit);
                                default: ;
                            endcase
                        end
                    end
                    // fetch DMA
                    4'h2: begin
                        if (core_we) begin
                            case (io_off)
                                12'h000: f_src[31:0]  <= mem_wdata;
                                12'h004: f_src[63:32] <= mem_wdata;
                                12'h008: f_len        <= mem_wdata[14:0];
                                12'h00C: f_bank       <= mem_wdata[0];
                                12'h010: if (mem_wdata[0] && !f_busy) begin
                                             f_start <= 1'b1; f_done <= 1'b0;
                                         end
                                12'h014: begin
                                             if (mem_wdata[1]) f_done    <= 1'b0;
                                             if (mem_wdata[2]) f_axi_err <= 1'b0;
                                         end
                                default: ;
                            endcase
                        end else begin
                            case (io_off)
                                12'h000: io_rdata <= f_src[31:0];
                                12'h004: io_rdata <= f_src[63:32];
                                12'h008: io_rdata <= {17'd0, f_len};
                                12'h00C: io_rdata <= {31'd0, f_bank};
                                12'h014: io_rdata <= {29'd0, f_axi_err, f_done, f_busy || f_start};
                                default: ;
                            endcase
                        end
                    end
                    // mailbox
                    4'h3: begin
                        if (core_we) begin
                            case (io_off)
                                12'h018: begin
                                    status <= mem_wdata;
                                    case (mem_wdata[3:0])
                                        ST_DONE:  irq_status[0] <= 1'b1;
                                        ST_BREAK: irq_status[1] <= 1'b1;
                                        ST_ERROR: irq_status[2] <= 1'b1;
                                        default: ;
                                    endcase
                                end
                                12'h200: fw_version   <= mem_wdata;
                                12'h204: fw_arch      <= mem_wdata;
                                12'h208: fw_heartbeat <= mem_wdata;
                                12'h210: err_desc     <= mem_wdata;
                                12'h214: err_unit     <= mem_wdata;
                                12'h218: err_info0    <= mem_wdata;
                                12'h21C: err_info1    <= mem_wdata;
                                12'h340: desc_count   <= mem_wdata;
                                12'h400: if (mem_wdata[0]) clear_done <= 1'b1;
                                default: ;
                            endcase
                        end else begin
                            if (io_off == 12'h014) io_rdata <= {30'd0, doorbell};
                            else                   io_rdata <= shared_rd(io_off);
                        end
                    end
                    // issue engine
                    4'h5: begin
                        if (core_we) begin
                            casez (io_off)
                                12'h000: begin ise_go <= mem_wdata[0]; ise_stop <= mem_wdata[1]; end
                                12'h004: ise_start <= mem_wdata[8:0];
                                12'h008: ise_count <= mem_wdata[8:0];
                                12'b0000_010?_??00: loop_add[io_off[4:2]] <= mem_wdata;
                                default: ;
                            endcase
                        end else begin
                            casez (io_off)
                                12'h004: io_rdata <= {23'd0, ise_pos};
                                12'h008: io_rdata <= {23'd0, ise_count};
                                12'h00C: io_rdata <= {7'd0, ise_n, ise_stop_op, 4'd0, ise_reason, ise_busy || ise_go};
                                12'h010, 12'h014, 12'h018, 12'h01C:
                                         io_rdata <= {{(32-CNT_W){1'b0}}, issued[io_off[3:2]]};
                                12'b0000_010?_??00: io_rdata <= loop_add[io_off[4:2]];
                                default: ;
                            endcase
                        end
                    end
                    // timer / trace
                    4'h4: begin
                        if (core_we) begin
                            case (io_off)
                                12'h010, 12'h014, 12'h018, 12'h01C: trace_w[io_off[3:2]] <= mem_wdata;
                                12'h020: begin
                                    trace_push  <= 1'b1;
                                    trace_count <= trace_count + 1'b1;
                                end
                                default: ;
                            endcase
                        end else begin
                            case (io_off)
                                12'h000: begin io_rdata <= cycle_ctr[31:0]; cycle_hi_latched <= cycle_ctr[63:32]; end
                                12'h004: io_rdata <= cycle_hi_latched;
                                default: ;
                            endcase
                        end
                    end
                    default: ;
                endcase
            end
            if (trace_push) trace_ptr <= trace_ptr + 1'b1;

            // doorbell read by firmware clears it (a host write in the same cycle wins)
            if (core_db_rd) doorbell <= 2'd0;

            // ---------------- watchdog ----------------
            if (!running || watchdog == 0 || !work_pending || unit_done != 0)
                wd_ctr <= 0;
            else
                wd_ctr <= wd_ctr + 1'b1;
            if (running && watchdog != 0 && wd_ctr == watchdog && !wd_fired) begin
                wd_fired  <= 1'b1;
                status    <= {16'd0, ERR_WATCHDOG, 4'd0, ST_ERROR};
                err_unit  <= 32'hFF;
                for (u = 0; u < 4; u = u + 1)
                    err_info0[u*8 +: 8] <= 8'(FIFO_DEPTH) - 8'(free[u*FREE_W +: FREE_W]);
                irq_status[2] <= 1'b1;
            end

            // ---------------- core trap ----------------
            if (trap && !trap_seen && !core_reset) begin
                trap_seen     <= 1'b1;
                status        <= {16'd0, ERR_CORE_TRAP, 4'd0, ST_ERROR};
                err_unit      <= 32'hFF;
                irq_status[2] <= 1'b1;
            end

            // ---------------- host writes ----------------
            if (hb_we && hb_w_regs) begin
                casez (hb_waddr[11:0])
                    12'h010: core_reset <= hb_wdata[0];
                    12'h014: if (hb_wdata[1:0] != 2'd0) begin
                        doorbell <= hb_wdata[1:0];
                        if (hb_wdata[1:0] == DB_RUN || hb_wdata[1:0] == DB_RESUME)
                            status <= {28'd0, ST_RUNNING};       // visible before firmware reacts
                        if (hb_wdata[1:0] == DB_RUN) begin
                            perf_cycles <= 0; desc_count <= 0; trace_count <= 0; trace_ptr <= 0; wd_fired <= 1'b0;
                            for (u = 0; u < 4; u = u + 1) begin perf_busy[u] <= 0; perf_wait[u] <= 0; perf_starved[u] <= 0; end
                        end
                    end
                    12'h01C: irq_status <= irq_status & ~hb_wdata[2:0];
                    12'h020: prog_base[31:0]  <= hb_wdata;
                    12'h024: prog_base[63:32] <= hb_wdata;
                    12'h030: watchdog <= hb_wdata;
                    12'b0001_00??_?000: region[hb_waddr[5:3]][31:0]  <= hb_wdata;
                    12'b0001_00??_?100: region[hb_waddr[5:3]][63:32] <= hb_wdata;
                    default: ;
                endcase
                if (hb_waddr[11:0] == 12'h010) irq_en <= hb_wdata[1];
            end

            // ---------------- core reset clears core-owned state ----------------
            if (core_reset) begin
                status <= 0; doorbell <= 2'd0;
                fw_version <= 0; fw_arch <= 0; fw_heartbeat <= 0;
                err_desc <= 0; err_unit <= 0; err_info0 <= 0; err_info1 <= 0;
                wd_fired <= 1'b0; trap_seen <= 1'b0;
                io_ack <= 1'b0;
            end

            // ---------------- host read pipeline (stage 1 -> register value) ----------------
            if (hb_re) host_reg_q <= shared_rd(hb_raddr[11:0]);
        end
    end

    // host read stage 2: select window
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axil_rvalid <= 1'b0; s_axil_rdata <= 0;
        end else begin
            if (rd_s2) begin
                s_axil_rvalid <= 1'b1;
                case (1'b1)
                    hb_r_regs:  s_axil_rdata <= host_reg_q;
                    hb_r_trace: s_axil_rdata <= trace_q[hb_raddr[3:2]*32 +: 32];
                    hb_r_dmem:  s_axil_rdata <= dmem_qb;
                    hb_r_imem:  s_axil_rdata <= imem_qb;
                    default:    s_axil_rdata <= 32'd0;
                endcase
            end else if (s_axil_rvalid && s_axil_rready)
                s_axil_rvalid <= 1'b0;
        end
    end

endmodule
