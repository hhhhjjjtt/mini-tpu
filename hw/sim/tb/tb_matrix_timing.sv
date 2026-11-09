`timescale 1ns/1ps
`include "mini_tpu_config.svh"

// =============================================================================
// tb_matrix_timing — where the matrix unit's cycles go, per M-tile
// =============================================================================
//
// Timing only (tb_matrix_unit checks the numbers). Ideal surroundings so that
// every lost cycle belongs to the matrix unit or its systolic array:
//   * vmem: grant every request, data one cycle later (no bank contention)
//   * VECTOR side: m_ready held high
//   * weights preloaded through the weight port before the commands
//
// Every core run (one M-tile) is split using only the core's interface
// signals (start, act_enb, out_enb, done), so it works for either core:
//
//   start -> first act read | act reads (useful: k) | read bubbles inside the
//   run | last read -> first out write (drain) | out writes (writeback) |
//   last write -> done | done -> next start (adapter handoff)
//
// Per scenario (independent of when commands are accepted, so it also holds
// with cross-GEMM prefill): exposed fill = first command accept -> first core
// start; every gap between consecutive tiles (those that cross a GEMM boundary
// are reported separately); tail = last core done -> last unit_done. These
// add up to the elapsed cycles.
//
// Scenarios:
//   S1  one GEMM, k = 512, 4 M-tiles
//   S2  two back-to-back GEMMs, k = 512, 4 M-tiles each (gap between commands)
//   S3  one GEMM, k = 64, 8 M-tiles (short K: fixed costs dominate)
//   S4  six back-to-back GEMMs, k = 512, 1 M-tile each, alternating weight
//       halves (decode: one row, a GEMM per 16 output columns)
// =============================================================================

module tb_matrix_timing;

    localparam integer N       = 16;
    localparam integer VM_AW   = 17;
    localparam integer VM_WORDS = 16384;

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    integer cyc = 0;
    always @(posedge clk) cyc <= cyc + 1;

    // ---------------- frontend + matrix unit ----------------
    reg          push = 0;
    reg  [511:0] push_cmd = 0;
    wire [23:0]  my_done;
    wire [95:0]  done_counts = {24'd0, 24'd0, my_done, 24'd0};
    wire         fe_valid, fe_ready;
    wire [511:0] fe_cmd;
    wire         unit_done, unit_err, unit_busy;
    wire [7:0]   unit_err_code;

    unit_frontend u_fe (
        .clk(clk), .rst_n(rst_n), .clear(1'b0),
        .push(push), .push_cmd(push_cmd), .free(), .overflow(),
        .done_counts(done_counts),
        .cmd_valid(fe_valid), .cmd(fe_cmd), .cmd_ready(fe_ready),
        .unit_done(unit_done), .done_count(my_done),
        .wait_stall(), .starved()
    );

    reg              rd_rvalid = 0;
    reg  [127:0]     rd_rdata = 0;
    wire             rd_req;
    wire [VM_AW-1:0] rd_addr;
    reg [127:0]      vmem [0:VM_WORDS-1];

    reg          wgt_wr_en = 0;
    reg  [1:0]   wgt_wr_half = 0;
    reg  [8:0]   wgt_wr_addr = 0;
    reg  [127:0] wgt_wr_data = 0;
    wire         m_valid, m_last;
    wire [511:0] m_data;

    matrix_unit #(.N(`MTPU_ARRAY_N), .VM_AW(VM_AW)) dut (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(fe_valid), .cmd(fe_cmd), .cmd_ready(fe_ready),
        .unit_done(unit_done), .err(unit_err), .err_code(unit_err_code), .busy(unit_busy),
        .vm_rd_req(rd_req), .vm_rd_addr(rd_addr), .vm_rd_gnt(rd_req),
        .vm_rd_rvalid(rd_rvalid), .vm_rd_rdata(rd_rdata),
        .wgt_wr_en(wgt_wr_en), .wgt_wr_half(wgt_wr_half),
        .wgt_wr_addr(wgt_wr_addr), .wgt_wr_data(wgt_wr_data),
        .m_valid(m_valid), .m_data(m_data), .m_last(m_last), .m_ready(1'b1)
    );

    // ideal vmem: every request granted, data the next cycle
    always @(posedge clk) begin
        rd_rvalid <= rd_req;
        rd_rdata  <= vmem[rd_addr % VM_WORDS];
    end

    // ---------------- per-tile event capture ----------------
    wire core_start = dut.c_start;
    wire core_read  = dut.core_act_en;
    wire core_write = dut.core_out_en;
    wire core_fin   = dut.c_running && dut.core_done;
    wire cmd_accept = fe_valid && fe_ready;
    wire row_out    = m_valid;                       // m_ready = 1

    integer t_start, t_rd0, t_rd1, n_rd, t_wr0, t_wr1, n_wr, t_done, t_prev_done;
    integer tile_no, in_tile;
    // per-scenario sums
    integer s_tiles, s_lead, s_read, s_bubble, s_drain, s_wb, s_fin, s_gap, s_xgap, s_nx;
    integer s_fill, s_tail, t_first_accept, t_last_done, t_last_unit_done;

    task automatic clear_sums;
        begin
            s_tiles = 0; s_lead = 0; s_read = 0; s_bubble = 0; s_drain = 0; s_wb = 0;
            s_fin = 0; s_gap = 0; s_xgap = 0; s_nx = 0; s_fill = 0; s_tail = 0; tile_no = 0;
            t_prev_done = -1; t_first_accept = -1;
        end
    endtask

    initial begin clear_sums(); in_tile = 0; end

    always @(posedge clk) if (rst_n) begin
        if (cmd_accept && t_first_accept < 0) t_first_accept = cyc;
        if (unit_done) t_last_unit_done = cyc;
        if (core_start) begin
            if (t_prev_done < 0) s_fill = s_fill + (cyc - t_first_accept);       // exposed fill
            else if (dut.c_tile == 0) begin s_xgap = s_xgap + (cyc - t_prev_done); s_nx = s_nx + 1; end
            else s_gap = s_gap + (cyc - t_prev_done);
            t_start = cyc; t_rd0 = -1; n_rd = 0; t_wr0 = -1; n_wr = 0; in_tile = 1;
        end
        if (in_tile && core_read) begin
            if (t_rd0 < 0) t_rd0 = cyc;
            t_rd1 = cyc; n_rd = n_rd + 1;
        end
        if (in_tile && core_write) begin
            if (t_wr0 < 0) t_wr0 = cyc;
            t_wr1 = cyc; n_wr = n_wr + 1;
        end
        if (in_tile && core_fin) begin
            t_done = cyc; in_tile = 0;
            if (tile_no < 2)
                $display("    tile %0d: lead %0d | reads %0d (+%0d bubbles) | drain %0d | writeback %0d | finish %0d | total %0d",
                         tile_no, t_rd0 - t_start, n_rd, (t_rd1 - t_rd0 + 1) - n_rd, t_wr0 - t_rd1 - 1,
                         t_wr1 - t_wr0 + 1, t_done - t_wr1, t_done - t_start + 1);
            s_tiles  = s_tiles + 1;
            s_lead   = s_lead + (t_rd0 - t_start);
            s_read   = s_read + n_rd;
            s_bubble = s_bubble + (t_rd1 - t_rd0 + 1) - n_rd;
            s_drain  = s_drain + (t_wr0 - t_rd1 - 1);
            s_wb     = s_wb + (t_wr1 - t_wr0 + 1);
            s_fin    = s_fin + (t_done - t_wr1);
            t_prev_done = t_done; t_last_done = t_done;
            tile_no = tile_no + 1;
        end
    end

    // ---------------- helpers ----------------
    task automatic load_weights(input integer kk);
        integer t, h;
        begin
            for (h = 0; h < 2; h = h + 1)
                for (t = 0; t < kk; t = t + 1) begin
                    @(negedge clk);
                    wgt_wr_en = 1; wgt_wr_half = h; wgt_wr_addr = t;
                    wgt_wr_data = {$urandom, $urandom, $urandom, $urandom};
                end
            @(negedge clk); wgt_wr_en = 0;
        end
    endtask

    task automatic push_gemm(input integer act_b, input integer pitch_b, input integer mt,
                             input integer kk, input integer h);
        begin
            push_cmd = 0;
            push_cmd[7:0]       = 8'h10;
            push_cmd[192 +: 32] = act_b;
            push_cmd[224 +: 16] = pitch_b;
            push_cmd[240 +: 16] = mt;
            push_cmd[256 +: 16] = kk;
            push_cmd[272]       = h;
            @(negedge clk); push = 1;
            @(negedge clk); push = 0;
        end
    endtask

    task automatic wait_done(input integer target);
        integer c;
        begin
            c = 0;
            while (my_done < target && c < 200000) begin @(posedge clk); c = c + 1; end
            if (my_done < target) $display("TIMEOUT waiting for DONE=%0d", target);
            repeat (5) @(posedge clk);
        end
    endtask

    task automatic report(input string name, input integer ngemm, input integer mt, input integer kk);
        integer ideal, elapsed, tiles_cyc;
        string  xg;
        begin
            if (s_nx > 0) xg = $sformatf("%0.1f avg over %0d", 1.0 * s_xgap / s_nx, s_nx);
            else          xg = "none";
            s_tail    = t_last_unit_done - t_last_done;
            elapsed   = t_last_unit_done - t_first_accept;
            tiles_cyc = s_lead + s_read + s_bubble + s_drain + s_wb + s_fin;
            ideal     = ngemm * mt * kk;
            $display("  %s: %0d GEMM x %0d tiles x k=%0d | %0d cycles, ideal %0d -> %0.1f %% of peak",
                     name, ngemm, mt, kk, elapsed, ideal, 100.0 * ideal / elapsed);
            $display("    per tile (avg over %0d): lead %0.1f, reads %0.1f, read bubbles %0.1f, drain %0.1f, writeback %0.1f, finish %0.1f",
                     s_tiles, 1.0 * s_lead / s_tiles, 1.0 * s_read / s_tiles, 1.0 * s_bubble / s_tiles,
                     1.0 * s_drain / s_tiles, 1.0 * s_wb / s_tiles, 1.0 * s_fin / s_tiles);
            $display("    exposed fill %0d | gaps inside a GEMM %0.1f avg | gaps between GEMMs %0s | tail %0d",
                     s_fill, (s_tiles > s_nx + 1) ? 1.0 * s_gap / (s_tiles - s_nx - 1) : 0.0,
                     xg, s_tail);
            $display("    share of elapsed: fill %0.1f %%, tiles %0.1f %% (reads %0.1f %%), gaps %0.1f %%, GEMM switches %0.1f %%, tail %0.1f %%  (accounted %0d of %0d)",
                     100.0 * s_fill / elapsed, 100.0 * tiles_cyc / elapsed, 100.0 * s_read / elapsed,
                     100.0 * s_gap / elapsed, 100.0 * s_xgap / elapsed, 100.0 * s_tail / elapsed,
                     s_fill + tiles_cyc + s_gap + s_xgap + s_tail - s_tiles, elapsed);   // each tile's start cycle is also a gap end
        end
    endtask

    integer i;
    initial begin
        for (i = 0; i < VM_WORDS; i = i + 1) vmem[i] = {$urandom, $urandom, $urandom, $urandom};
        repeat (5) @(negedge clk);
        rst_n = 1;
        repeat (2) @(negedge clk);
        load_weights(512);

        $display("S1: one GEMM, k=512, 4 M-tiles");
        clear_sums();
        push_gemm(0, 512, 4, 512, 0);
        wait_done(1);
        report("S1", 1, 4, 512);

        $display("S2: two back-to-back GEMMs, k=512, 4 M-tiles each");
        clear_sums();
        push_gemm(0, 512, 4, 512, 0);
        push_gemm(0, 512, 4, 512, 1);
        wait_done(3);
        report("S2", 2, 4, 512);

        $display("S3: one GEMM, k=64, 8 M-tiles");
        clear_sums();
        push_gemm(0, 64, 8, 64, 0);
        wait_done(4);
        report("S3", 1, 8, 64);

        $display("S4: six back-to-back GEMMs, k=512, 1 M-tile each (decode)");
        clear_sums();
        for (i = 0; i < 6; i = i + 1) push_gemm(0, 512, 1, 512, i & 1);
        wait_done(10);
        report("S4", 6, 1, 512);

        if (unit_err) $display("TB FAIL: unit error %h", unit_err_code);
        else          $display("TB DONE");
        $finish;
    end

    initial begin
        #20_000_000;
        $display("TB FAIL: global timeout");
        $finish;
    end

endmodule
