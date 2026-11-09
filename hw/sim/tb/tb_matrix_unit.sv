`timescale 1ns/1ps
`include "mini_tpu_config.svh"

// =============================================================================
// tb_matrix_unit — self-checking TB for unit_frontend + matrix_unit
// =============================================================================
//
// * vmem model: random grant stalls, in-order responses with random latency.
// * Weights written straight into the weight buffer (stands in for LOAD_WGT).
// * VECTOR side: random m_ready backpressure; every streamed row is checked
//   against a golden i8 x i8 -> i32 GEMM, plus m_last placement.
// * Wait conditions: one command waits on unit 0's DONE counter, which the
//   TB bumps late; the command must not issue before that.
// * LOAD_WGT_VM (weights from vmem): straight (V) while the TB writes the
//   other half through the LOAD_WGT port, and transposed (K^T); each is
//   followed by GEMMs checked against the golden model.
//
// =============================================================================

module tb_matrix_unit;

    localparam integer N       = 16;
    localparam integer VM_AW   = 17;
    localparam integer VM_WORDS = 16384;          // 256 KB modeled
    localparam integer MAXM    = 96;
    localparam integer MAXK    = 512;

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    // ---------------- frontend <-> unit ----------------
    reg          push = 0;
    reg  [511:0] push_cmd = 0;
    wire [3:0]   free;
    wire         overflow;
    reg  [23:0]  ext_done [0:3];                  // DONE counters of other units
    wire [23:0]  my_done;
    wire [95:0]  done_counts = {ext_done[3], ext_done[2], my_done, ext_done[0]};
    wire         fe_valid, fe_ready;
    wire [511:0] fe_cmd;
    wire         unit_done, unit_err, unit_busy;
    wire [7:0]   unit_err_code;
    wire         wait_stall, starved;

    unit_frontend u_fe (
        .clk(clk), .rst_n(rst_n), .clear(1'b0),
        .push(push), .push_cmd(push_cmd), .free(free), .overflow(overflow),
        .done_counts(done_counts),
        .cmd_valid(fe_valid), .cmd(fe_cmd), .cmd_ready(fe_ready),
        .unit_done(unit_done), .done_count(my_done),
        .wait_stall(wait_stall), .starved(starved)
    );

    // ---------------- vmem model ----------------
    reg  [127:0] vmem [0:VM_WORDS-1];
    wire         rd_req;
    wire [VM_AW-1:0] rd_addr;
    reg          rd_gnt;
    reg          rd_rvalid;
    reg  [127:0] rd_rdata;

    // in-order response queue
    reg  [127:0] q_data [0:63];
    integer      q_due  [0:63];
    integer      q_head = 0, q_tail = 0, cyc = 0;

    // ---------------- weight buffer write ----------------
    reg          wgt_wr_en = 0;
    reg  [1:0]   wgt_wr_half = 0;
    reg  [8:0]   wgt_wr_addr = 0;
    reg  [127:0] wgt_wr_data = 0;

    // ---------------- stream ----------------
    wire         m_valid, m_last;
    wire [511:0] m_data;
    reg          m_ready = 0;

    matrix_unit #(.N(`MTPU_ARRAY_N), .K_TILE(16), .VM_AW(VM_AW)) dut (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(fe_valid), .cmd(fe_cmd), .cmd_ready(fe_ready),
        .unit_done(unit_done), .err(unit_err), .err_code(unit_err_code), .busy(unit_busy),
        .vm_rd_req(rd_req), .vm_rd_addr(rd_addr), .vm_rd_gnt(rd_gnt),
        .vm_rd_rvalid(rd_rvalid), .vm_rd_rdata(rd_rdata),
        .wgt_wr_en(wgt_wr_en), .wgt_wr_half(wgt_wr_half),
        .wgt_wr_addr(wgt_wr_addr), .wgt_wr_data(wgt_wr_data),
        .m_valid(m_valid), .m_data(m_data), .m_last(m_last), .m_ready(m_ready)
    );

    always @(posedge clk) cyc <= cyc + 1;

    // grant: ~75 %, re-drawn every cycle
    reg gnt_rand = 0;
    always @(negedge clk) gnt_rand <= ($urandom % 4 != 0);
    always @(*) rd_gnt = rd_req && gnt_rand;

    always @(posedge clk) begin
        if (rd_req && rd_gnt) begin
            q_data[q_tail % 64] <= vmem[rd_addr % VM_WORDS];
            q_due [q_tail % 64] <= cyc + 1 + ($urandom % 3);
            q_tail <= q_tail + 1;
        end
        if (q_head != q_tail && q_due[q_head % 64] <= cyc) begin
            rd_rvalid <= 1'b1;
            rd_rdata  <= q_data[q_head % 64];
            q_head    <= q_head + 1;
        end else begin
            rd_rvalid <= 1'b0;
        end
    end

    always @(posedge clk) m_ready <= ($urandom % 5 != 0);

    // ---------------- golden model storage ----------------
    reg signed [7:0]  A [0:MAXM-1][0:MAXK-1];
    reg signed [7:0]  Wt [0:2][0:MAXK-1][0:N-1];

    integer errors = 0;
    integer rows_checked = 0;

    // expected-stream bookkeeping (commands complete in order)
    integer exp_m_tiles [0:15];
    integer exp_k       [0:15];
    integer exp_half    [0:15];
    integer n_cmds = 0, cur_cmd = 0, cur_row = 0;

    function automatic signed [31:0] golden(input integer r, input integer j,
                                            input integer kk, input integer h);
        integer t; reg signed [31:0] acc;
        begin
            acc = 0;
            for (t = 0; t < kk; t = t + 1)
                acc = acc + A[r][t] * Wt[h][t][j];
            golden = acc;
        end
    endfunction

    // stream checker
    integer j;
    reg signed [31:0] got, exp;
    always @(posedge clk) begin
        if (m_valid && m_ready) begin
            for (j = 0; j < N; j = j + 1) begin
                got = m_data[j*32 +: 32];
                exp = golden(cur_row, j, exp_k[cur_cmd], exp_half[cur_cmd]);
                if (got !== exp) begin
                    if (errors < 10)
                        $display("MISMATCH cmd %0d row %0d lane %0d: got %0d exp %0d",
                                 cur_cmd, cur_row, j, got, exp);
                    errors = errors + 1;
                end
            end
            if (m_last !== (cur_row == exp_m_tiles[cur_cmd]*16 - 1)) begin
                $display("BAD m_last cmd %0d row %0d", cur_cmd, cur_row);
                errors = errors + 1;
            end
            rows_checked = rows_checked + 1;
            if (cur_row == exp_m_tiles[cur_cmd]*16 - 1) begin
                cur_cmd = cur_cmd + 1;
                cur_row = 0;
            end else begin
                cur_row = cur_row + 1;
            end
        end
    end

    // ---------------- helpers ----------------
    task automatic fill_A(input integer m_rows, input integer kk,
                          input integer base_w, input integer pitch_b);
        integer r, t, c;
        reg [127:0] w;
        begin
            for (r = 0; r < m_rows; r = r + 1)
                for (t = 0; t < kk; t = t + 1)
                    A[r][t] = $urandom;
            for (r = 0; r < m_rows; r = r + 1)
                for (c = 0; c < kk / 16; c = c + 1) begin
                    for (t = 0; t < 16; t = t + 1)
                        w[t*8 +: 8] = A[r][c*16 + t];
                    vmem[base_w + r * (pitch_b / 16) + c] = w;
                end
        end
    endtask

    task automatic load_W(input integer h, input integer kk);
        integer t, c;
        begin
            for (t = 0; t < kk; t = t + 1) begin
                for (c = 0; c < N; c = c + 1) begin
                    Wt[h][t][c] = $urandom;
                    wgt_wr_data[c*8 +: 8] = Wt[h][t][c];
                end
                @(negedge clk);
                wgt_wr_en   = 1;
                wgt_wr_half = h;
                wgt_wr_addr = t;
                @(negedge clk);
                wgt_wr_en   = 0;
            end
        end
    endtask

    task automatic push_gemm(input integer act_addr_b, input integer pitch_b,
                             input integer m_tiles, input integer kk, input integer h,
                             input [3:0] wmask, input integer t0);
        begin
            push_cmd = 0;
            push_cmd[7:0]        = 8'h10;
            push_cmd[19:16]      = wmask;
            push_cmd[32 +: 24]   = t0;
            push_cmd[192 +: 32]  = act_addr_b;
            push_cmd[224 +: 16]  = pitch_b;
            push_cmd[240 +: 16]  = m_tiles;
            push_cmd[256 +: 16]  = kk;
            push_cmd[272 +: 8]   = h;
            exp_m_tiles[n_cmds]  = m_tiles;
            exp_k[n_cmds]        = kk;
            exp_half[n_cmds]     = h;
            n_cmds = n_cmds + 1;
            @(negedge clk); push = 1;
            @(negedge clk); push = 0;
        end
    endtask

    // LOAD_WGT_VM: weights from vmem into half h (no output stream)
    task automatic push_wl(input integer src_b, input integer pitch_b, input integer kk,
                           input integer h, input bit trans);
        begin
            push_cmd = 0;
            push_cmd[7:0]       = 8'h11;
            push_cmd[8]         = trans;
            push_cmd[192 +: 32] = src_b;
            push_cmd[224 +: 16] = pitch_b;
            push_cmd[256 +: 16] = kk;
            push_cmd[272 +: 8]  = h;
            @(negedge clk); push = 1;
            @(negedge clk); push = 0;
        end
    endtask

    // straight source: weight row t = vmem word at base + t * pitch
    task automatic put_V(input integer h, input integer kk, input integer base_w, input integer pitch_b);
        integer t, c; reg [127:0] w;
        begin
            for (t = 0; t < kk; t = t + 1) begin
                for (c = 0; c < N; c = c + 1) begin
                    Wt[h][t][c] = $urandom;
                    w[c*8 +: 8] = Wt[h][t][c];
                end
                vmem[base_w + t * (pitch_b / 16)] = w;
            end
        end
    endtask

    // transposed source: 16 rows j of K (pitch apart), K[j][t] -> weight row t, lane j
    task automatic put_K(input integer h, input integer kk, input integer base_w, input integer pitch_b);
        integer j, c, b; reg [127:0] w;
        begin
            for (j = 0; j < N; j = j + 1)
                for (b = 0; b < kk; b = b + 1)
                    Wt[h][b][j] = $urandom;
            for (j = 0; j < N; j = j + 1)
                for (c = 0; c < kk / 16; c = c + 1) begin
                    for (b = 0; b < 16; b = b + 1)
                        w[b*8 +: 8] = Wt[h][c*16 + b][j];
                    vmem[base_w + j * (pitch_b / 16) + c] = w;
                end
        end
    endtask

    task automatic wait_cmds_done(input integer target, input integer limit);
        integer c;
        begin
            c = 0;
            while (my_done < target && c < limit) begin
                @(posedge clk); c = c + 1;
            end
            if (my_done < target) begin
                $display("TIMEOUT waiting for DONE=%0d (have %0d)", target, my_done);
                $display("  running=%0b f_active=%0b f_tile=%0d f_issued=%0d e_cnt=%0d stg=%0d/%0d c_running=%0b c_tile=%0d core_busy=%0b s_tile=%0d out_full=%b",
                         dut.running, dut.f_active, dut.f_tile, dut.f_issued, dut.e_cnt,
                         dut.stg_state[0], dut.stg_state[1], dut.c_running, dut.c_tile,
                         dut.core_busy, dut.s_tile, dut.out_full);
                errors = errors + 1;
            end
        end
    endtask

    // cycles where a LOAD_WGT_VM row had to wait for an external weight write
    integer wq_blocked = 0;
    always @(posedge clk)
        if (dut.wl && !dut.wq_empty && dut.wgt_wr_en) wq_blocked = wq_blocked + 1;

    // ---------------- test sequence ----------------
    integer issued_cycle, released_cycle;

    initial begin
        ext_done[0] = 0; ext_done[1] = 0; ext_done[2] = 0; ext_done[3] = 0;
        repeat (5) @(negedge clk);
        rst_n = 1;
        repeat (2) @(negedge clk);

        // T1: k=16, 1 tile, half 0, tight pitch
        fill_A(16, 16, 0, 16);
        load_W(0, 16);
        push_gemm(0, 16, 1, 16, 0, 4'b0000, 0);
        wait_cmds_done(1, 20000);

        // T2: k=64, 3 tiles, half 1, padded pitch (row stride 96 B)
        fill_A(48, 64, 256, 96);
        load_W(1, 64);
        push_gemm(256*16, 96, 3, 64, 1, 4'b0000, 0);
        wait_cmds_done(2, 40000);

        // T3: k=512, 2 tiles, half 0 (full K, both halves of staging in use)
        fill_A(32, 512, 1024, 512);
        load_W(0, 512);
        push_gemm(1024*16, 512, 2, 512, 0, 4'b0000, 0);
        wait_cmds_done(3, 200000);

        // T4: back-to-back commands reusing weights (half 0 and 1), 5 + 1 tiles
        fill_A(80, 48, 2048, 64);
        load_W(1, 48);
        push_gemm(2048*16, 64, 5, 48, 1, 4'b0000, 0);
        push_gemm(2048*16, 64, 1, 48, 1, 4'b0000, 0);
        wait_cmds_done(5, 100000);

        // T5: wait condition — command waits on unit 0 DONE >= 1
        fill_A(16, 32, 4096, 32);
        load_W(0, 32);
        push_gemm(4096*16, 32, 1, 32, 0, 4'b0001, 1);
        repeat (200) @(negedge clk);
        if (unit_busy || my_done != 5) begin
            $display("FAIL: command issued before its wait condition held");
            errors = errors + 1;
        end
        if (!wait_stall) begin
            $display("FAIL: wait_stall not reported");
            errors = errors + 1;
        end
        ext_done[0] = 1;
        wait_cmds_done(6, 20000);

        // T7: LOAD_WGT_VM straight (k = 48, padded pitch) into half 1 while the
        //     LOAD_WGT port writes half 0; then GEMMs on both halves
        put_V(1, 48, 6144, 64);
        push_wl(6144*16, 64, 48, 1, 1'b0);
        load_W(0, 48);                                   // concurrent external writes
        wait_cmds_done(7, 20000);
        if (wq_blocked == 0) begin
            $display("FAIL T7: LOAD_WGT_VM never contended with the LOAD_WGT port");
            errors = errors + 1;
        end
        fill_A(32, 48, 7168, 48);
        push_gemm(7168*16, 48, 2, 48, 1, 4'b0000, 0);
        push_gemm(7168*16, 48, 1, 48, 0, 4'b0000, 0);
        wait_cmds_done(9, 40000);

        // T8: LOAD_WGT_VM transposed (K^T, k = 64, pitch 80) into half 0; then GEMM
        put_K(0, 64, 8192, 80);
        push_wl(8192*16, 80, 64, 0, 1'b1);
        fill_A(16, 64, 9216, 64);
        push_gemm(9216*16, 64, 1, 64, 0, 4'b0000, 0);
        wait_cmds_done(11, 40000);

        // T9: decode pattern: back-to-back one-tile GEMMs over all three weight halves,
        //     each with its own k; the core moves on to the next GEMM while the
        //     previous one's rows are still being taken
        fill_A(16, 64, 10240, 64);
        load_W(2, 64);
        load_W(1, 32);
        load_W(0, 48);
        push_gemm(10240*16, 64, 1, 64, 2, 4'b0000, 0);
        push_gemm(10240*16, 64, 1, 32, 1, 4'b0000, 0);
        push_gemm(10240*16, 64, 1, 48, 0, 4'b0000, 0);
        push_gemm(10240*16, 64, 1, 64, 2, 4'b0000, 0);
        wait_cmds_done(15, 60000);

        // T6: bad field (k not multiple of 16) -> ERR_BAD_FIELD, no DONE
        push_gemm(0, 16, 1, 20, 0, 4'b0000, 0);
        n_cmds = n_cmds - 1;
        repeat (20) @(negedge clk);
        if (!unit_err || unit_err_code != 8'h11 || my_done != 15) begin
            $display("FAIL: bad-field command not rejected (err=%0b code=%h done=%0d)",
                     unit_err, unit_err_code, my_done);
            errors = errors + 1;
        end

        repeat (10) @(negedge clk);
        if (overflow) begin
            $display("FAIL: frontend overflow");
            errors = errors + 1;
        end
        if (rows_checked != (1 + 3 + 2 + 5 + 1 + 1 + 2 + 1 + 1 + 4) * 16) begin
            $display("FAIL: rows checked %0d, expected %0d", rows_checked, 21 * 16);
            errors = errors + 1;
        end

        if (errors == 0)
            $display("TB PASS: %0d rows checked, %0d commands", rows_checked, my_done);
        else
            $display("TB FAIL: %0d errors", errors);
        $finish;
    end

    initial begin
        #50_000_000;
        $display("TB FAIL: global timeout");
        $finish;
    end

endmodule
