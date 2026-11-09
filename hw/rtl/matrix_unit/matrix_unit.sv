`timescale 1ns/1ps

// =============================================================================
// matrix_unit — mini-tpu matrix unit (adapter around systolic_array.sv)
// =============================================================================
//
// INTERFACE_SPEC.md §5.4 (GEMM), §5.4.1 (LOAD_WGT_VM) and §10 (adapter).
//
// N is the array width: 16 by default, or 8, or a power of two above 16. T is the tile
// width the ISA and the buffers see: N, but never below 16, because vmem words are 16
// bytes and an i8 result row must fill whole words. GEMM K stays a multiple of 16.
//
// An 8-wide array computes each 16 x 16 tile in 4 core runs, one per quarter: a pass
// picks 8 lanes of the staged activations and of the weights, and writes its quarter
// of the out buffer. Nothing outside this file sees the difference.
//
// GEMM: weight-stationary over m_tiles T-row M-tiles.
//
//   for t in 0 .. m_tiles-1:
//     C[TxT] = A[rows Tt..Tt+T-1, 0..k-1] . W[0..k-1, 0..T-1]
//     stream T rows of C (T x i32) to VECTOR
//
// Three decoupled pipelines, each tile-ordered:
//
//   filler  : vmem row words -> act_transposer -> act staging half (t mod 2)
//   core    : systolic_array reads staging half + weight half,
//             writes T result rows into out buffer half (t mod 2)
//   streamer: out buffer half (t mod 2) -> T x i32 stream (valid/ready/last)
//
// Half state machines (staging: FREE -> FILLING -> FILLED -> FREE on core
// done; out buffer: empty -> full on core done -> empty after T rows) let
// the filler prepare tile t+1 while the core computes tile t, and the core
// start tile t+1 while the streamer drains tile t. Halves alternate per tile
// over the whole command stream (f_hp / c_hp / s_hp), not per command.
//
// Cross-GEMM prefill: up to two GEMMs wait behind the running one, in pending
// slots P and Q. Once the filler has finished the newest GEMM, the next one is
// accepted and the filler starts on its first tiles. Once the core has
// finished the running GEMM's tiles it moves on to P (c_on_p) while the
// streamer still hands the running GEMM's last rows to VECTOR. When that last
// row is taken the running GEMM is done, P becomes the running one and Q
// moves to P. With one-tile GEMMs (decode) the filler, the core and the
// streamer each work on a different GEMM, so none of them waits. The front end only
// offers a command whose wait conditions hold, so a GEMM that reads the
// current GEMM's results (it waits on VECTOR) is not accepted early.
// LOAD_WGT_VM and bad commands are taken only when the unit is idle.
//
// Weight buffer: three halves (wgt_half 0 .. 2) of K_MAX rows, so a LOAD_WGT
// can fill one while the GEMMs before it read the other two.
//
// The array keeps FlexNPU's core handshake (see systolic_array.sv). act/wgt
// buf_ready are held high while a resident half is in use (allowed by §5).
//
// LOAD_WGT_VM: fills a weight-buffer half from vmem (activation x activation
// GEMMs, e.g. attention). k rows:
//   flags.TRANSPOSE = 0: weight row t = the T/16 vmem words at src + t * pitch (V in P.V)
//   flags.TRANSPOSE = 1: the filler reads T rows at src (pitch apart) and the
//                        transposer turns each T x 16 byte block into 16
//                        weight rows: row t lane j = byte t of source row j (K^T)
// Rows go through a small queue into the weight buffer; LOAD_WGT writes from
// the LOAD unit (other half) have priority on the shared write port.
// Runs between GEMMs (commands execute in order), so the filler and the
// transposer are free.
//
// =============================================================================

module matrix_unit #(
    parameter integer N        = 16,      // array width: 8, 16, 32, 64
    parameter integer T        = (N < 16) ? 16 : N,    // tile width: derived, do not set
    parameter integer K_TILE   = 16,
    parameter integer K_MAX    = 512,     // per GEMM command (INTERFACE_SPEC I5)
    parameter integer CORE_MAX_K = 1024,  // systolic_array MAX_K (its default)
    // Core FIFO sizing as driven by the production T=16 shell
    // (fpga/dma/top_axis_bram_16_pingpong.v).
    parameter integer ACT_FIFO_DEPTH = 16,
    parameter integer OUT_FIFO_DEPTH = 2 * N,
    parameter integer VM_AW    = 17,      // vmem word address (2 MB / 16 B)
    parameter integer CMD_W    = 512
)(
    input  wire                 clk,
    input  wire                 rst_n,

    // ---- Command (from unit_frontend) ----
    input  wire                 cmd_valid,
    input  wire [CMD_W-1:0]     cmd,
    output wire                 cmd_ready,
    output reg                  unit_done,      // 1-cycle pulse per completed GEMM
    output reg                  err,            // sticky
    output reg  [7:0]           err_code,
    output wire                 busy,

    // ---- vmem read port (word address, in-order responses) ----
    output wire                 vm_rd_req,
    output wire [VM_AW-1:0]     vm_rd_addr,
    input  wire                 vm_rd_gnt,
    input  wire                 vm_rd_rvalid,
    input  wire [127:0]         vm_rd_rdata,

    // ---- Weight buffer write port (from LOAD unit, LOAD_WGT; has priority) ----
    // one 16-byte word per write; addr = row * T/16 + word within the row
    input  wire                 wgt_wr_en,
    input  wire [1:0]           wgt_wr_half,
    input  wire [$clog2(K_MAX*T/16)-1:0] wgt_wr_addr,
    input  wire [127:0]         wgt_wr_data,

    // ---- Output stream (to VECTOR) ----
    output wire                 m_valid,
    output wire [T*32-1:0]      m_data,
    output wire                 m_last,
    input  wire                 m_ready
);

    localparam integer LW      = T * 8;            // one weight / column word
    localparam integer OW      = T * 32;           // one result row
    localparam integer KA      = $clog2(K_MAX);    // address within a half
    localparam integer SA      = KA + 1;           // {half, addr}
    localparam integer RW      = $clog2(T);        // row index
    localparam integer WPR     = T / 16;           // vmem words per weight row
    localparam integer WPB     = $clog2(WPR);
    localparam integer FW      = KA + 1 + WPB;     // counts the vmem words of k rows
    localparam integer Q       = T / N;            // core runs per tile side
    localparam integer PASSES  = Q * Q;
    localparam integer PB      = (PASSES <= 1) ? 1 : $clog2(PASSES);
    localparam integer BUF_D   = 2 * K_MAX;        // staging: two halves of K_MAX rows
    localparam integer WGT_D   = 3 * K_MAX;        // weights: three halves of K_MAX rows

    localparam [7:0] OP_GEMM       = 8'h10;
    localparam [7:0] OP_LOAD_WGT_VM = 8'h11;
    localparam integer WQ_DEPTH    = 4;
    localparam [7:0] ERR_BAD_OPCODE = 8'h10;
    localparam [7:0] ERR_BAD_FIELD  = 8'h11;

    initial begin
        if (N != 8 && (N < 16 || (N & (N - 1)) != 0))
            $error("N must be 8 or a power of two >= 16");
        if (K_MAX > CORE_MAX_K)
            $error("K_MAX must fit the core");
    end

    localparam [1:0] H_FREE    = 2'd0;
    localparam [1:0] H_FILLING = 2'd1;
    localparam [1:0] H_FILLED  = 2'd2;

    // ------------------------------------------------------------
    // Command decode (descriptor byte o >= 0x18 at cmd bit 192 + 8(o - 0x18))
    // ------------------------------------------------------------
    wire [7:0]  c_opcode   = cmd[7:0];
    wire [7:0]  c_flags    = cmd[15:8];
    wire [31:0] c_act_addr = cmd[192 +: 32];   // 0x18
    wire [15:0] c_act_pitch= cmd[224 +: 16];   // 0x1C
    wire [15:0] c_m_tiles  = cmd[240 +: 16];   // 0x1E
    wire [15:0] c_k        = cmd[256 +: 16];   // 0x20
    wire [1:0]  c_wgt_half = cmd[272 +: 2];    // 0x22
    wire [7:0]  c_half_byte= cmd[272 +: 8];    // 0x22

    wire c_bad_field = (c_k == 0) || (c_k[3:0] != 0) || (c_k > K_MAX) || (c_half_byte > 8'd2) ||
                       (c_m_tiles == 0) || (c_act_addr[3:0] != 0) ||
                       (c_act_pitch[3:0] != 0) || (c_act_addr[31:VM_AW+4] != 0);

    // LOAD_WGT_VM: same field offsets as GEMM (src 0x18, pitch 0x1C, k 0x20, half 0x22)
    wire c_wl_bad = (c_k == 0) || (c_k[3:0] != 0) || (c_k > K_MAX) || (c_half_byte > 8'd2) ||
                    (c_act_addr[3:0] != 0) || (c_act_pitch[3:0] != 0) ||
                    (c_act_addr[31:VM_AW+4] != 0);

    reg              running;
    reg [VM_AW-1:0]  pitch_w;
    reg [15:0]       m_tiles;
    reg [15:0]       k;
    reg [1:0]        wgt_half;
    reg              wl, wl_trans;              // LOAD_WGT_VM in progress
    reg [1:0]        wl_half;
    reg [KA:0]       w_cnt;                     // weight rows written
    reg [FW-1:0]     s_iss, s_rcv;              // straight mode: reads issued / returned
    reg [VM_AW-1:0]  s_addr;
    reg [LW-1:0]     s_hold;                    // straight mode: the row so far

    // pending GEMMs, oldest first (accepted early; their tiles may already be filling)
    reg              p_valid, q_valid;
    reg [15:0]       p_k, p_mt, q_k, q_mt;
    reg [1:0]        p_half, q_half;

    // filler's command (the running GEMM's, or the pending one's once accepted)
    reg [15:0]       f_k, f_mt;
    reg [VM_AW-1:0]  f_pitch;

    // half pointers: filler, core and streamer each step through the halves in tile order
    reg              f_hp, c_hp, s_hp;

    // ------------------------------------------------------------
    // Half state
    // ------------------------------------------------------------
    reg [1:0] stg_state [0:1];
    reg [1:0] out_full;

    wire       s_take_w;                         // streamer handshake (defined below)
    wire       last_row_w;

    // ------------------------------------------------------------
    // Filler state
    // ------------------------------------------------------------
    reg              f_active;
    reg [15:0]       f_tile;
    reg [VM_AW-1:0]  f_tile_base, f_next_base, f_row_off;
    reg [FW-1:0]     f_issued;      // words issued this tile (<= k * T/16)
    reg [RW-1:0]     f_r;
    reg [KA-4:0]     f_kb;
    reg [KA:0]       e_cnt;         // column words written this tile

    wire f_start    = running && (!wl || wl_trans) && !f_active && (f_tile < f_mt) &&
                      (stg_state[f_hp] == H_FREE);
    wire f_done     = !f_active && (f_tile == f_mt);
    wire [FW-1:0] f_words = (f_k >> 4) << RW;    // T rows of k/16 words
    wire [FW-1:0] held = f_issued - {e_cnt[KA:4], {RW{1'b0}}};
    wire credit_ok  = held < (2 * T);

    // weight queue (LOAD_WGT_VM rows on their way to the weight buffer)
    wire          wq_empty, wq_full;
    wire [LW-1:0] wq_dout;
    wire [$clog2(WQ_DEPTH+1)-1:0] wq_count;
    wire          wl_straight = wl && !wl_trans;
    wire          s_room      = ({1'b0, wq_count} + (s_iss - s_rcv)) < WQ_DEPTH;
    wire [15:0]   k_words     = k << WPB;
    wire          s_req       = wl_straight && (s_iss < k_words) && s_room;
    wire          s_iss_end   = (s_iss & (WPR - 1)) == WPR - 1;     // last word of a row
    wire          s_rcv_end   = (s_rcv & (WPR - 1)) == WPR - 1;

    // words of a straight row arrive low first
    wire [LW+127:0] s_cat     = {vm_rd_rdata, s_hold};
    wire [LW-1:0]   s_next    = s_cat[LW+127:128];

    assign vm_rd_req  = (f_active && (f_issued < f_words) && credit_ok) || s_req;
    assign vm_rd_addr = wl_straight ? s_addr : f_tile_base + f_row_off + f_kb;
    wire   rd_issue   = vm_rd_req && vm_rd_gnt && !wl_straight;
    wire   s_issue    = s_req && vm_rd_gnt;

    // Transposer
    wire          tp_valid;
    wire [LW-1:0] tp_col;
    wire [3:0]    tp_idx;
    wire          tp_ready = !wl || !wq_full;  // staging BRAM accepts every cycle
    wire          tp_take = tp_valid && tp_ready;
    wire          f_last_col = tp_take && (e_cnt == f_k - 1);

    act_transposer #(.ROWS(T), .COLS(16), .LW(8)) u_tp (
        .clk       (clk),
        .rst_n     (rst_n),
        .clear     (f_start),
        .in_valid  (vm_rd_rvalid && f_active),
        .in_row    (vm_rd_rdata),
        .out_valid (tp_valid),
        .out_col   (tp_col),
        .out_idx   (tp_idx),
        .out_ready (tp_ready)
    );

    // LOAD_WGT_VM rows -> weight buffer (LOAD unit writes win the port)
    wire wq_push = wl_trans ? (wl && tp_take) : (wl_straight && vm_rd_rvalid && s_rcv_end);
    wire wl_wr   = wl && !wq_empty && !wgt_wr_en;

    sfifo #(.W(LW), .DEPTH(WQ_DEPTH)) u_wq (
        .clk(clk), .rst_n(rst_n),
        .push(wq_push), .din(wl_trans ? tp_col : s_next), .pop(wl_wr),
        .dout(wq_dout), .empty(wq_empty), .full(wq_full), .count(wq_count)
    );

    // ------------------------------------------------------------
    // Core runner state
    // ------------------------------------------------------------
    reg         c_running;
    reg [15:0]  c_tile;
    reg         c_half;
    reg         c_on_p;                         // the core is on the pending GEMM

    // the GEMM the core works on: the running one, or the pending one once c_on_p
    wire [15:0] cr_mt   = c_on_p ? p_mt   : m_tiles;
    wire [15:0] cr_k    = c_on_p ? p_k    : k;
    wire [1:0]  cr_half = c_on_p ? p_half : wgt_half;

    wire core_busy, core_done, core_tile_done;
    wire [$clog2(CORE_MAX_K)-1:0] core_act_addr, core_wgt_addr;
    wire        core_act_en, core_wgt_en;
    wire [LW-1:0] stg_dout, wgt_dout;
    wire [$clog2(N)-1:0] core_out_addr;
    wire        core_out_en;
    wire [N*4-1:0] core_out_we;
    wire [N*32-1:0] core_out_din;

    // pass over the tile: row part and column part (one pass unless the array is
    // narrower than the tile)
    reg  [PB-1:0] c_pass;
    reg           c_again;
    wire          c_last   = (c_pass == PASSES - 1);
    wire          core_fin = c_running && core_done && c_last;
    wire [31:0]   c_rp     = c_pass / Q;
    wire [31:0]   c_cp     = c_pass % Q;
    wire [N*8-1:0] core_act_dout = stg_dout[c_rp*N*8 +: N*8];
    wire [N*8-1:0] core_wgt_dout = wgt_dout[c_cp*N*8 +: N*8];
    wire [RW-1:0]  core_out_row  = c_rp * N + core_out_addr;

    wire c_start = running && !wl && !c_running && (c_tile < cr_mt) &&
                   (stg_state[c_hp] == H_FILLED) && !out_full[c_hp];

    // ------------------------------------------------------------
    // Streamer state
    // ------------------------------------------------------------
    reg [15:0]   s_tile;
    reg [RW-1:0] s_row;
    reg [OW-1:0] obuf [0:2*T-1];

    assign m_valid = running && !wl && (s_tile < m_tiles) && out_full[s_hp];
    assign m_data  = obuf[{s_hp, s_row}];
    assign m_last  = (s_tile == m_tiles - 1) && (s_row == T - 1);
    wire   s_take  = m_valid && m_ready;
    assign s_take_w   = s_take;
    assign last_row_w = s_take && m_last;

    // ------------------------------------------------------------
    // Accept rules
    // ------------------------------------------------------------
    // idle: any command (bad ones are rejected with an error)
    // early: the next GEMM into P, or into Q behind P, once the filler is done with the newest
    wire early_ok  = running && !wl && !q_valid && f_done && !last_row_w &&
                     (c_opcode == OP_GEMM) && !c_bad_field;
    assign cmd_ready = !running || early_ok;
    assign busy      = running;

    wire acc_gemm  = cmd_valid && cmd_ready && (c_opcode == OP_GEMM) && !c_bad_field;
    wire acc_wl    = cmd_valid && !running && (c_opcode == OP_LOAD_WGT_VM) && !c_wl_bad;
    wire acc_load  = acc_gemm || acc_wl;             // filler takes the new command's fields
    wire exec_new  = (acc_gemm && !running) || (last_row_w && p_valid);   // streamer restart
    // the core has finished the running GEMM's tiles: on to the pending one
    wire c_switch  = running && !wl && p_valid && !c_on_p && !c_running && (c_tile == m_tiles) && !last_row_w;

    // ------------------------------------------------------------
    // Command accept / completion
    // ------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            running    <= 1'b0;
            unit_done  <= 1'b0;
            err        <= 1'b0;
            err_code   <= 8'h00;
            pitch_w    <= 0;
            m_tiles    <= 0;
            k          <= 0;
            wgt_half   <= 2'd0;
            wl         <= 1'b0;
            wl_trans   <= 1'b0;
            wl_half    <= 2'd0;
            p_valid    <= 1'b0;
            p_k        <= 0;
            p_mt       <= 0;
            p_half     <= 2'd0;
            q_valid    <= 1'b0;
            q_k        <= 0;
            q_mt       <= 0;
            q_half     <= 2'd0;
        end else begin
            unit_done <= 1'b0;
            if (early_ok && cmd_valid) begin
                // next GEMM into the first free pending slot; the filler starts on it now
                if (!p_valid) begin
                    p_valid <= 1'b1;
                    p_k     <= c_k;
                    p_mt    <= c_m_tiles;
                    p_half  <= c_wgt_half;
                end else begin
                    q_valid <= 1'b1;
                    q_k     <= c_k;
                    q_mt    <= c_m_tiles;
                    q_half  <= c_wgt_half;
                end
            end
            if (!running && cmd_valid) begin
                if (c_opcode == OP_GEMM) begin
                    if (c_bad_field) begin
                        err      <= 1'b1;
                        err_code <= ERR_BAD_FIELD;
                    end else begin
                        running    <= 1'b1;
                        wl         <= 1'b0;
                        pitch_w    <= c_act_pitch[15:4];
                        m_tiles    <= c_m_tiles;
                        k          <= c_k;
                        wgt_half   <= c_wgt_half;
                    end
                end else if (c_opcode == OP_LOAD_WGT_VM) begin
                    if (c_wl_bad) begin
                        err      <= 1'b1;
                        err_code <= ERR_BAD_FIELD;
                    end else begin
                        running    <= 1'b1;
                        wl         <= 1'b1;
                        wl_trans   <= c_flags[0];
                        wl_half    <= c_wgt_half;
                        pitch_w    <= c_act_pitch[15:4];
                        m_tiles    <= 16'd1;            // transposed: one T-row block
                        k          <= c_k;
                    end
                end else begin
                    err      <= 1'b1;
                    err_code <= ERR_BAD_OPCODE;
                end
            end else if (running && !wl && last_row_w) begin
                unit_done <= 1'b1;
                if (p_valid) begin                  // P becomes the running one, Q moves to P
                    m_tiles  <= p_mt;
                    k        <= p_k;
                    wgt_half <= p_half;
                    if (q_valid) begin
                        p_k     <= q_k;
                        p_mt    <= q_mt;
                        p_half  <= q_half;
                        q_valid <= 1'b0;
                    end else
                        p_valid <= 1'b0;
                end else
                    running  <= 1'b0;
            end else if (running && wl && w_cnt == k) begin
                running   <= 1'b0;
                wl        <= 1'b0;
                unit_done <= 1'b1;
            end
        end
    end

    // ------------------------------------------------------------
    // LOAD_WGT_VM counters
    // ------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            w_cnt <= 0; s_iss <= 0; s_rcv <= 0; s_addr <= 0;
        end else if (!running) begin
            w_cnt  <= 0;
            s_iss  <= 0;
            s_rcv  <= 0;
            s_addr <= c_act_addr[VM_AW+3:4];
        end else begin
            if (wl_wr)   w_cnt <= w_cnt + 1'b1;
            if (s_issue) begin
                s_iss  <= s_iss + 1'b1;
                s_addr <= s_iss_end ? s_addr + pitch_w - (WPR - 1) : s_addr + 1'b1;
            end
            if (wl_straight && vm_rd_rvalid) begin
                s_rcv  <= s_rcv + 1'b1;
                s_hold <= s_next;
            end
        end
    end

    // ------------------------------------------------------------
    // Filler
    // ------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            f_active    <= 1'b0;
            f_tile      <= 0;
            f_tile_base <= 0;
            f_next_base <= 0;
            f_row_off   <= 0;
            f_issued    <= 0;
            f_r         <= 0;
            f_kb        <= 0;
            e_cnt       <= 0;
            f_k         <= 0;
            f_mt        <= 0;
            f_pitch     <= 0;
            f_hp        <= 1'b0;
        end else if (acc_load) begin
            // filler is idle here (unit idle, or done with the running GEMM)
            f_tile      <= 0;
            f_k         <= c_k;
            f_mt        <= acc_wl ? 16'd1 : c_m_tiles;      // transposed LOAD_WGT_VM: one block
            f_pitch     <= c_act_pitch[15:4];
            f_next_base <= c_act_addr[VM_AW+3:4];
        end else begin
            if (f_start) begin
                f_active    <= 1'b1;
                f_tile_base <= f_next_base;
                f_next_base <= f_next_base + (f_pitch << RW);
                f_row_off   <= 0;
                f_issued    <= 0;
                f_r         <= 0;
                f_kb        <= 0;
                e_cnt       <= 0;
            end else begin
                if (rd_issue) begin
                    f_issued <= f_issued + 1'b1;
                    if (f_r == T - 1) begin
                        f_r       <= 0;
                        f_row_off <= 0;
                        f_kb      <= f_kb + 1'b1;
                    end else begin
                        f_r       <= f_r + 1'b1;
                        f_row_off <= f_row_off + f_pitch;
                    end
                end
                if (tp_take)
                    e_cnt <= e_cnt + 1'b1;
                if (f_last_col) begin
                    f_active <= 1'b0;
                    f_tile   <= f_tile + 1'b1;
                    if (!wl) f_hp <= ~f_hp;
                end
            end
        end
    end

    // ------------------------------------------------------------
    // Half state updates (filler start/finish, core done)
    // ------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stg_state[0] <= H_FREE;
            stg_state[1] <= H_FREE;
            out_full     <= 2'b00;
            s_hp         <= 1'b0;
        end else begin
            if (f_start && !wl)
                stg_state[f_hp] <= H_FILLING;
            if (f_last_col && !wl)
                stg_state[f_hp] <= H_FILLED;
            if (core_fin) begin
                stg_state[c_half] <= H_FREE;
                out_full[c_half]  <= 1'b1;
            end
            if (s_take && s_row == T - 1) begin
                out_full[s_hp] <= 1'b0;
                s_hp           <= ~s_hp;
            end
        end
    end

    // ------------------------------------------------------------
    // Core runner
    // ------------------------------------------------------------
    // At exec_new the core is idle unless it is already on the pending GEMM, which then
    // becomes the running one with its tile count unchanged.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            c_running <= 1'b0;
            c_tile    <= 0;
            c_half    <= 1'b0;
            c_hp      <= 1'b0;
            c_on_p    <= 1'b0;
        end else begin
            if (c_switch || (exec_new && !c_on_p))
                c_tile <= 0;
            else if (core_fin)
                c_tile <= c_tile + 1'b1;
            if (c_start) begin
                c_running <= 1'b1;
                c_half    <= c_hp;
                c_hp      <= ~c_hp;
            end else if (core_fin)
                c_running <= 1'b0;
            if (c_switch)
                c_on_p <= 1'b1;
            else if (exec_new)
                c_on_p <= 1'b0;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            c_pass  <= 0;
            c_again <= 1'b0;
        end else begin
            c_again <= c_running && core_done && !c_last;
            if (c_running && core_done)
                c_pass <= c_last ? {PB{1'b0}} : c_pass + 1'b1;
        end
    end

    // ------------------------------------------------------------
    // Streamer
    // ------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_tile <= 0;
            s_row  <= 0;
        end else if (exec_new) begin
            s_tile <= 0;
            s_row  <= 0;
        end else if (s_take) begin
            s_row <= s_row + 1'b1;
            if (s_row == T - 1)
                s_tile <= s_tile + 1'b1;
        end
    end

    // ------------------------------------------------------------
    // Out buffer (core write port, byte enables honored)
    // ------------------------------------------------------------
    integer bi;
    always @(posedge clk) begin
        if (core_out_en) begin
            for (bi = 0; bi < N*4; bi = bi + 1)
                if (core_out_we[bi])
                    obuf[{c_half, core_out_row}][(c_cp*N*4 + bi)*8 +: 8] <= core_out_din[bi*8 +: 8];
        end
    end

    // ------------------------------------------------------------
    // Act staging (2 halves x K_MAX x T bytes) and weight buffer (3 halves)
    // ------------------------------------------------------------
    // a LOAD_WGT write is one word of a row, a LOAD_WGT_VM write a whole row
    wire [KA-1:0]   wgt_wr_row = wgt_wr_addr >> WPB;
    wire [LW/8-1:0] wgt_wr_sel = {16{1'b1}} << (16 * (wgt_wr_addr & (WPR - 1)));

    dual_port_ram #(.W(LW), .DEPTH(BUF_D)) u_stage (
        .clk     (clk),
        .rst     (!rst_n),
        .wr_en   (tp_take && !wl),
        .wr_we   ({(LW/8){1'b1}}),
        .wr_addr ({f_hp, e_cnt[KA-1:0]}),
        .wr_din  (tp_col),
        .rd_en   (core_act_en),
        .rd_addr ({c_half, core_act_addr[KA-1:0]}),
        .rd_dout (stg_dout)
    );

    dual_port_ram #(.W(LW), .DEPTH(WGT_D)) u_wgt (
        .clk     (clk),
        .rst     (!rst_n),
        .wr_en   (wgt_wr_en || wl_wr),
        .wr_we   (wgt_wr_en ? wgt_wr_sel : {(LW/8){1'b1}}),
        .wr_addr (wgt_wr_en ? {wgt_wr_half, wgt_wr_row} : {wl_half, w_cnt[KA-1:0]}),
        .wr_din  (wgt_wr_en ? {WPR{wgt_wr_data}} : wq_dout),
        .rd_en   (core_wgt_en),
        .rd_addr ({cr_half, core_wgt_addr[KA-1:0]}),
        .rd_dout (wgt_dout)
    );

    // ------------------------------------------------------------
    // Systolic array
    // ------------------------------------------------------------
    systolic_array #(
        .ACT_WIDTH (8),
        .ACC_WIDTH (32),
        .N              (N),
        .ACT_FIFO_DEPTH (ACT_FIFO_DEPTH),
        .OUT_FIFO_DEPTH (OUT_FIFO_DEPTH),
        .K_TILE         (K_TILE),
        .MAX_K          (CORE_MAX_K)
    ) u_core (
        .clk             (clk),
        .rst_n           (rst_n),
        .start           (c_start || c_again),
        .K_total         (cr_k[$clog2(CORE_MAX_K+1)-1:0]),
        .i_accumulate_in (1'b0),
        .i_writeback_en  (1'b1),
        .busy            (core_busy),
        .done            (core_done),
        .tile_done_pulse (core_tile_done),
        .act_addrb       (core_act_addr),
        .act_enb         (core_act_en),
        .act_buf_ready   (c_running && stg_state[c_half] == H_FILLED),
        .act_doutb       (core_act_dout),
        .wgt_addrb       (core_wgt_addr),
        .wgt_enb         (core_wgt_en),
        .wgt_buf_ready   (c_running),
        .wgt_doutb       (core_wgt_dout),
        .out_addrb       (core_out_addr),
        .out_enb         (core_out_en),
        .out_web         (core_out_we),
        .out_dinb        (core_out_din)
    );

endmodule
