`timescale 1ns/1ps

// =============================================================================
// issue_engine — hardware translation of unit descriptors into unit commands
// =============================================================================
//
// INTERFACE_SPEC.md §8.7. Firmware hands the engine a run of descriptors
// (bank, index, count); for each one the engine
//
//   1. reads the 64 B descriptor from the descriptor buffer (4 x 128 b),
//   2. maps the opcode to its unit (§2.1); any other opcode stops the run so
//      firmware can handle it (control op or bad opcode),
//   3. computes absolute wait targets t_u = max(0, issued[u] - d_u) (§4.2),
//   4. resolves the DDR address REGION_BASE[r] + (ddr_offset + LOOP_ADD[r])
//      for memory ops (§2.2, §5.8),
//   5. waits for a free FIFO slot and pushes the 16-word command (§6).
//
// issued[] lives in the controller and counts every push (engine or
// firmware), so firmware and engine agree on targets.
//
// ~7 cycles per descriptor. Firmware must not read the descriptor buffer or
// push commands itself while the engine is busy.
//
// =============================================================================

module issue_engine #(
    parameter integer FREE_W = 4,
    parameter integer CNT_W  = 24
)(
    input  wire                 clk,
    input  wire                 rst_n,

    // ---- Control (firmware IO registers) ----
    input  wire                 go,             // pulse: start at start_pos
    input  wire                 stop,           // pulse: abandon the run
    input  wire [8:0]           start_pos,      // {bank, index}
    input  wire [8:0]           count,          // descriptors, 1..256
    output wire                 busy,
    output reg  [2:0]           reason,         // why the last run ended
    output reg  [7:0]           stop_op,        // opcode at pos when reason = CONTROL
    output reg  [8:0]           n_issued,       // commands pushed in the last run
    output reg  [8:0]           pos,            // next descriptor

    // ---- Descriptor buffer read port (1-cycle latency) ----
    output wire                 rd_en,
    output wire [10:0]          rd_addr,        // {bank, index, word}
    input  wire [127:0]         rd_data,

    // ---- Address resolution ----
    input  wire [8*64-1:0]      region_base,
    input  wire [8*32-1:0]      loop_add,

    // ---- Units ----
    input  wire [4*CNT_W-1:0]   issued,
    input  wire [4*FREE_W-1:0]  free,
    output reg  [3:0]           push,
    output reg  [511:0]         cmd
);

    localparam [2:0] R_DONE = 3'd0,     // count exhausted
                     R_CTRL = 3'd1,     // non-unit opcode at pos
                     R_REGION = 3'd2,   // memory op with ddr_region > 7
                     R_OVF  = 3'd3,     // issued[u] at 2^24 - 1
                     R_STOP = 3'd4;     // stopped by firmware

    localparam [1:0] S_IDLE = 2'd0, S_RD = 2'd1, S_DEC = 2'd2, S_PUSH = 2'd3;

    reg [1:0]   state;
    reg [2:0]   rcnt;
    reg         rd_v;
    reg [1:0]   rd_k;
    reg [511:0] d;
    reg [8:0]   remaining;
    reg [1:0]   unit_q;

    assign busy    = (state != S_IDLE);
    assign rd_en   = (state == S_RD) && !rcnt[2];
    assign rd_addr = {pos, rcnt[1:0]};

    // descriptor words arrive one cycle after their read
    always @(posedge clk) begin
        rd_v <= rd_en;
        rd_k <= rcnt[1:0];
        if (rd_v) d[rd_k*128 +: 128] <= rd_data;
    end

    // ---- decode (valid in S_DEC) ----
    wire [7:0] op = d[7:0];
    reg        is_unit, is_mem;
    reg [1:0]  unit;
    always @(*) begin
        is_unit = 1'b1; is_mem = 1'b0; unit = 2'd0;
        case (op)
            8'h01, 8'h02:        begin unit = 2'd0; is_mem = 1'b1; end   // LOAD, LOAD_WGT
            8'h03:               begin unit = 2'd3; is_mem = 1'b1; end   // STORE
            8'h10, 8'h11:              unit = 2'd1;                      // GEMM, LOAD_WGT_VM
            8'h20, 8'h21, 8'h22, 8'h23, 8'h24: unit = 2'd2;              // VEC_*
            default:                   is_unit = 1'b0;
        endcase
    end

    wire [7:0]  region = d[128 +: 8];
    wire [2:0]  r3     = region[2:0];
    wire [31:0] off    = d[160 +: 32] + loop_add[r3*32 +: 32];
    wire [63:0] addr   = region_base[r3*64 +: 64] + {32'd0, off};

    reg [CNT_W-1:0] tgt [0:3];
    integer u;
    always @(*) begin
        for (u = 0; u < 4; u = u + 1) begin
            if (d[16 + u] && issued[u*CNT_W +: CNT_W] > {8'd0, d[32 + 16*u +: 16]})
                tgt[u] = issued[u*CNT_W +: CNT_W] - {8'd0, d[32 + 16*u +: 16]};
            else
                tgt[u] = {CNT_W{1'b0}};
        end
    end

    wire overflow = (issued[unit*CNT_W +: CNT_W] == {CNT_W{1'b1}});

    // ---- sequencer ----
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; rcnt <= 0; remaining <= 0; unit_q <= 0;
            reason <= R_DONE; stop_op <= 0; n_issued <= 0; pos <= 0;
            push <= 0; cmd <= 0;
        end else begin
            push <= 4'd0;
            case (state)
                S_IDLE: if (go) begin
                    pos       <= start_pos;
                    remaining <= count;
                    n_issued  <= 0;
                    rcnt      <= 0;
                    reason    <= R_DONE;
                    if (count != 0) state <= S_RD;
                end

                S_RD: begin
                    rcnt <= rcnt + 1'b1;
                    if (rcnt == 3'd4) state <= S_DEC;      // word 3 lands this cycle
                end

                S_DEC: begin
                    if (!is_unit) begin
                        reason <= R_CTRL; stop_op <= op; state <= S_IDLE;
                    end else if (is_mem && region > 8'd7) begin
                        reason <= R_REGION; stop_op <= op; state <= S_IDLE;
                    end else if (overflow) begin
                        reason <= R_OVF; stop_op <= op; state <= S_IDLE;
                    end else begin
                        cmd[0   +: 32]  <= {12'd0, d[19:16], d[15:0]};
                        cmd[32  +: 32]  <= {tgt[1][7:0], tgt[0]};
                        cmd[64  +: 32]  <= {tgt[2][15:0], tgt[1][23:8]};
                        cmd[96  +: 32]  <= {tgt[3], tgt[2][23:16]};
                        cmd[128 +: 64]  <= is_mem ? addr : 64'd0;
                        cmd[192 +: 320] <= d[192 +: 320];
                        unit_q <= unit;
                        state  <= S_PUSH;
                    end
                end

                S_PUSH: if (free[unit_q*FREE_W +: FREE_W] != 0) begin
                    push[unit_q] <= 1'b1;
                    pos          <= pos + 1'b1;
                    n_issued     <= n_issued + 1'b1;
                    remaining    <= remaining - 1'b1;
                    rcnt         <= 0;
                    state        <= (remaining == 9'd1) ? S_IDLE : S_RD;
                end
            endcase
            if (stop && state != S_IDLE) begin       // a push in this cycle still counts
                reason <= R_STOP;
                state  <= S_IDLE;
            end
        end
    end

endmodule
