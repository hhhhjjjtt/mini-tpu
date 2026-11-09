`timescale 1ns/1ps

// =============================================================================
// unit_frontend — command FIFO + wait-condition check + done counter
// =============================================================================
//
// Shared front end of every mini-tpu unit (INTERFACE_SPEC.md §4.2, §6).
//
//   * Controller pushes 16 x 32-bit commands (512 b) into a show-ahead FIFO.
//   * The head command is offered to the unit only when every wait condition
//     holds: for each unit u in wait_mask, DONE[u] >= t_u.
//   * The unit pulses unit_done once per completed command; this increments
//     the unit's own 24-bit DONE counter (exported to all other front ends).
//   * clear (program start) resets the DONE counter. The FIFO must be empty.
//
// Command word layout (INTERFACE_SPEC §6):
//   W0 [7:0] opcode, [15:8] flags, [19:16] wait_mask
//   W1 [23:0] t0, [31:24] t1[7:0]
//   W2 [15:0] t1[23:8], [31:16] t2[15:0]
//   W3 [7:0] t2[23:16], [31:8] t3
//   W4-W5 resolved DDR address, W6-W15 payload (descriptor bytes 0x18..0x3F)
//
// =============================================================================

module unit_frontend #(
    parameter integer NUM_UNITS = 4,
    parameter integer CMD_W     = 512,
    parameter integer DEPTH     = 8,
    parameter integer CNT_W     = 24,
    parameter integer FREE_W    = $clog2(DEPTH+1)
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         clear,

    // ---- Push side (controller) ----
    input  wire                         push,
    input  wire [CMD_W-1:0]             push_cmd,
    output wire [FREE_W-1:0]            free,
    output reg                          overflow,       // sticky: push while full

    // ---- DONE counters of all units, unit u at [u*CNT_W +: CNT_W] ----
    input  wire [NUM_UNITS*CNT_W-1:0]   done_counts,

    // ---- Issue side (unit) ----
    output wire                         cmd_valid,
    output wire [CMD_W-1:0]             cmd,
    input  wire                         cmd_ready,

    // ---- Completion (unit) ----
    input  wire                         unit_done,
    output reg  [CNT_W-1:0]             done_count,

    // ---- Perf ----
    output wire                         wait_stall,     // head present, waits unmet
    output wire                         starved         // FIFO empty
);

    localparam integer PTR_W = $clog2(DEPTH);

    // ------------------------------------------------------------
    // Show-ahead FIFO
    // ------------------------------------------------------------
    reg [CMD_W-1:0] mem [0:DEPTH-1];
    reg [PTR_W-1:0] wr_ptr, rd_ptr;
    reg [FREE_W-1:0] count;

    wire empty = (count == 0);
    wire full  = (count == DEPTH);
    wire pop   = cmd_valid && cmd_ready;
    wire wr    = push && !full;

    assign free = DEPTH - count;
    assign cmd  = mem[rd_ptr];

    // storage has no reset, so it maps to LUTRAM
    always @(posedge clk)
        if (wr) mem[wr_ptr] <= push_cmd;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr   <= 0;
            rd_ptr   <= 0;
            count    <= 0;
            overflow <= 1'b0;
        end else begin
            if (wr)
                wr_ptr <= (wr_ptr == DEPTH - 1) ? {PTR_W{1'b0}} : wr_ptr + 1'b1;
            if (pop)
                rd_ptr <= (rd_ptr == DEPTH - 1) ? {PTR_W{1'b0}} : rd_ptr + 1'b1;
            case ({wr, pop})
                2'b10:   count <= count + 1'b1;
                2'b01:   count <= count - 1'b1;
                default: count <= count;
            endcase
            if (push && full)
                overflow <= 1'b1;
        end
    end

    // ------------------------------------------------------------
    // Wait-condition decode and check
    // ------------------------------------------------------------
    wire [3:0]       wait_mask = cmd[19:16];
    wire [CNT_W-1:0] tgt [0:3];

    assign tgt[0] = cmd[32 +: 24];
    assign tgt[1] = {cmd[64 +: 16], cmd[56 +: 8]};
    assign tgt[2] = {cmd[96 +: 8],  cmd[80 +: 16]};
    assign tgt[3] = cmd[104 +: 24];

    reg waits_ok;
    integer u;
    always @(*) begin
        waits_ok = 1'b1;
        for (u = 0; u < NUM_UNITS && u < 4; u = u + 1)
            if (wait_mask[u] && (done_counts[u*CNT_W +: CNT_W] < tgt[u]))
                waits_ok = 1'b0;
    end

    assign cmd_valid  = !empty && waits_ok;
    assign wait_stall = !empty && !waits_ok;
    assign starved    = empty;

    // ------------------------------------------------------------
    // DONE counter
    // ------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            done_count <= 0;
        else if (clear)
            done_count <= 0;
        else if (unit_done)
            done_count <= done_count + 1'b1;
    end

endmodule
