`timescale 1ns/1ps

// =============================================================================
// act_transposer — double-banked ROWS x COLS byte transposer
// =============================================================================
//
// INTERFACE_SPEC.md §10.1. vmem holds A row-major: one 128-bit word =
// COLS = 16 consecutive k of one row. The frozen core wants column words: lane
// i = A[i, k], ROWS lanes. This block collects ROWS row words (one k-block) into
// a bank, then emits COLS column words, one per cycle:
//
//     col_word[c][8i +: 8] = row_word[i][8c +: 8]
//
// Two banks: one fills while the other drains. At ROWS = COLS throughput is
// one word in and one word out per cycle; past that the input limits it.
//
// Row words must arrive in order (row 0 .. ROWS-1 of a k-block, k-blocks in
// order). The producer must not push into a bank that is still full — use
// can_push (rows in flight are counted by the caller as credits).
//
// =============================================================================

module act_transposer #(
    parameter integer ROWS = 16,
    parameter integer COLS = 16,
    parameter integer LW   = 8
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     clear,          // start of a tile

    // ---- Row words in ----
    input  wire                     in_valid,
    input  wire [COLS*LW-1:0]       in_row,

    // ---- Column words out ----
    output wire                     out_valid,
    output wire [ROWS*LW-1:0]       out_col,
    output wire [$clog2(COLS)-1:0]  out_idx,        // column index within k-block
    input  wire                     out_ready
);

    reg [COLS*LW-1:0]        bank [0:1][0:ROWS-1];
    reg [1:0]                bank_full;
    reg                      in_bank, out_bank;
    reg [$clog2(ROWS)-1:0]   in_row_idx;
    reg [$clog2(COLS)-1:0]   out_col_idx;

    // ------------------------------------------------------------
    // Fill
    // ------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            in_bank    <= 1'b0;
            in_row_idx <= 0;
        end else if (clear) begin
            in_bank    <= 1'b0;
            in_row_idx <= 0;
        end else if (in_valid) begin
            bank[in_bank][in_row_idx] <= in_row;
            in_row_idx <= in_row_idx + 1'b1;
            if (in_row_idx == ROWS-1)
                in_bank <= ~in_bank;
        end
    end

    // ------------------------------------------------------------
    // Drain
    // ------------------------------------------------------------
    wire drain = out_valid && out_ready;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_bank    <= 1'b0;
            out_col_idx <= 0;
            bank_full   <= 2'b00;
        end else if (clear) begin
            out_bank    <= 1'b0;
            out_col_idx <= 0;
            bank_full   <= 2'b00;
        end else begin
            if (drain) begin
                out_col_idx <= out_col_idx + 1'b1;
                if (out_col_idx == COLS-1)
                    out_bank <= ~out_bank;
            end
            // full set when last row lands; cleared after last column leaves
            if (in_valid && in_row_idx == ROWS-1)
                bank_full[in_bank] <= 1'b1;
            if (drain && out_col_idx == COLS-1)
                bank_full[out_bank] <= 1'b0;
        end
    end

    assign out_valid = bank_full[out_bank];
    assign out_idx   = out_col_idx;

    genvar gi;
    generate
        for (gi = 0; gi < ROWS; gi = gi + 1) begin : gen_col
            assign out_col[gi*LW +: LW] = bank[out_bank][gi][out_col_idx*LW +: LW];
        end
    endgenerate

endmodule
