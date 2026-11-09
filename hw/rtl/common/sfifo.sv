`timescale 1ns/1ps

// =============================================================================
// sfifo — small synchronous show-ahead FIFO (dout = head while !empty)
// =============================================================================
// Push and pop may happen in the same cycle (also when full: the pop frees the
// slot first only if the caller gates push with !full; full blocks push).

module sfifo #(
    parameter integer W     = 128,
    parameter integer DEPTH = 16,
    parameter integer CW    = $clog2(DEPTH + 1)
)(
    input  wire          clk,
    input  wire          rst_n,
    input  wire          push,
    input  wire [W-1:0]  din,
    input  wire          pop,
    output wire [W-1:0]  dout,
    output wire          empty,
    output wire          full,
    output reg  [CW-1:0] count
);

    localparam integer PW = (DEPTH <= 1) ? 1 : $clog2(DEPTH);

    reg [W-1:0]  mem [0:DEPTH-1];
    reg [PW-1:0] wp, rp;

    wire do_push = push && !full;
    wire do_pop  = pop && !empty;

    assign empty = (count == 0);
    assign full  = (count == DEPTH);
    assign dout  = mem[rp];

    // storage has no reset, so it maps to LUTRAM
    always @(posedge clk)
        if (do_push) mem[wp] <= din;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wp <= 0; rp <= 0; count <= 0;
        end else begin
            if (do_push)
                wp <= (wp == DEPTH - 1) ? {PW{1'b0}} : wp + 1'b1;
            if (do_pop)
                rp <= (rp == DEPTH - 1) ? {PW{1'b0}} : rp + 1'b1;
            count <= count + do_push - do_pop;
        end
    end

endmodule
