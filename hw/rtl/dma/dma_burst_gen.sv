`timescale 1ns/1ps

// =============================================================================
// dma_burst_gen — 2D strided AXI4 burst address generator
// =============================================================================
//
// Walks `rows` rows of `row_bytes` bytes starting at `base`, row starts
// `stride` bytes apart, and emits one INCR burst at a time (addr, len). Each
// burst stays inside one row and never crosses a 4 KB boundary (AXI4 rule).
// All inputs 16 B aligned; row_bytes <= 8192 so a burst is <= 256 beats.
//
// The caller presents (addr, len) on AR/AW while `active` and pulses `fire`
// on the handshake.
//
// =============================================================================

module dma_burst_gen #(
    parameter integer AW = 40
)(
    input  wire          clk,
    input  wire          rst_n,

    input  wire          start,
    input  wire [63:0]   base,
    input  wire [15:0]   rows,
    input  wire [15:0]   row_bytes,
    input  wire [31:0]   stride,

    output reg           active,
    output wire [AW-1:0] addr,
    output wire [7:0]    len,           // beats - 1
    input  wire          fire
);

    reg [63:0] row_addr;
    reg [15:0] off;
    reg [15:0] rows_left;

    wire [63:0] cur   = row_addr + {48'd0, off};
    wire [15:0] left  = row_bytes - off;
    wire [15:0] to4k  = 16'h1000 - {4'd0, cur[11:0]};
    wire [15:0] bytes = (left < to4k) ? left : to4k;
    wire        row_end = (bytes == left);

    wire [8:0]  beats = bytes[12:4];                 // bytes <= 4096 -> 1..256 beats
    wire [8:0]  beats_m1 = beats - 1'b1;

    assign addr = cur[AW-1:0];
    assign len  = beats_m1[7:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active <= 1'b0; row_addr <= 0; off <= 0; rows_left <= 0;
        end else if (start) begin
            row_addr  <= base;
            off       <= 0;
            rows_left <= rows;
            active    <= (rows != 0) && (row_bytes != 0);
        end else if (fire && active) begin
            if (row_end) begin
                off       <= 0;
                row_addr  <= row_addr + {32'd0, stride};
                rows_left <= rows_left - 1'b1;
                if (rows_left == 16'd1) active <= 1'b0;
            end else
                off <= off + bytes;
        end
    end

endmodule
