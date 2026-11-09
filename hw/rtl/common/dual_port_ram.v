`timescale 1ns/1ps

// =============================================================================
// dual_port_ram — simple dual-port RAM: one write port with byte enables, one
// read port. Inferred, so iverilog and Vivado read this same description;
// on the KV260 it maps to block RAM.
//
//   write: mem[wr_addr] byte b <= wr_din byte b, for each wr_we[b], when wr_en
//   read:  rd_dout <= mem[rd_addr] one cycle after rd_en; held while rd_en = 0
//          reading the address being written returns the old word
//   rst:   clears rd_dout (synchronous); the memory starts as zeros
// =============================================================================

module dual_port_ram #(
    parameter integer W     = 128,              // word bits, a multiple of 8
    parameter integer DEPTH = 1024,
    parameter integer AW    = $clog2(DEPTH)
)(
    input  wire             clk,
    input  wire             rst,

    input  wire             wr_en,
    input  wire [W/8-1:0]   wr_we,
    input  wire [AW-1:0]    wr_addr,
    input  wire [W-1:0]     wr_din,

    input  wire             rd_en,
    input  wire [AW-1:0]    rd_addr,
    output reg  [W-1:0]     rd_dout
);

    (* ram_style = "block" *) reg [W-1:0] mem [0:DEPTH-1];

    integer i, b;
    initial begin
        for (i = 0; i < DEPTH; i = i + 1) mem[i] = {W{1'b0}};
        rd_dout = {W{1'b0}};
    end

    always @(posedge clk) begin
        if (wr_en)
            for (b = 0; b < W / 8; b = b + 1)
                if (wr_we[b]) mem[wr_addr][b * 8 +: 8] <= wr_din[b * 8 +: 8];
    end

    always @(posedge clk) begin
        if (rst)
            rd_dout <= {W{1'b0}};
        else if (rd_en)
            rd_dout <= mem[rd_addr];
    end

endmodule
