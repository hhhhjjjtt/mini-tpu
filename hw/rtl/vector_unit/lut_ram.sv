`timescale 1ns/1ps

// =============================================================================
// lut_ram — one lane's copy of the VEC_LUT table: 256 x 8, one read per cycle
// =============================================================================

module lut_ram (
    input  wire       clk,
    input  wire       we,
    input  wire [7:0] wa,
    input  wire [7:0] wd,
    input  wire [7:0] ra,
    output reg  [7:0] rd
);

    (* ram_style = "block" *) reg [7:0] mem [0:255];

    always @(posedge clk) begin
        if (we) mem[wa] <= wd;
        rd <= mem[ra];
    end

endmodule
