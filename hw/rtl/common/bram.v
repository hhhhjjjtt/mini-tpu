
//module bram_sdp_64 (
//  input  wire        clk,
//  input  wire        rst,      // active-high

//  // write port (TB/host)
//  input  wire        wr_en,
//  input  wire [7:0]  wr_we,    // byte enables
//  input  wire [8:0]  wr_addr,  // 512 depth
//  input  wire [63:0] wr_din,

//  // read port (DUT)
//  input  wire        rd_en,
//  input  wire [8:0]  rd_addr,
//  output wire [63:0] rd_dout
//);
//  BRAM_SDP_MACRO #(
//    .BRAM_SIZE("36Kb"),
//    .DEVICE("7SERIES"),
//    .WRITE_WIDTH(64),
//    .READ_WIDTH(64),
//    .DO_REG(0),
//    .INIT_FILE("NONE"),
//    .SIM_COLLISION_CHECK("ALL"),
//    .SRVAL(64'h0),
//    .INIT(64'h0),
//    .WRITE_MODE("READ_FIRST")
//  ) u (
//    .DO     (rd_dout),
//    .DI     (wr_din),

//    .RDADDR (rd_addr),
//    .RDCLK  (clk),
//    .RDEN   (rd_en),
//    .REGCE  (1'b1),
//    .RST    (rst),

//    .WE     (wr_we),
//    .WRADDR (wr_addr),
//    .WRCLK  (clk),
//    .WREN   (wr_en)
//  );
//endmodule


`timescale 1ns/1ps

module bram_sdp_64 (
  input  wire        clk,
  input  wire        rst,      // active-high (SIM + macro both use this)

  // write port (TB/host)
  input  wire        wr_en,
  input  wire [7:0]  wr_we,    // byte enables
  input  wire [8:0]  wr_addr,  // 512 depth
  input  wire [63:0] wr_din,

  // read port (DUT)
  input  wire        rd_en,
  input  wire [8:0]  rd_addr,
  output wire [63:0] rd_dout
);

`ifndef SYNTHESIS
  // ============================================================
  // SIMULATION MODEL (xsim/iverilog friendly)
  // - true storage
  // - READ_FIRST on write port
  // - 1-cycle synchronous read (DO_REG=0 equivalent)
  // ============================================================
  reg [63:0] mem [0:511];
  reg [63:0] dout_r;

  assign rd_dout = dout_r;

  integer i;
  initial begin
    for (i = 0; i < 512; i = i + 1) mem[i] = 64'h0;
    dout_r = 64'h0;
  end

  always @(posedge clk) begin
    if (rst) begin
      dout_r <= 64'h0;
    end else begin
      // sync read
      if (rd_en) begin
        dout_r <= mem[rd_addr];
      end

      // byte write
      if (wr_en) begin
        if (wr_we[0]) mem[wr_addr][ 7: 0] <= wr_din[ 7: 0];
        if (wr_we[1]) mem[wr_addr][15: 8] <= wr_din[15: 8];
        if (wr_we[2]) mem[wr_addr][23:16] <= wr_din[23:16];
        if (wr_we[3]) mem[wr_addr][31:24] <= wr_din[31:24];
        if (wr_we[4]) mem[wr_addr][39:32] <= wr_din[39:32];
        if (wr_we[5]) mem[wr_addr][47:40] <= wr_din[47:40];
        if (wr_we[6]) mem[wr_addr][55:48] <= wr_din[55:48];
        if (wr_we[7]) mem[wr_addr][63:56] <= wr_din[63:56];
      end
    end
  end

`else
  // ============================================================
  // SYNTHESIS (FPGA) IMPLEMENTATION
  // ============================================================
  BRAM_SDP_MACRO #(
    .BRAM_SIZE("36Kb"),
    .DEVICE("7SERIES"),
    .WRITE_WIDTH(64),
    .READ_WIDTH(64),
    .DO_REG(0),
    .INIT_FILE("NONE"),
    .SIM_COLLISION_CHECK("ALL"),
    .SRVAL(64'h0),
    .INIT(64'h0),
    .WRITE_MODE("READ_FIRST")
  ) u (
    .DO     (rd_dout),
    .DI     (wr_din),

    .RDADDR (rd_addr),
    .RDCLK  (clk),
    .RDEN   (rd_en),
    .REGCE  (1'b1),
    .RST    (rst),

    .WE     (wr_we),
    .WRADDR (wr_addr),
    .WRCLK  (clk),
    .WREN   (wr_en)
  );
`endif

endmodule
