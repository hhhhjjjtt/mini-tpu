`timescale 1ns/1ps

module bram_extendable #(
    // requested size parameters
    parameter integer DATA_WIDTH        = 128,
    parameter integer DEPTH             = 512,
    parameter integer ADDR_WIDTH        = $clog2(DEPTH),

    // size parameters for one 512*64 bram instance
    parameter integer PHY_DATA_WIDTH    = 64,
    parameter integer PHY_DEPTH         = 512,
    parameter integer PHY_ADDR_WIDTH    = 9
)(
    input  wire                         clk,
    input  wire                         rst,

    input  wire                         wr_en,
    input  wire [DATA_WIDTH/8-1:0]      wr_we,
    input  wire [ADDR_WIDTH-1:0]        wr_addr,
    input  wire [DATA_WIDTH-1:0]        wr_din,

    input  wire                         rd_en,
    input  wire [ADDR_WIDTH-1:0]        rd_addr,
    output wire [DATA_WIDTH-1:0]        rd_dout
);

    localparam integer WIDTH_BLOCKS = DATA_WIDTH / PHY_DATA_WIDTH;
    localparam integer DEPTH_BLOCKS = DEPTH  / PHY_DEPTH;

    localparam integer DEPTH_SEL_WIDTH = (DEPTH_BLOCKS <= 1) ? 1 : $clog2(DEPTH_BLOCKS);

    initial begin
        if (DATA_WIDTH % PHY_DATA_WIDTH != 0) begin
            $error("DATA_WIDTH must be a multiple of PHY_DATA_WIDTH");
        end

        if (DEPTH % PHY_DEPTH != 0) begin
            $error("DEPTH must be a multiple of PHY_DEPTH");
        end

        if (DEPTH < PHY_DEPTH) begin
            $error("DEPTH must be at least PHY_DEPTH");
        end

        if (ADDR_WIDTH < PHY_ADDR_WIDTH) begin
            $error("ADDR_WIDTH must be at least PHY_ADDR_WIDTH");
        end
    end

    wire [PHY_ADDR_WIDTH-1:0] wr_addr_low = wr_addr[PHY_ADDR_WIDTH-1:0];
    wire [PHY_ADDR_WIDTH-1:0] rd_addr_low = rd_addr[PHY_ADDR_WIDTH-1:0];

    wire [DEPTH_SEL_WIDTH-1:0] wr_depth_sel;
    wire [DEPTH_SEL_WIDTH-1:0] rd_depth_sel;

    generate
        if (DEPTH_BLOCKS == 1) begin : gen_single_depth_sel
            assign wr_depth_sel = '0;
            assign rd_depth_sel = '0;
        end else begin : gen_multi_depth_sel
            assign wr_depth_sel = wr_addr[ADDR_WIDTH-1:PHY_ADDR_WIDTH];
            assign rd_depth_sel = rd_addr[ADDR_WIDTH-1:PHY_ADDR_WIDTH];
        end
    endgenerate

    // Delay read depth select because bram_sdp_64 has synchronous read output.
    reg [DEPTH_SEL_WIDTH-1:0] rd_depth_sel_d;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            rd_depth_sel_d <= '0;
        end
        else if (rd_en) begin
            rd_depth_sel_d <= rd_depth_sel;
        end
    end

    wire [DATA_WIDTH-1:0] depth_dout [0:DEPTH_BLOCKS-1];

    genvar d, w;

    generate
        for (d = 0; d < DEPTH_BLOCKS; d = d + 1) begin : gen_depth
            for (w = 0; w < WIDTH_BLOCKS; w = w + 1) begin : gen_width

                bram_sdp_64 u_bram (
                    .clk     (clk),
                    .rst     (rst),

                    .wr_en   (wr_en && (wr_depth_sel == d)),
                    .wr_we   (wr_we [w*8  +: 8]),
                    .wr_addr (wr_addr_low),
                    .wr_din  (wr_din[w*64 +: 64]),

                    .rd_en   (rd_en && (rd_depth_sel == d)),
                    .rd_addr (rd_addr_low),
                    .rd_dout (depth_dout[d][w*64 +: 64])
                );

            end
        end
    endgenerate

    assign rd_dout = depth_dout[rd_depth_sel_d];

endmodule