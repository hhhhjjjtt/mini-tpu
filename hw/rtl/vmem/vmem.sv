`timescale 1ns/1ps

// =============================================================================
// vmem — banked on-chip vector memory
// =============================================================================
//
// TENSOR_CORE_UPDATE_SPEC.md §4.2 (D3), INTERFACE_SPEC.md §1, §2.4.
//
//   * NB banks x BANK_WORDS words x 128 b, NB = 1, 2, 4 or 8. Default 8 x 16 K x
//     16 B = 2 MB (all 64 URAM on xck26).
//   * Word address = byte address >> 4. Bank = word_addr[top bits]
//     (contiguous 256 KB banks, so the compiler can place tensors per bank).
//   * Each bank: one read port + one write port per cycle.
//   * Read clients (NR):  0 MATRIX, 1 VECTOR, 2 STORE DMA
//     Write clients (NW): 0 LOAD DMA, 1 VECTOR
//   * Per bank, per port type: round-robin arbitration among requesting
//     clients. gnt is combinational in the request cycle.
//   * Reads: rdata/rvalid exactly 1 cycle after the granted request
//     (per client in order). Writes: full 128-bit words, written on grant.
//
// Hazards between clients are compiler-managed (INTERFACE_SPEC §4.2 rule 5).
//
// =============================================================================

module vmem #(
    parameter integer NB         = 8,
    parameter integer BANK_WORDS = 16384,
    parameter integer NR         = 3,
    parameter integer NW         = 2,
    parameter integer DW         = 128,
    parameter integer AW         = $clog2(NB * BANK_WORDS)   // word address
)(
    input  wire                 clk,
    input  wire                 rst_n,

    // ---- Read clients ----
    input  wire [NR-1:0]        rd_req,
    input  wire [NR*AW-1:0]     rd_addr,
    output reg  [NR-1:0]        rd_gnt,
    output reg  [NR-1:0]        rd_rvalid,
    output wire [NR*DW-1:0]     rd_rdata,

    // ---- Write clients ----
    input  wire [NW-1:0]        wr_req,
    input  wire [NW*AW-1:0]     wr_addr,
    input  wire [NW*DW-1:0]     wr_data,
    output reg  [NW-1:0]        wr_gnt
);

    localparam integer BW  = (NB <= 1) ? 1 : $clog2(NB);
    localparam integer IW  = $clog2(BANK_WORDS);
    localparam integer RCW = (NR <= 1) ? 1 : $clog2(NR);
    localparam integer WCW = (NW <= 1) ? 1 : $clog2(NW);

    function automatic [BW-1:0] bank_of(input [AW-1:0] a);
        bank_of = a >> IW;
    endfunction

    // ------------------------------------------------------------
    // Arbitration (combinational) — round-robin per bank
    // ------------------------------------------------------------
    reg [RCW-1:0] rr_rd [0:NB-1];
    reg [WCW-1:0] rr_wr [0:NB-1];

    reg           b_rd_en  [0:NB-1];
    reg [IW-1:0]  b_rd_idx [0:NB-1];
    reg [RCW-1:0] b_rd_cli [0:NB-1];
    reg           b_wr_en  [0:NB-1];
    reg [IW-1:0]  b_wr_idx [0:NB-1];
    reg [DW-1:0]  b_wr_dat [0:NB-1];
    reg [WCW-1:0] b_wr_cli [0:NB-1];

    integer b, i, c;
    always @(*) begin
        rd_gnt = 0;
        wr_gnt = 0;
        for (b = 0; b < NB; b = b + 1) begin
            b_rd_en[b]  = 1'b0;
            b_rd_idx[b] = 0;
            b_rd_cli[b] = 0;
            for (i = 0; i < NR; i = i + 1) begin
                c = (rr_rd[b] + 1 + i) % NR;
                if (!b_rd_en[b] && rd_req[c] && bank_of(rd_addr[c*AW +: AW]) == b) begin
                    b_rd_en[b]  = 1'b1;
                    b_rd_idx[b] = rd_addr[c*AW +: IW];
                    b_rd_cli[b] = c;
                    rd_gnt[c]   = 1'b1;
                end
            end
            b_wr_en[b]  = 1'b0;
            b_wr_idx[b] = 0;
            b_wr_dat[b] = 0;
            b_wr_cli[b] = 0;
            for (i = 0; i < NW; i = i + 1) begin
                c = (rr_wr[b] + 1 + i) % NW;
                if (!b_wr_en[b] && wr_req[c] && bank_of(wr_addr[c*AW +: AW]) == b) begin
                    b_wr_en[b]  = 1'b1;
                    b_wr_idx[b] = wr_addr[c*AW +: IW];
                    b_wr_dat[b] = wr_data[c*DW +: DW];
                    b_wr_cli[b] = c;
                    wr_gnt[c]   = 1'b1;
                end
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (b = 0; b < NB; b = b + 1) begin
                rr_rd[b] <= NR - 1;
                rr_wr[b] <= NW - 1;
            end
        end else begin
            for (b = 0; b < NB; b = b + 1) begin
                if (b_rd_en[b]) rr_rd[b] <= b_rd_cli[b];
                if (b_wr_en[b]) rr_wr[b] <= b_wr_cli[b];
            end
        end
    end

    // ------------------------------------------------------------
    // Banks
    // ------------------------------------------------------------
    wire [DW-1:0] b_rd_dout [0:NB-1];

    // Always 8 scopes: banks past NB are one-word stubs, so the backdoor below elaborates.
    genvar gb;
    generate
        for (gb = 0; gb < 8; gb = gb + 1) begin : gen_bank
            (* ram_style = "ultra" *)
            reg [DW-1:0] mem [0:(gb < NB) ? BANK_WORDS-1 : 0];

            if (gb < NB) begin : g_port
                reg [DW-1:0] dout;

                always @(posedge clk) begin
                    if (b_wr_en[gb])
                        mem[b_wr_idx[gb]] <= b_wr_dat[gb];
                    if (b_rd_en[gb])
                        dout <= mem[b_rd_idx[gb]];
                end
                assign b_rd_dout[gb] = dout;
            end
        end
    endgenerate

    // ------------------------------------------------------------
    // Read return: per client, 1 cycle after grant
    // ------------------------------------------------------------
    reg [BW-1:0] rd_bank_d [0:NR-1];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_rvalid <= 0;
            for (c = 0; c < NR; c = c + 1)
                rd_bank_d[c] <= 0;
        end else begin
            rd_rvalid <= rd_gnt;
            for (c = 0; c < NR; c = c + 1)
                if (rd_gnt[c])
                    rd_bank_d[c] <= bank_of(rd_addr[c*AW +: AW]);
        end
    end

    genvar gc;
    generate
        for (gc = 0; gc < NR; gc = gc + 1) begin : gen_rdata
            assign rd_rdata[gc*DW +: DW] = b_rd_dout[rd_bank_d[gc]];
        end
    endgenerate

`ifndef SYNTHESIS
    // ------------------------------------------------------------
    // Simulation backdoor (TB preload / checking)
    // ------------------------------------------------------------
    task automatic bd_write(input [AW-1:0] a, input [DW-1:0] d);
        case (bank_of(a))
            0: gen_bank[0].mem[a[IW-1:0]] = d;
            1: gen_bank[1].mem[a[IW-1:0]] = d;
            2: gen_bank[2].mem[a[IW-1:0]] = d;
            3: gen_bank[3].mem[a[IW-1:0]] = d;
            4: gen_bank[4].mem[a[IW-1:0]] = d;
            5: gen_bank[5].mem[a[IW-1:0]] = d;
            6: gen_bank[6].mem[a[IW-1:0]] = d;
            7: gen_bank[7].mem[a[IW-1:0]] = d;
        endcase
    endtask

    function automatic [DW-1:0] bd_read(input [AW-1:0] a);
        case (bank_of(a))
            0: bd_read = gen_bank[0].mem[a[IW-1:0]];
            1: bd_read = gen_bank[1].mem[a[IW-1:0]];
            2: bd_read = gen_bank[2].mem[a[IW-1:0]];
            3: bd_read = gen_bank[3].mem[a[IW-1:0]];
            4: bd_read = gen_bank[4].mem[a[IW-1:0]];
            5: bd_read = gen_bank[5].mem[a[IW-1:0]];
            6: bd_read = gen_bank[6].mem[a[IW-1:0]];
            7: bd_read = gen_bank[7].mem[a[IW-1:0]];
        endcase
    endfunction
`endif

endmodule
