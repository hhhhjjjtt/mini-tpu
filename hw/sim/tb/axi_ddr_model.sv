`timescale 1ns/1ps

// =============================================================================
// axi_ddr_model — AXI4 slave DDR model for testbenches (not synthesizable)
// =============================================================================
//
// One shared memory (`mem`, 128-bit words, byte address >> 4) behind NR read
// ports and NW write ports, each an independent AXI4 slave:
//
//   * Up to 16 outstanding bursts per port; in-order responses (single ID).
//   * Each burst becomes ready LAT_MIN .. LAT_MAX cycles after its AR / last W.
//   * Random AR / R / AW / W / B stalls, off with the plusarg +nostall (for
//     timing measurements closer to the KV260's DDR rate).
//   * Addresses >= ERR_BASE answer SLVERR (reads return 0, writes are dropped).
//   * Protocol checks: 16 B beats, INCR, no 4 KB crossing, WLAST on the right
//     beat, valid held until ready with a stable payload. Violations print
//     "FAIL" and increment `violations`.
//
// Testbenches read and write `mem` directly.
// =============================================================================

module axi_ddr_model #(
    parameter integer AW       = 40,
    parameter integer NR       = 2,
    parameter integer NW       = 1,
    parameter integer WORDS    = 65536,         // 1 MB
    parameter [63:0] ERR_BASE = 64'hF_0000,
    parameter integer LAT_MIN  = 8,
    parameter integer LAT_MAX  = 40
)(
    input  wire                 clk,

    input  wire [NR*AW-1:0]     araddr,
    input  wire [NR*8-1:0]      arlen,
    input  wire [NR*3-1:0]      arsize,
    input  wire [NR*2-1:0]      arburst,
    input  wire [NR-1:0]        arvalid,
    output wire [NR-1:0]        arready,
    output wire [NR*128-1:0]    rdata,
    output wire [NR*2-1:0]      rresp,
    output wire [NR-1:0]        rlast,
    output wire [NR-1:0]        rvalid,
    input  wire [NR-1:0]        rready,

    input  wire [NW*AW-1:0]     awaddr,
    input  wire [NW*8-1:0]      awlen,
    input  wire [NW*3-1:0]      awsize,
    input  wire [NW*2-1:0]      awburst,
    input  wire [NW-1:0]        awvalid,
    output wire [NW-1:0]        awready,
    input  wire [NW*128-1:0]    wdata,
    input  wire [NW*16-1:0]     wstrb,
    input  wire [NW-1:0]        wlast,
    input  wire [NW-1:0]        wvalid,
    output wire [NW-1:0]        wready,
    output wire [NW*2-1:0]      bresp,
    output wire [NW-1:0]        bvalid,
    input  wire [NW-1:0]        bready
);

    reg [127:0] mem [0:WORDS-1];
    integer violations = 0;
    reg [63:0] cyc = 0;
    reg nostall = 0;
    initial nostall = $test$plusargs("nostall");
    always @(posedge clk) cyc <= cyc + 1;

    function automatic bit burst_ok(input [AW-1:0] a, input [7:0] len, input [2:0] size, input [1:0] burst);
        burst_ok = (a[3:0] == 0) && (size == 3'd4) && (burst == 2'b01) &&
                   ({1'b0, a[11:0]} + ({5'd0, len} + 13'd1) * 13'd16 <= 13'h1000);
    endfunction

    genvar g;

    // =========================================================================
    // Read ports
    // =========================================================================
    generate
        for (g = 0; g < NR; g = g + 1) begin : g_rd
            reg [AW-1:0] q_addr [0:15];
            reg [7:0]    q_len  [0:15];
            reg [63:0]   q_t    [0:15];
            reg [3:0]    wp = 0, rp = 0;
            reg [4:0]    cnt = 0;
            reg [7:0]    beat = 0;
            reg          ar_go = 0, r_go = 0;
            reg          ar_hold = 0, r_hold = 0;
            reg [AW+7:0] ar_snap;
            reg [129:0]  r_snap;

            wire [AW-1:0] a   = araddr[g*AW +: AW];
            wire [AW-1:0] ra  = q_addr[rp] + {beat, 4'd0};
            wire          err = (q_addr[rp] >= ERR_BASE);

            always @(negedge clk) begin
                ar_go <= nostall || ($urandom % 4) != 0;
                r_go  <= nostall || ($urandom % 5) != 0;
            end

            assign arready[g]           = ar_go && (cnt < 16);
            assign rvalid[g]            = r_go && (cnt != 0) && (cyc >= q_t[rp]);
            assign rdata[g*128 +: 128]  = err ? 128'd0 : mem[ra >> 4];
            assign rresp[g*2 +: 2]      = err ? 2'b10 : 2'b00;
            assign rlast[g]             = (beat == q_len[rp]);

            always @(posedge clk) begin
                // valid / payload stability
                if (ar_hold && (!arvalid[g] || ar_snap != {a, arlen[g*8 +: 8]})) begin
                    $display("FAIL: ddr rd%0d AR dropped or changed while waiting", g); violations++;
                end
                ar_hold <= arvalid[g] && !arready[g];
                ar_snap <= {a, arlen[g*8 +: 8]};

                if (arvalid[g] && arready[g]) begin
                    if (!burst_ok(a, arlen[g*8 +: 8], arsize[g*3 +: 3], arburst[g*2 +: 2])) begin
                        $display("FAIL: ddr rd%0d bad AR addr=%h len=%0d size=%0d burst=%0d",
                                 g, a, arlen[g*8 +: 8], arsize[g*3 +: 3], arburst[g*2 +: 2]);
                        violations++;
                    end
                    q_addr[wp] <= a;
                    q_len[wp]  <= arlen[g*8 +: 8];
                    q_t[wp]    <= cyc + LAT_MIN + ($urandom % (LAT_MAX - LAT_MIN + 1));
                    wp         <= wp + 1'b1;
                end
                if (rvalid[g] && rready[g]) begin
                    if (rlast[g]) begin rp <= rp + 1'b1; beat <= 0; end
                    else          beat <= beat + 1'b1;
                end
                cnt <= cnt + (arvalid[g] && arready[g]) - (rvalid[g] && rready[g] && rlast[g]);
            end
        end
    endgenerate

    // =========================================================================
    // Write ports
    // =========================================================================
    generate
        for (g = 0; g < NW; g = g + 1) begin : g_wr
            reg [AW-1:0] q_addr [0:15];
            reg [7:0]    q_len  [0:15];
            reg [3:0]    wp = 0, rp = 0;
            reg [4:0]    cnt = 0;
            reg [7:0]    beat = 0;
            reg          burst_err = 0;
            // B queue
            reg [1:0]    b_resp [0:15];
            reg [63:0]   b_t    [0:15];
            reg [3:0]    bwp = 0, brp = 0;
            reg [4:0]    bcnt = 0;
            reg          aw_go = 0, w_go = 0, b_go = 0;
            reg          aw_hold = 0, w_hold = 0;
            reg [AW+7:0] aw_snap;
            reg [144:0]  w_snap;

            wire [AW-1:0] a  = awaddr[g*AW +: AW];
            wire [AW-1:0] wa = q_addr[rp] + {beat, 4'd0};

            always @(negedge clk) begin
                aw_go <= nostall || ($urandom % 4) != 0;
                w_go  <= nostall || ($urandom % 5) != 0;
                b_go  <= nostall || ($urandom % 3) != 0;
            end

            // W accepted only for bursts whose AW has arrived (legal slave behavior)
            assign awready[g]         = aw_go && (cnt < 16);
            assign wready[g]          = w_go && (cnt != 0) && (bcnt < 15);
            assign bvalid[g]          = b_go && (bcnt != 0) && (cyc >= b_t[brp]);
            assign bresp[g*2 +: 2]    = b_resp[brp];

            integer k;
            always @(posedge clk) begin
                if (aw_hold && (!awvalid[g] || aw_snap != {a, awlen[g*8 +: 8]})) begin
                    $display("FAIL: ddr wr%0d AW dropped or changed while waiting", g); violations++;
                end
                aw_hold <= awvalid[g] && !awready[g];
                aw_snap <= {a, awlen[g*8 +: 8]};
                if (w_hold && (!wvalid[g] || w_snap != {wlast[g], wstrb[g*16 +: 16], wdata[g*128 +: 128]})) begin
                    $display("FAIL: ddr wr%0d W dropped or changed while waiting", g); violations++;
                end
                w_hold <= wvalid[g] && !wready[g];
                w_snap <= {wlast[g], wstrb[g*16 +: 16], wdata[g*128 +: 128]};

                if (awvalid[g] && awready[g]) begin
                    if (!burst_ok(a, awlen[g*8 +: 8], awsize[g*3 +: 3], awburst[g*2 +: 2])) begin
                        $display("FAIL: ddr wr%0d bad AW addr=%h len=%0d", g, a, awlen[g*8 +: 8]);
                        violations++;
                    end
                    q_addr[wp] <= a;
                    q_len[wp]  <= awlen[g*8 +: 8];
                    wp         <= wp + 1'b1;
                end
                if (wvalid[g] && wready[g]) begin
                    if (wlast[g] != (beat == q_len[rp])) begin
                        $display("FAIL: ddr wr%0d WLAST=%0b on beat %0d of %0d", g, wlast[g], beat, q_len[rp] + 1);
                        violations++;
                    end
                    if (q_addr[rp] < ERR_BASE) begin
                        for (k = 0; k < 16; k = k + 1)
                            if (wstrb[g*16 + k]) mem[wa >> 4][k*8 +: 8] = wdata[g*128 + k*8 +: 8];
                    end
                    if (beat == q_len[rp]) begin
                        b_resp[bwp] <= (q_addr[rp] >= ERR_BASE) ? 2'b10 : 2'b00;
                        b_t[bwp]    <= cyc + LAT_MIN + ($urandom % (LAT_MAX - LAT_MIN + 1));
                        bwp         <= bwp + 1'b1;
                        rp          <= rp + 1'b1;
                        beat        <= 0;
                    end else
                        beat <= beat + 1'b1;
                end
                cnt  <= cnt + (awvalid[g] && awready[g]) - (wvalid[g] && wready[g] && beat == q_len[rp]);
                bcnt <= bcnt + (wvalid[g] && wready[g] && beat == q_len[rp]) - (bvalid[g] && bready[g]);
                if (bvalid[g] && bready[g]) brp <= brp + 1'b1;
            end
        end
    endgenerate

endmodule
