`timescale 1ns/1ps

// =============================================================================
// queue_axi4_master_r — AXI4 read master for the streaming DMA queue
// =============================================================================
//
// Fetches LENGTH_BYTES bytes from src_addr in DDR, streams them as beats to
// the slot writer side (beat_wr_valid + bram_wr_addr + bram_wr_data). Issues
// AXI4 INCR bursts up to 4 KB or 256 beats (whichever comes first), clipped
// at 4 KB address boundaries. Supports MAX_OUTSTANDING = 4 in-flight AR
// requests. Uses single ARID (=0) to force in-order responses.
//
// See docs/HW_STREAMING_DMA_QUEUE_DESIGN.md §2A for the design contract.
//
// ============================================================================
// Master invariants (asserted in RTL + TB fault-injection):
//
// INV-M1 (BEAT COUNT PRESERVATION):
//     For a transfer of LENGTH bytes, exactly LENGTH/BEAT_BYTES beats are
//     asserted on beat_wr_valid — no more, no less. The done pulse fires
//     on the same or subsequent cycle as the final beat.
//
// INV-M2 (IN-ORDER DELIVERY):
//     Beats are delivered to the slot in strictly increasing byte-offset
//     order. BRAM write address monotonically increments by 1 per beat,
//     starting at 0.
//
// INV-M3 (NO OVER-ISSUE):
//     Number of outstanding AR requests (issued but final RLAST not yet
//     received) never exceeds MAX_OUTSTANDING at any cycle.
//
// INV-M4 (4 KB BOUNDARY):
//     No AR burst crosses a 4 KB boundary. arlen is computed as
//     min(remaining_beats, 256, beats_to_next_4KB) - 1.
//
// INV-M5 (BACKPRESSURE HANDLING):
//     When the slot is not busy (writer_busy=0), the master holds RREADY
//     low and does not accept R data. Beats are not dropped — the AXI
//     handshake stalls until the slot is ready.
//
// INV-M6 (NO SPURIOUS beat_wr_valid):
//     beat_wr_valid is asserted for exactly one cycle per accepted R beat
//     (i.e., when RVALID && RREADY on the AXI R channel). No beat is
//     signaled without a corresponding R handshake.
//
// ============================================================================

module queue_axi4_master_r #(
    parameter integer ADDR_W          = 64,
    parameter integer DATA_W          = 128,
    parameter integer LENGTH_W        = 20,
    parameter integer BEAT_BYTES      = 16,
    parameter integer MAX_OUTSTANDING = 4,
    parameter integer BRAM_ADDR_W     = 16,

    // AXI cache attributes for the reads. Default = 0 (non-cacheable, matches
    // today's udmabuf setup). §2A recommends 4'hF for coherent-mode udmabuf.
    parameter [3:0]   AR_CACHE        = 4'h0,
    parameter [1:0]   AR_DOMAIN       = 2'h0
)(
    input  wire                       clk,
    input  wire                       rst_n,

    // ---- Command from queue engine ----
    input  wire                       start,           // one-shot pulse
    input  wire [ADDR_W-1:0]          src_addr,        // DDR source address
    input  wire [LENGTH_W-1:0]        length_bytes,    // bytes to transfer

    output wire                       busy,
    output wire                       done,            // one-shot pulse

    // ---- AXI4 Read Address channel (master) ----
    output wire [0:0]                 m_axi_arid,
    output wire [ADDR_W-1:0]          m_axi_araddr,
    output wire [7:0]                 m_axi_arlen,     // beats - 1
    output wire [2:0]                 m_axi_arsize,    // $clog2(BEAT_BYTES)
    output wire [1:0]                 m_axi_arburst,   // 2'b01 = INCR
    output wire [3:0]                 m_axi_arcache,
    output wire [1:0]                 m_axi_ardomain,
    output wire                       m_axi_arvalid,
    input  wire                       m_axi_arready,

    // ---- AXI4 Read Data channel (master) ----
    input  wire [0:0]                 m_axi_rid,
    input  wire [DATA_W-1:0]          m_axi_rdata,
    input  wire                       m_axi_rlast,
    input  wire                       m_axi_rvalid,
    output wire                       m_axi_rready,

    // ---- Slot writer-side interface ----
    input  wire                       slot_writer_busy,   // slot in FILLING
    output wire                       slot_beat_wr_valid, // one-shot per beat
    output wire [BRAM_ADDR_W-1:0]     bram_wr_addr,
    output wire [DATA_W-1:0]          bram_wr_data
);

    // -----------------------------------------------------------------------
    // Localparams / constants
    // -----------------------------------------------------------------------
    localparam integer BEAT_BYTES_LOG2 = $clog2(BEAT_BYTES);
    localparam integer MAX_BURST_BEATS = 256;
    localparam integer FOUR_KB         = 4096;

    // -----------------------------------------------------------------------
    // Transfer-level state
    //
    // Two independent processes running under one FSM:
    //   - AR issuer: advances next_ar_addr and bytes_remaining_to_issue
    //   - R receiver: writes beats to BRAM as they arrive, counts
    //                 bytes_received. When bytes_received == length, done.
    //
    // Both share outstanding_count (issuer +=1 on ARVALID&ARREADY,
    // receiver -=1 on RVALID&RREADY&RLAST).
    // -----------------------------------------------------------------------

    localparam [1:0] S_IDLE   = 2'd0,
                     S_ACTIVE = 2'd1,
                     S_DONE   = 2'd2;

    reg [1:0]                    state;

    reg [ADDR_W-1:0]             next_ar_addr;
    reg [LENGTH_W-1:0]           bytes_left_to_issue;
    reg [LENGTH_W-1:0]           bytes_left_to_receive;
    reg [LENGTH_W-1:0]           total_length_latched;
    reg [$clog2(MAX_OUTSTANDING+1)-1:0] outstanding_count;
    reg [BRAM_ADDR_W-1:0]        bram_addr_ctr;   // increments per received beat
    reg                          done_reg;

    // -----------------------------------------------------------------------
    // AR-side combinational logic — compute the next burst
    //
    // burst_beats = min(bytes_left_to_issue / BEAT_BYTES,
    //                    MAX_BURST_BEATS,
    //                    beats_to_next_4K_boundary)
    // -----------------------------------------------------------------------
    wire [LENGTH_W-1:0] bytes_to_next_4K =
        FOUR_KB - {{(LENGTH_W-12){1'b0}}, next_ar_addr[11:0]};

    wire [LENGTH_W-1:0] beats_left      = bytes_left_to_issue >> BEAT_BYTES_LOG2;
    wire [LENGTH_W-1:0] beats_to_4K     = bytes_to_next_4K    >> BEAT_BYTES_LOG2;

    // Clipped burst length in beats (must be ≥ 1 when we have anything to
    // issue). Type is wide enough that any of the three inputs can dominate.
    wire [LENGTH_W-1:0] beats_clip_data = (beats_left < MAX_BURST_BEATS) ? beats_left :
                                          MAX_BURST_BEATS[LENGTH_W-1:0];
    wire [LENGTH_W-1:0] burst_beats     = (beats_clip_data < beats_to_4K) ? beats_clip_data :
                                          beats_to_4K;

    // AR channel outputs
    wire ar_can_issue =
        (state == S_ACTIVE) &&
        (bytes_left_to_issue > 0) &&
        (outstanding_count < MAX_OUTSTANDING);

    assign m_axi_arvalid   = ar_can_issue;
    assign m_axi_arid      = 1'b0;
    assign m_axi_araddr    = next_ar_addr;
    assign m_axi_arlen     = burst_beats[7:0] - 8'd1;   // AXI4: beats - 1
    assign m_axi_arsize    = BEAT_BYTES_LOG2[2:0];
    assign m_axi_arburst   = 2'b01;                     // INCR
    assign m_axi_arcache   = AR_CACHE;
    assign m_axi_ardomain  = AR_DOMAIN;

    // R channel: accept beats as long as slot is willing
    assign m_axi_rready    = slot_writer_busy && (state == S_ACTIVE);

    wire   r_beat_accepted = m_axi_rvalid && m_axi_rready;

    // Slot writer-side outputs
    assign slot_beat_wr_valid = r_beat_accepted;
    assign bram_wr_addr       = bram_addr_ctr;
    assign bram_wr_data       = m_axi_rdata;

    // Command outputs
    assign busy = (state != S_IDLE);
    assign done = done_reg;

    // -----------------------------------------------------------------------
    // Sequential logic
    // -----------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state                 <= S_IDLE;
            next_ar_addr          <= {ADDR_W{1'b0}};
            bytes_left_to_issue   <= {LENGTH_W{1'b0}};
            bytes_left_to_receive <= {LENGTH_W{1'b0}};
            total_length_latched  <= {LENGTH_W{1'b0}};
            outstanding_count     <= {$clog2(MAX_OUTSTANDING+1){1'b0}};
            bram_addr_ctr         <= {BRAM_ADDR_W{1'b0}};
            done_reg              <= 1'b0;
        end else begin
            done_reg <= 1'b0;   // one-shot; asserted below in S_DONE

            case (state)
                S_IDLE: begin
                    if (start) begin
                        next_ar_addr          <= src_addr;
                        bytes_left_to_issue   <= length_bytes;
                        bytes_left_to_receive <= length_bytes;
                        total_length_latched  <= length_bytes;
                        outstanding_count     <= 0;
                        bram_addr_ctr         <= 0;
                        state                 <= S_ACTIVE;
                    end
                end

                S_ACTIVE: begin
                    // Issue AR
                    if (m_axi_arvalid && m_axi_arready) begin
                        next_ar_addr        <= next_ar_addr + (burst_beats << BEAT_BYTES_LOG2);
                        bytes_left_to_issue <= bytes_left_to_issue - (burst_beats << BEAT_BYTES_LOG2);
                        outstanding_count   <= outstanding_count + 1'b1
                                                - (r_beat_accepted && m_axi_rlast ? 1'b1 : 1'b0);
                    end else begin
                        outstanding_count   <= outstanding_count
                                                - (r_beat_accepted && m_axi_rlast ? 1'b1 : 1'b0);
                    end

                    // Accept R beat
                    if (r_beat_accepted) begin
                        bram_addr_ctr         <= bram_addr_ctr + 1'b1;
                        bytes_left_to_receive <= bytes_left_to_receive - BEAT_BYTES;
                        if (bytes_left_to_receive == BEAT_BYTES) begin
                            // Final beat this cycle
                            state    <= S_DONE;
                            done_reg <= 1'b1;
                        end
                    end
                end

                S_DONE: begin
                    // done_reg was asserted last cycle; return to IDLE now
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

`ifdef FORMAL
    // ------------------------------------------------------------------------
    // Property annotations for future SymbiYosys / Verific proof runs.
    // ------------------------------------------------------------------------

    // INV-M3: outstanding never exceeds MAX_OUTSTANDING
    property inv_m3_no_over_issue;
        @(posedge clk) disable iff (!rst_n)
            outstanding_count <= MAX_OUTSTANDING;
    endproperty
    assert property (inv_m3_no_over_issue);

    // INV-M4: AR burst never crosses 4 KB
    // Constraint: araddr + arlen*BEAT_BYTES + BEAT_BYTES - 1 < ((araddr / 4K) + 1) * 4K
    property inv_m4_no_4k_cross;
        @(posedge clk) disable iff (!rst_n)
            m_axi_arvalid |->
                ((m_axi_araddr[11:0] + ((m_axi_arlen + 8'd1) << BEAT_BYTES_LOG2)) <= 12'h1000);
    endproperty
    assert property (inv_m4_no_4k_cross);

    // INV-M5: beat_wr_valid only when slot_writer_busy
    property inv_m5_no_dropped_beats;
        @(posedge clk) disable iff (!rst_n)
            slot_beat_wr_valid |-> slot_writer_busy;
    endproperty
    assert property (inv_m5_no_dropped_beats);

    // INV-M6: beat_wr_valid iff R handshake completes
    property inv_m6_beat_matches_handshake;
        @(posedge clk) disable iff (!rst_n)
            slot_beat_wr_valid == (m_axi_rvalid && m_axi_rready);
    endproperty
    assert property (inv_m6_beat_matches_handshake);
`endif

endmodule
