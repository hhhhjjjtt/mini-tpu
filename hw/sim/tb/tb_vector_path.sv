`timescale 1ns/1ps
`include "mini_tpu_config.svh"

// =============================================================================
// tb_vector_path — vmem + matrix_unit + vector_unit, end to end
// =============================================================================
//
// Data path under test:
//   vmem (A) -> matrix_unit -> stream -> vector_unit (VEC_EPILOGUE) -> vmem
//   vmem (A, B) -> vector_unit (VEC_ELTWISE) -> vmem
//
// Golden models mirror INTERFACE_SPEC §9 with 64-bit integer math, written
// from runtime/flexnpu_module.cc apply_scale_tosa.
//
// Bank contention: the unused LOAD (write) and STORE (read) vmem clients
// issue random traffic — STORE reads hit bank 0 (A matrices), LOAD writes
// hit bank 7 (scratch).
//
// Tests:
//   T1  GEMM + REQUANT_I8, per-channel, DOUBLE_ROUND, padded dst pitch
//   T2  K = 1536 split into 3 GEMMs: STORE_I32 -> ACC_I32 (in place) -> ACC_REQUANT_I8
//   T3  PER_TENSOR + ReLU clamp
//   T4  ELTWISE: ADD (FULL), MUL (ROW), SUB i32 (SCALAR, i32 out), MAX, MIN, PASS_A i32->i8
//   T5  stream-row mismatch -> ERR_STREAM_MISMATCH
//   T7  VEC_LUT, T8 VEC_LAYERNORM: bit-exact vs the reference models below
//   T6  VEC_SOFTMAX, with and without CAUSAL: bit-exact vs the integer reference model below, and within
//       +-1 i8 step of true (double) softmax. a: i8 in, len 64; b: i32 in + mask
//       (incl. -10000 blocked entries), padded pitches; c: uniform row and one
//       dominant element; d: len 1024; f: flat rows (all outputs mid-range);
//       e: bad field -> ERR_BAD_FIELD
// =============================================================================

module tb_vector_path;

    localparam integer N     = 16;
    localparam integer VM_AW = 17;

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    // ---------------- front ends ----------------
    reg          m_push = 0, v_push = 0;
    reg  [511:0] m_push_cmd = 0, v_push_cmd = 0;
    wire [23:0]  m_done, v_done;
    wire [95:0]  done_counts = {24'd0, v_done, m_done, 24'd0};
    wire         mf_valid, mf_ready, vf_valid, vf_ready;
    wire [511:0] mf_cmd, vf_cmd;
    wire         m_unit_done, v_unit_done;

    unit_frontend u_mfe (
        .clk(clk), .rst_n(rst_n), .clear(1'b0),
        .push(m_push), .push_cmd(m_push_cmd), .free(), .overflow(),
        .done_counts(done_counts),
        .cmd_valid(mf_valid), .cmd(mf_cmd), .cmd_ready(mf_ready),
        .unit_done(m_unit_done), .done_count(m_done),
        .wait_stall(), .starved()
    );

    unit_frontend u_vfe (
        .clk(clk), .rst_n(rst_n), .clear(1'b0),
        .push(v_push), .push_cmd(v_push_cmd), .free(), .overflow(),
        .done_counts(done_counts),
        .cmd_valid(vf_valid), .cmd(vf_cmd), .cmd_ready(vf_ready),
        .unit_done(v_unit_done), .done_count(v_done),
        .wait_stall(), .starved()
    );

    // ---------------- vmem ----------------
    wire [2:0]         rd_req, rd_gnt, rd_rvalid;
    wire [3*VM_AW-1:0] rd_addr;
    wire [3*128-1:0]   rd_rdata;
    wire [1:0]         wr_req, wr_gnt;
    wire [2*VM_AW-1:0] wr_addr;
    wire [2*128-1:0]   wr_data;

    vmem u_vmem (
        .clk(clk), .rst_n(rst_n),
        .rd_req(rd_req), .rd_addr(rd_addr), .rd_gnt(rd_gnt),
        .rd_rvalid(rd_rvalid), .rd_rdata(rd_rdata),
        .wr_req(wr_req), .wr_addr(wr_addr), .wr_data(wr_data), .wr_gnt(wr_gnt)
    );

    // dummy STORE reads (client 2) into bank 0, dummy LOAD writes (client 0) into bank 7
    reg              st_req = 0, ld_req = 0;
    reg  [VM_AW-1:0] st_addr = 0, ld_addr = 0;
    always @(negedge clk) begin
        st_req  <= ($urandom % 2) == 0;
        st_addr <= $urandom % 16384;                         // bank 0
        ld_req  <= ($urandom % 2) == 0;
        ld_addr <= (7 << 14) | ($urandom % 16384);           // bank 7
    end

    // ---------------- matrix unit ----------------
    reg          wgt_wr_en = 0, wgt_wr_half = 0;
    reg  [8:0]   wgt_wr_addr = 0;
    reg  [127:0] wgt_wr_data = 0;
    wire         s_valid, s_last, s_ready;
    wire [511:0] s_data;
    wire         m_err, v_err;
    wire [7:0]   m_err_code, v_err_code;

    matrix_unit #(.N(`MTPU_ARRAY_N), .VM_AW(VM_AW)) u_mat (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(mf_valid), .cmd(mf_cmd), .cmd_ready(mf_ready),
        .unit_done(m_unit_done), .err(m_err), .err_code(m_err_code), .busy(),
        .vm_rd_req(rd_req[0]), .vm_rd_addr(rd_addr[0 +: VM_AW]), .vm_rd_gnt(rd_gnt[0]),
        .vm_rd_rvalid(rd_rvalid[0]), .vm_rd_rdata(rd_rdata[0 +: 128]),
        .wgt_wr_en(wgt_wr_en), .wgt_wr_half(wgt_wr_half),
        .wgt_wr_addr(wgt_wr_addr), .wgt_wr_data(wgt_wr_data),
        .m_valid(s_valid), .m_data(s_data), .m_last(s_last), .m_ready(s_ready)
    );

    // ---------------- vector unit ----------------
    vector_unit #(.VM_AW(VM_AW)) u_vec (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(vf_valid), .cmd(vf_cmd), .cmd_ready(vf_ready),
        .unit_done(v_unit_done), .err(v_err), .err_code(v_err_code), .busy(),
        .s_valid(s_valid), .s_data(s_data), .s_last(s_last), .s_ready(s_ready),
        .vm_rd_req(rd_req[1]), .vm_rd_addr(rd_addr[VM_AW +: VM_AW]), .vm_rd_gnt(rd_gnt[1]),
        .vm_rd_rvalid(rd_rvalid[1]), .vm_rd_rdata(rd_rdata[128 +: 128]),
        .vm_wr_req(wr_req[1]), .vm_wr_addr(wr_addr[VM_AW +: VM_AW]),
        .vm_wr_data(wr_data[128 +: 128]), .vm_wr_gnt(wr_gnt[1])
    );

    assign rd_req[2]                = st_req;
    assign rd_addr[2*VM_AW +: VM_AW] = st_addr;
    assign wr_req[0]                = ld_req;
    assign wr_addr[0 +: VM_AW]      = ld_addr;
    assign wr_data[0 +: 128]        = {4{$urandom}};

    // =========================================================================
    // Golden models
    // =========================================================================
    function automatic int apply_scale_ref(input int v, input int m, input int sh, input bit dr);
        longint prod, rnd;
        begin
            prod = longint'(v) * longint'(m);
            rnd  = (sh == 0) ? 0 : (longint'(1) << (sh - 1));
            if (dr && sh > 31)
                rnd += (v >= 0) ? (longint'(1) << 30) : -(longint'(1) << 30);
            apply_scale_ref = int'((prod + rnd) >>> sh);
        end
    endfunction

    function automatic int clamp_ref(input int v, input int lo, input int hi);
        clamp_ref = (v < lo) ? lo : (v > hi) ? hi : v;
    endfunction

    // =========================================================================
    // Storage + helpers
    // =========================================================================
    localparam integer MAXR = 32;
    localparam integer MAXK = 1536;

    int  A [0:MAXR-1][0:MAXK-1];     // i8 values, sign-extended (iverilog byte arrays lose sign)
    int  W [0:MAXK-1][0:N-1];
    int  C [0:MAXR-1][0:N-1];
    int  bias [0:N-1], mult [0:N-1], shf [0:N-1];

    integer errors = 0, checked = 0;

    function automatic [VM_AW-1:0] wa(input int byte_addr);
        wa = byte_addr >> 4;
    endfunction

    function automatic int B(input int n);   // bank base (bytes)
        B = n * 32'h40000;
    endfunction

    task automatic put_A(input int rows, input int kk, input int addr, input int pitch);
        int r, t, c; reg [127:0] w;
        begin
            for (r = 0; r < rows; r++)
                for (t = 0; t < kk; t++)
                    A[r][t] = $signed(8'($urandom));
            for (r = 0; r < rows; r++)
                for (c = 0; c < kk / 16; c++) begin
                    for (t = 0; t < 16; t++) w[t*8 +: 8] = A[r][c*16 + t];
                    u_vmem.bd_write(wa(addr + r * pitch + c * 16), w);
                end
        end
    endtask

    task automatic put_W(input int h, input int k0, input int kk);
        int t, c;
        begin
            for (t = 0; t < kk; t++) begin
                for (c = 0; c < N; c++) begin
                    W[k0 + t][c] = $signed(8'($urandom));
                    wgt_wr_data[c*8 +: 8] = W[k0 + t][c];
                end
                @(negedge clk);
                wgt_wr_en = 1; wgt_wr_half = h; wgt_wr_addr = t;
                @(negedge clk);
                wgt_wr_en = 0;
            end
        end
    endtask

    task automatic gold_C(input int rows, input int kk);
        int r, j, t;
        begin
            for (r = 0; r < rows; r++)
                for (j = 0; j < N; j++) begin
                    C[r][j] = 0;
                    for (t = 0; t < kk; t++) C[r][j] += A[r][t] * W[t][j];
                end
        end
    endtask

    task automatic put_requant_table(input int addr, input int shmin, input int shmax);
        int j; reg [127:0] w;
        begin
            for (j = 0; j < N; j++) begin
                bias[j] = int'($urandom % 20001) - 10000;
                mult[j] = 32'h4000_0000 | ($urandom & 32'h3FFF_FFFF);
                shf[j]  = shmin + ($urandom % (shmax - shmin + 1));
                w = 0;
                w[31:0] = bias[j]; w[63:32] = mult[j]; w[71:64] = shf[j];
                u_vmem.bd_write(wa(addr + j * 16), w);
            end
        end
    endtask

    task automatic check_byte(input int addr, input int lane, input int exp, input string what);
        reg [127:0] w; int got;
        begin
            w = u_vmem.bd_read(wa(addr));
            got = $signed(w[lane*8 +: 8]);
            if (got != exp) begin
                if (errors < 12) $display("MISMATCH %s @%h lane %0d: got %0d exp %0d", what, addr, lane, got, exp);
                errors++;
            end
            checked++;
        end
    endtask

    task automatic check_i32(input int addr, input int lane, input int exp, input string what);
        reg [127:0] w; int got;
        begin
            w = u_vmem.bd_read(wa(addr + (lane / 4) * 16));
            got = w[(lane % 4)*32 +: 32];
            if (got != exp) begin
                if (errors < 12) $display("MISMATCH %s @%h lane %0d: got %0d exp %0d", what, addr, lane, got, exp);
                errors++;
            end
            checked++;
        end
    endtask

    // Epilogue requant golden (per-channel or per-tensor)
    task automatic check_epi_i8(input int rows, input int dst, input int pitch, input bit per_tensor,
                                input bit dr, input int zp, input int lo, input int hi, input string what);
        int r, j, e, v;
        begin
            for (r = 0; r < rows; r++)
                for (j = 0; j < N; j++) begin
                    e = per_tensor ? 0 : j;
                    v = apply_scale_ref(C[r][j] + bias[e], mult[e], shf[e], dr) + zp;
                    check_byte(dst + r * pitch, j, clamp_ref(v, lo, hi), what);
                end
        end
    endtask

    // ---------------- command builders ----------------
    function automatic [511:0] mk_gemm(input int act, input int pitch, input int mt,
                                       input int kk, input int h);
        reg [511:0] c;
        begin
            c = 0; c[7:0] = 8'h10;
            c[192 +: 32] = act; c[224 +: 16] = pitch; c[240 +: 16] = mt;
            c[256 +: 16] = kk;  c[272] = h[0];
            mk_gemm = c;
        end
    endfunction

    function automatic [511:0] mk_epi(input int mode, input int flags, input int zp,
                                      input int lo, input int hi, input int rows,
                                      input int dpitch, input int dst, input int part,
                                      input int ppitch, input int params);
        reg [511:0] c;
        begin
            c = 0; c[7:0] = 8'h20; c[15:8] = flags;
            c[192 +: 8] = mode; c[200 +: 8] = zp; c[208 +: 8] = lo; c[216 +: 8] = hi;
            c[224 +: 16] = rows; c[240 +: 16] = dpitch; c[256 +: 32] = dst;
            c[288 +: 32] = part; c[320 +: 16] = ppitch; c[352 +: 32] = params;
            mk_epi = c;
        end
    endfunction

    function automatic [511:0] mk_elt(input int op, input int at, input int bt, input int ot,
                                      input int bc, input int rows, input int cols,
                                      input int a, input int b, input int o,
                                      input int ap, input int bp, input int opitch,
                                      input int params);
        reg [511:0] c;
        begin
            c = 0; c[7:0] = 8'h21;
            c[192 +: 8] = op; c[200 +: 4] = at; c[204 +: 4] = bt; c[208 +: 8] = ot;
            c[216 +: 8] = bc; c[224 +: 16] = rows; c[240 +: 16] = cols;
            c[256 +: 32] = a; c[288 +: 32] = b; c[320 +: 32] = o;
            c[352 +: 16] = ap; c[368 +: 16] = bp; c[384 +: 16] = opitch;
            c[416 +: 32] = params;
            mk_elt = c;
        end
    endfunction

    task automatic push_m(input [511:0] c);
        begin @(negedge clk); m_push_cmd = c; m_push = 1; @(negedge clk); m_push = 0; end
    endtask

    task automatic push_v(input [511:0] c);
        begin @(negedge clk); v_push_cmd = c; v_push = 1; @(negedge clk); v_push = 0; end
    endtask

    task automatic wait_done(input int mt, input int vt, input int limit);
        int c;
        begin
            c = 0;
            while ((m_done < mt || v_done < vt) && c < limit && !v_err && !m_err) begin
                @(posedge clk); c++;
            end
            if (m_done < mt || v_done < vt) begin
                $display("TIMEOUT/ERR: m_done=%0d/%0d v_done=%0d/%0d m_err=%0b(%h) v_err=%0b(%h) vstate=%0d",
                         m_done, mt, v_done, vt, m_err, m_err_code, v_err, v_err_code, u_vec.state);
                errors++;
            end
        end
    endtask

    // ---------------- eltwise test ----------------
    int EA [0:7][0:63];
    int EB [0:7][0:63];
    int ep_a_zp, ep_a_m, ep_a_sh, ep_b_zp, ep_b_m, ep_b_sh, ep_o_m, ep_o_sh, ep_mul_sh;
    int ep_o_zp, ep_lo, ep_hi; bit ep_dr;

    task automatic put_elt_params(input int addr);
        reg [127:0] w;
        begin
            ep_a_zp = int'($urandom % 21) - 10; ep_a_m = 32'h4000_0000 | ($urandom & 32'h3FFF_FFFF); ep_a_sh = 28 + $urandom % 6;
            ep_b_zp = int'($urandom % 21) - 10; ep_b_m = 32'h4000_0000 | ($urandom & 32'h3FFF_FFFF); ep_b_sh = 28 + $urandom % 6;
            ep_o_m  = 32'h4000_0000 | ($urandom & 32'h3FFF_FFFF); ep_o_sh = 30 + $urandom % 6;
            ep_mul_sh = $urandom % 5;
            ep_o_zp = int'($urandom % 41) - 20; ep_lo = -128 + int'($urandom % 20); ep_hi = 127 - int'($urandom % 20);
            ep_dr   = $urandom % 2;
            w = 0; w[31:0] = ep_a_zp; w[63:32] = ep_a_m; w[71:64] = ep_a_sh;            u_vmem.bd_write(wa(addr),      w);
            w = 0; w[31:0] = ep_b_zp; w[63:32] = ep_b_m; w[71:64] = ep_b_sh;            u_vmem.bd_write(wa(addr + 16), w);
            w = 0; w[31:0] = ep_o_m;  w[39:32] = ep_o_sh; w[47:40] = ep_mul_sh;         u_vmem.bd_write(wa(addr + 32), w);
            w = 0; w[7:0] = ep_o_zp;  w[15:8] = ep_lo; w[23:16] = ep_hi; w[24] = ep_dr; u_vmem.bd_write(wa(addr + 48), w);
        end
    endtask

    task automatic put_tensor(input bit is_a, input bit i32, input int rows, input int cols,
                              input int addr, input int pitch);
        int r, c, v; reg [127:0] w;
        begin
            for (r = 0; r < rows; r++) begin
                for (c = 0; c < cols; c++) begin
                    v = i32 ? int'($urandom % 200001) - 100000 : int'($signed(8'($urandom)));
                    if (is_a) EA[r][c] = v; else EB[r][c] = v;
                end
                for (c = 0; c < cols; c += (i32 ? 4 : 16)) begin
                    w = 0;
                    for (v = 0; v < (i32 ? 4 : 16); v++)
                        if (i32) w[v*32 +: 32] = is_a ? EA[r][c+v] : EB[r][c+v];
                        else     w[v*8 +: 8]   = is_a ? EA[r][c+v] : EB[r][c+v];
                    u_vmem.bd_write(wa(addr + r * pitch + (i32 ? c * 4 : c)), w);
                end
            end
        end
    endtask

    function automatic int elt_ref(input int op, input int a, input int b, input bit o_i32);
        int va, vb, r; longint mp, mr;
        begin
            va = apply_scale_ref(a - ep_a_zp, ep_a_m, ep_a_sh, ep_dr);
            vb = apply_scale_ref(b - ep_b_zp, ep_b_m, ep_b_sh, ep_dr);
            case (op)
                0: r = va + vb;
                1: r = va - vb;
                2: begin
                    mp = longint'(va) * longint'(vb);
                    mr = (ep_mul_sh == 0) ? 0 : (longint'(1) << (ep_mul_sh - 1));
                    r  = int'((mp + mr) >>> ep_mul_sh);
                   end
                3: r = (va > vb) ? va : vb;
                4: r = (va < vb) ? va : vb;
                default: r = va;
            endcase
            if (o_i32) elt_ref = r;
            else       elt_ref = clamp_ref(apply_scale_ref(r, ep_o_m, ep_o_sh, ep_dr) + ep_o_zp, ep_lo, ep_hi);
        end
    endfunction

    task automatic run_elt(input int op, input bit a_i32, input bit b_i32, input bit o_i32,
                           input int bc, input int rows, input int cols, input int tag);
        int a, b, o, ap, bp, op_, pr, r, c, bv, vt;
        begin
            a  = B(4) + tag * 32'h1000; b = B(5) + tag * 32'h1000; o = B(6) + tag * 32'h1000;
            pr = B(1) + 32'h400 + tag * 64;
            ap = cols * (a_i32 ? 4 : 1) + 16;          // padded pitches
            bp = cols * (b_i32 ? 4 : 1);
            op_ = cols * (o_i32 ? 4 : 1) + 32;
            put_elt_params(pr);
            put_tensor(1, a_i32, rows, cols, a, ap);
            put_tensor(0, b_i32, (bc == 0) ? rows : 1, (bc == 2) ? 16 : cols, b, bp);
            vt = v_done + 1;
            push_v(mk_elt(op, a_i32, b_i32, o_i32, bc, rows, cols, a, b, o, ap, bp, op_, pr));
            wait_done(m_done, vt, 50000);
            for (r = 0; r < rows; r++)
                for (c = 0; c < cols; c++) begin
                    bv = (bc == 2) ? EB[0][0] : (bc == 1) ? EB[0][c] : EB[r][c];
                    if (o_i32) check_i32(o + r * op_ + (c / 16) * 64, c % 16, elt_ref(op, EA[r][c], bv, 1), "elt");
                    else       check_byte(o + r * op_ + (c / 16) * 16, c % 16, elt_ref(op, EA[r][c], bv, 0), "elt");
                end
        end
    endtask

    // =========================================================================
    // VEC_SOFTMAX reference model (same integer algorithm as the RTL, §5.7)
    // =========================================================================
    int EXP2_LUT [0:255];
    initial for (int i = 0; i < 256; i++) EXP2_LUT[i] = $rtoi($floor(65536.0 * $pow(2.0, -i / 256.0) + 0.5));

    function automatic longint exp2neg_ref(input longint z);       // z >= 0, Q16
        longint zr, ip;
        begin
            zr = z + 128; ip = zr >>> 16;
            exp2neg_ref = (ip > 16) ? 0 : (EXP2_LUT[(zr >>> 8) & 255] >>> ip);
        end
    endfunction

    function automatic int sat32(input longint v);
        sat32 = (v > 64'sd2147483647) ? 32'sh7FFFFFFF : (v < -64'sd2147483648) ? 32'sh80000000 : int'(v);
    endfunction

    // real scale -> (mult, shift) with mult in [2^30, 2^31)
    task automatic real_to_ms(input real r, output int m, output int sh);
        begin
            sh = 30 - $rtoi($floor($ln(r) / $ln(2.0)));
            m  = $rtoi($floor(r * $pow(2.0, sh) + 0.5));
            if (m >= 32'sh40000000 * 2 || m < 0) begin m = m / 2; sh = sh - 1; end
        end
    endtask

    localparam real LOG2E = 1.4426950408889634;
    int SQ [0:1023];            // one row of inputs (q, before zp)
    int SM [0:1023];            // mask row (Q16 log2 units)
    real SMR [0:1023];          // mask row, real (natural-log units)
    int sm_worst = 0, sm_live = 0, sm_diff = 0;
    int sm_dump = 0;            // +dump_smax=<file>: inputs, parameters and RTL outputs of every softmax row
    string sm_dump_file;
    initial if ($value$plusargs("dump_smax=%s", sm_dump_file)) sm_dump = $fopen(sm_dump_file, "w");   // worst step error, outputs above -128, outputs differing from double

    task automatic put_sm_row(input bit i32, input int len, input int addr);
        int c, v; reg [127:0] w;
        for (c = 0; c < len; c += (i32 ? 4 : 16)) begin
            w = 0;
            for (v = 0; v < (i32 ? 4 : 16); v++)
                if (i32) w[v*32 +: 32] = SQ[c+v]; else w[v*8 +: 8] = SQ[c+v];
            u_vmem.bd_write(wa(addr + (i32 ? c * 4 : c)), w);
        end
    endtask

    task automatic put_mask_row(input int len, input int addr);
        int c, v; reg [127:0] w;
        for (c = 0; c < len; c += 4) begin
            for (v = 0; v < 4; v++) w[v*32 +: 32] = SM[c+v];
            u_vmem.bd_write(wa(addr + c * 4), w);
        end
    endtask

    function automatic [511:0] mk_sm(input bit i32, input int otype, input int rows, input int len,
                                     input int ip, input int in_a, input int out_a, input int op,
                                     input int in_sh, input int out_sh, input int mask_a,
                                     input int in_m, input int in_zp, input int out_m, input int out_zp,
                                     input bit has_mask);
        reg [511:0] c;
        begin
            c = 0; c[7:0] = 8'h22; c[8] = has_mask;
            c[192 +: 8] = i32; c[200 +: 8] = otype; c[208 +: 16] = rows; c[224 +: 16] = len;
            c[240 +: 16] = ip; c[256 +: 32] = in_a; c[288 +: 32] = out_a; c[320 +: 16] = op;
            c[336 +: 8] = in_sh; c[344 +: 8] = out_sh; c[352 +: 32] = mask_a;
            c[384 +: 32] = in_m; c[416 +: 32] = in_zp; c[448 +: 32] = out_m; c[480 +: 8] = out_zp;
            mk_sm = c;
        end
    endfunction

    // check one output row against the integer model (exact) and real softmax (+-1)
    task automatic check_sm_row(input int len, input int in_m, input int in_sh, input int in_zp,
                                input real in_scale, input int out_m, input int out_sh, input int out_zp,
                                input real out_inv, input bit has_mask, input int out_a, input string what);
        int t [0:1023];
        longint mx, mnew, bm, sum, fac, bsum, e, R, pq;
        int i, b, exp_q, got, ref_i;
        real xm, den, pr;
        reg [127:0] w;
        begin
            for (i = 0; i < len; i++)
                t[i] = sat32(longint'(apply_scale_ref(SQ[i] - in_zp, in_m, in_sh, 0)) + (has_mask ? SM[i] : 0));
            mx = -64'sd2147483648; sum = 0;
            for (b = 0; b < len; b += 16) begin
                bm = t[b];
                for (i = b + 1; i < b + 16; i++) if (t[i] > bm) bm = t[i];
                mnew = (bm > mx) ? bm : mx;
                fac  = exp2neg_ref(mnew - mx);
                bsum = 0;
                for (i = b; i < b + 16; i++) bsum += exp2neg_ref(mnew - t[i]);
                sum = ((sum * fac) >>> 16) + bsum;
                mx = mnew;
            end
            R = (longint'(1) <<< 46) / sum;
            // real softmax
            xm = -1.0e300;
            for (i = 0; i < len; i++) begin
                pr = (SQ[i] - in_zp) * in_scale + (has_mask ? SMR[i] : 0.0);
                if (pr > xm) xm = pr;
            end
            den = 0.0;
            for (i = 0; i < len; i++) den += $exp((SQ[i] - in_zp) * in_scale + (has_mask ? SMR[i] : 0.0) - xm);
            if (sm_dump) begin
                $fdisplay(sm_dump, "ROW %0d %0d %0d %0d %0d %0d %0d %0d", len, in_m, in_sh, in_zp, out_m, out_sh, out_zp, has_mask);
                for (i = 0; i < len; i++) begin
                    w = u_vmem.bd_read(wa(out_a + (i / 16) * 16));
                    got = $signed(w[(i % 16)*8 +: 8]);
                    $fdisplay(sm_dump, "%0d %0d %0d", SQ[i], has_mask ? SM[i] : 0, got);
                end
            end
            for (i = 0; i < len; i++) begin
                e  = exp2neg_ref(mx - t[i]);
                pq = apply_scale_ref(int'(e), int'(R), 30, 0);
                exp_q = clamp_ref(apply_scale_ref(int'(pq), out_m, out_sh, 0) + out_zp, -128, 127);
                w = u_vmem.bd_read(wa(out_a + (i / 16) * 16));
                got = $signed(w[(i % 16)*8 +: 8]);
                if (got != exp_q) begin
                    if (errors < 12) $display("MISMATCH %s [%0d]: got %0d, model %0d", what, i, got, exp_q);
                    errors++;
                end
                pr = $exp((SQ[i] - in_zp) * in_scale + (has_mask ? SMR[i] : 0.0) - xm) / den;
                ref_i = clamp_ref($rtoi($floor(pr * out_inv + 0.5)) + out_zp, -128, 127);
                if ((got - ref_i > sm_worst) || (ref_i - got > sm_worst)) sm_worst = (got > ref_i) ? got - ref_i : ref_i - got;
                if (got > -128) sm_live++;
                if (got != ref_i) sm_diff++;
                checked++;
            end
        end
    endtask

    // T5 leaves the (sticky) error flag set, so wait on the DONE counter only
    task automatic wait_vdone(input int vt, input int limit);
        int c;
        begin
            c = 0;
            while (v_done < vt && c < limit) begin @(posedge clk); c++; end
            if (v_done < vt) begin
                $display("TIMEOUT: v_done=%0d/%0d vstate=%0d", v_done, vt, u_vec.state);
                errors++;
            end
        end
    endtask

    // one VEC_SOFTMAX command over `rows` rows; returns after DONE
    task automatic run_sm(input bit i32, input int rows, input int len, input real in_scale, input int in_zp,
                          input bit has_mask, input int pad, input int qlo, input int qhi, input int tag,
                          input string what);
        int in_m, in_sh, out_m, out_sh, ip, op, in_a, out_a, mask_a, rr, i, vt;
        int qs [0:7][0:1023];
        real out_inv;
        begin
            out_inv = 256.0;                                   // p in [0, 1] -> i8 with zp -128
            real_to_ms(in_scale * LOG2E * 65536.0, in_m, in_sh);
            real_to_ms(out_inv / 65536.0, out_m, out_sh);
            ip     = len * (i32 ? 4 : 1) + pad;
            op     = len + pad;
            in_a   = B(4) + tag * 32'h8000;
            out_a  = B(6) + tag * 32'h1000;
            mask_a = B(5) + tag * 32'h1000;
            for (i = 0; i < len; i++) begin
                SMR[i] = 0.0;
                if (has_mask) case ($urandom % 4)
                    0:       SMR[i] = -10000.0;                // blocked
                    1:       SMR[i] = -($urandom % 1000) / 250.0;
                    default: SMR[i] = 0.0;
                endcase
                SM[i] = $rtoi($floor(SMR[i] * LOG2E * 65536.0 + 0.5));
            end
            if (has_mask) put_mask_row(len, mask_a);
            for (rr = 0; rr < rows; rr++) begin
                for (i = 0; i < len; i++) begin
                    SQ[i] = qlo + int'($urandom % (qhi - qlo + 1));
                    qs[rr][i] = SQ[i];
                end
                put_sm_row(i32, len, in_a + rr * ip);
            end
            vt = v_done + 1;
            push_v(mk_sm(i32, 0, rows, len, ip, in_a, out_a, op, in_sh, out_sh, mask_a,
                         in_m, in_zp, out_m, -128, has_mask));
            wait_vdone(vt, 400000);
            for (rr = 0; rr < rows; rr++) begin
                for (i = 0; i < len; i++) SQ[i] = qs[rr][i];
                check_sm_row(len, in_m, in_sh, in_zp, in_scale, out_m, out_sh, -128, out_inv, has_mask,
                             out_a + rr * op, what);
            end
        end
    endtask

    // CAUSAL: row r sees columns 0 .. q_pos + r. The reference model blocks the others
    // through its per-row mask; the command itself carries no mask unless has_mask.
    task automatic run_sm_causal(input int rows, input int len, input int q_pos, input bit has_mask,
                                 input int tag, input string what);
        int in_m, in_sh, out_m, out_sh, in_a, out_a, mask_a, rr, i, vt;
        int qs [0:15][0:1023];
        int base [0:1023];
        real base_r [0:1023];
        reg [511:0] c;
        begin
            real_to_ms(0.08 * LOG2E * 65536.0, in_m, in_sh);
            real_to_ms(256.0 / 65536.0, out_m, out_sh);
            in_a   = B(4) + tag * 32'h8000;
            out_a  = B(6) + tag * 32'h1000;
            mask_a = B(5) + tag * 32'h1000;
            for (i = 0; i < len; i++) begin
                SMR[i] = has_mask ? -($urandom % 1000) / 250.0 : 0.0;
                SM[i]  = $rtoi($floor(SMR[i] * LOG2E * 65536.0 + 0.5));
                base[i] = SM[i]; base_r[i] = SMR[i];
            end
            if (has_mask) put_mask_row(len, mask_a);
            for (rr = 0; rr < rows; rr++) begin
                for (i = 0; i < len; i++) begin
                    SQ[i] = -128 + int'($urandom % 256);
                    qs[rr][i] = SQ[i];
                end
                put_sm_row(0, len, in_a + rr * len);
            end
            vt = v_done + 1;
            c = mk_sm(0, 0, rows, len, len, in_a, out_a, len, in_sh, out_sh, mask_a, in_m, 0, out_m, -128, has_mask);
            c[9] = 1'b1;
            c[496 +: 16] = q_pos;
            push_v(c);
            wait_vdone(vt, 400000);
            for (rr = 0; rr < rows; rr++) begin
                for (i = 0; i < len; i++) begin
                    SQ[i]  = qs[rr][i];
                    SM[i]  = (i > q_pos + rr) ? 32'sh80000000 : base[i];
                    SMR[i] = (i > q_pos + rr) ? -1.0e9 : base_r[i];
                end
                check_sm_row(len, in_m, in_sh, 0, 0.08, out_m, out_sh, -128, 256.0, 1, out_a + rr * len, what);
            end
        end
    endtask

    // VEC_LUT: every output is table[(u8) input]
    task automatic run_lut(input int rows, input int cols, input int pad, input int tag, input string what);
        int in_a, out_a, tb_a, rr, i, vt, p;
        int tbl [0:255];
        int qs [0:7][0:1023];
        reg [511:0] c;
        begin
            p     = cols + pad;
            in_a  = B(4) + tag * 32'h8000;
            out_a = B(6) + tag * 32'h1000;
            tb_a  = B(5) + tag * 32'h1000;
            for (i = 0; i < 256; i++) begin tbl[i] = $signed(8'($urandom)); SQ[i] = tbl[i]; end
            put_sm_row(0, 256, tb_a);
            for (rr = 0; rr < rows; rr++) begin
                for (i = 0; i < cols; i++) begin SQ[i] = $signed(8'($urandom)); qs[rr][i] = SQ[i]; end
                put_sm_row(0, cols, in_a + rr * p);
            end
            vt = v_done + 1;
            c = 0; c[7:0] = 8'h24; c[224 +: 16] = rows; c[240 +: 16] = cols;
            c[256 +: 32] = in_a; c[320 +: 32] = out_a; c[352 +: 16] = p; c[384 +: 16] = p; c[416 +: 32] = tb_a;
            push_v(c);
            wait_vdone(vt, 400000);
            for (rr = 0; rr < rows; rr++)
                for (i = 0; i < cols; i++)
                    check_byte(out_a + rr * p + (i / 16) * 16, i % 16, tbl[qs[rr][i] & 255], what);
        end
    endtask

    // VEC_LAYERNORM: integer reference model (§5.7.1), one row
    function automatic int ln_ref_row(input int len, input int im, input int ish, input int eps,
                                      input int cm, input int csh, input int zp, input int col);
        longint s1, s2, d, x, dn, q, r;
        int i, c, mean, u, bit_i;
        begin
            s1 = 0; s2 = 0;
            for (i = 0; i < len; i++) begin
                x = SQ[i];
                if (x > 1048575) x = 1048575; else if (x < -1048575) x = -1048575;
                s1 += x; s2 += x * x;
            end
            d = len * s2 - s1 * s1 + longint'(eps);
            mean = apply_scale_ref(int'(s1), im, ish - 8, 0);
            c = 0;
            while (((d >> 62) & 3) == 0) begin d = d << 2; c++; end
            dn = (d >> 32) & 64'hFFFF_FFFF;
            q = 0;
            for (bit_i = 15; bit_i >= 0; bit_i--)
                if ((q + (longint'(1) <<< bit_i)) * (q + (longint'(1) <<< bit_i)) <= dn) q += longint'(1) <<< bit_i;
            r = (longint'(1) <<< 46) / (2 * q);
            x = SQ[col];
            if (x > 1048575) x = 1048575; else if (x < -1048575) x = -1048575;
            u = apply_scale_ref((int'(x) <<< 8) - mean, int'(r), 45 - c, 0);
            ln_ref_row = clamp_ref(apply_scale_ref(u, cm, csh, 0) + zp, -128, 127);
        end
    endfunction

    task automatic run_ln(input bit i32, input int rows, input int len, input int bits, input int zp,
                          input int tag, input string what);
        int im, ish, cm, csh, eps, in_a, out_a, ip, rr, i, vt;
        int qs [0:7][0:1023];
        reg [511:0] c;
        begin
            real_to_ms(1.0 / len, im, ish);
            real_to_ms(len / (0.07 * 16777216.0), cm, csh);
            eps    = 1 + ($urandom % 1000);
            ip     = len * (i32 ? 4 : 1);
            in_a   = B(4) + tag * 32'h8000;
            out_a  = B(6) + tag * 32'h1000;
            for (rr = 0; rr < rows; rr++) begin
                for (i = 0; i < len; i++) begin
                    SQ[i] = i32 ? int'($urandom % (1 << bits)) - (1 << (bits - 1)) + 300 : $signed(8'($urandom));
                    qs[rr][i] = SQ[i];
                end
                put_sm_row(i32, len, in_a + rr * ip);
            end
            vt = v_done + 1;
            c = 0; c[7:0] = 8'h23; c[192 +: 8] = i32; c[208 +: 16] = rows; c[224 +: 16] = len;
            c[240 +: 16] = ip; c[256 +: 32] = in_a; c[288 +: 32] = out_a; c[320 +: 16] = len;
            c[336 +: 8] = ish; c[344 +: 8] = csh; c[352 +: 32] = eps; c[384 +: 32] = im;
            c[448 +: 32] = cm; c[480 +: 8] = zp;
            push_v(c);
            wait_vdone(vt, 400000);
            for (rr = 0; rr < rows; rr++) begin
                for (i = 0; i < len; i++) SQ[i] = qs[rr][i];
                for (i = 0; i < len; i++)
                    check_byte(out_a + rr * len + (i / 16) * 16, i % 16, ln_ref_row(len, im, ish, eps, cm, csh, zp, i), what);
            end
        end
    endtask

    // =========================================================================
    // Test sequence
    // =========================================================================
    int zp, lo, mt, vt, i;

    initial begin
        repeat (5) @(negedge clk);
        rst_n = 1;
        repeat (2) @(negedge clk);

        // ---- T1: GEMM k=64, 2 tiles + REQUANT_I8 per-channel, DOUBLE_ROUND ----
        put_A(32, 64, B(0), 64);
        put_W(0, 0, 64);
        gold_C(32, 64);
        put_requant_table(B(1), 30, 45);
        zp = 7;
        push_m(mk_gemm(B(0), 64, 2, 64, 0));
        push_v(mk_epi(0, 8'b10, zp, -128, 127, 32, 32, B(3), 0, 0, B(1)));
        wait_done(1, 1, 50000);
        check_epi_i8(32, B(3), 32, 0, 1, zp, -128, 127, "T1");
        $display("T1 done, %0d checked, %0d errors", checked, errors);

        // ---- T2: K=1536 split 3 x 512 ----
        put_A(32, 1536, B(0) + 32'h8000, 1536);
        put_W(0, 0, 512);
        put_W(1, 512, 512);
        put_requant_table(B(1) + 32'h100, 25, 31);
        zp = -3;
        push_m(mk_gemm(B(0) + 32'h8000,        1536, 2, 512, 0));
        push_v(mk_epi(1, 0, 0, 0, 0, 32, 64, B(2), 0, 0, B(1) + 32'h100));             // STORE_I32
        push_m(mk_gemm(B(0) + 32'h8000 + 512,  1536, 2, 512, 1));
        push_v(mk_epi(2, 0, 0, 0, 0, 32, 64, B(2), B(2), 64, B(1) + 32'h100));         // ACC_I32 in place
        wait_done(2, 1, 100000);                         // GEMM0 done -> half 0 free
        put_W(0, 1024, 512);                             // overlaps GEMM1 on half 1
        gold_C(32, 1536);                                // all three weight chunks now known
        push_m(mk_gemm(B(0) + 32'h8000 + 1024, 1536, 2, 512, 0));
        push_v(mk_epi(3, 0, zp, -128, 127, 32, 16, B(3) + 32'h1000, B(2), 64, B(1) + 32'h100)); // ACC_REQUANT_I8
        wait_done(3, 4, 200000);
        check_epi_i8(32, B(3) + 32'h1000, 16, 0, 0, zp, -128, 127, "T2");
        $display("T2 done, %0d checked, %0d errors", checked, errors);

        // ---- T3: PER_TENSOR + ReLU (clamp_min = zp), shift > 31 with DOUBLE_ROUND ----
        put_A(16, 32, B(0) + 32'h20000, 32);
        put_W(1, 0, 32);
        gold_C(16, 32);
        put_requant_table(B(1) + 32'h200, 36, 40);
        zp = -20;
        push_m(mk_gemm(B(0) + 32'h20000, 32, 1, 32, 1));
        push_v(mk_epi(0, 8'b11, zp, zp, 127, 16, 16, B(3) + 32'h2000, 0, 0, B(1) + 32'h200));
        wait_done(4, 5, 50000);
        check_epi_i8(16, B(3) + 32'h2000, 16, 1, 1, zp, zp, 127, "T3");
        $display("T3 done, %0d checked, %0d errors", checked, errors);

        // ---- T4: ELTWISE ----
        run_elt(0, 0, 0, 0, 0, 5, 48, 0);    // ADD    i8 + i8 FULL   -> i8
        run_elt(2, 0, 0, 0, 1, 4, 32, 1);    // MUL    i8 * ROW i8    -> i8 (NoNorm-like)
        run_elt(1, 1, 1, 1, 2, 3, 32, 2);    // SUB    i32 - SCALAR   -> i32
        run_elt(3, 0, 0, 0, 0, 2, 16, 3);    // MAX
        run_elt(4, 0, 1, 0, 0, 2, 32, 4);    // MIN    i8 vs i32
        run_elt(5, 1, 0, 0, 0, 3, 48, 5);    // PASS_A i32 -> i8 rescale
        $display("T4 done, %0d checked, %0d errors", checked, errors);

        // ---- T5: stream mismatch (GEMM 16 rows, EPILOGUE expects 32) ----
        put_A(16, 16, B(0), 16);
        put_W(0, 0, 16);
        mt = m_done; vt = v_done;
        push_m(mk_gemm(B(0), 16, 1, 16, 0));
        push_v(mk_epi(0, 0, 0, -128, 127, 32, 16, B(3) + 32'h3000, 0, 0, B(1)));
        repeat (3000) @(negedge clk);
        if (!v_err || v_err_code != 8'h23 || v_done != vt) begin
            $display("FAIL T5: expected ERR_STREAM_MISMATCH (err=%0b code=%h v_done=%0d)", v_err, v_err_code, v_done);
            errors++;
        end
        if (m_done != mt + 1) begin
            $display("FAIL T5: matrix did not complete its GEMM");
            errors++;
        end

        // ---- T6: VEC_SOFTMAX (errors from T5 are sticky; T6 checks data only) ----
        run_sm(0, 4, 64,   0.08,   3, 0, 0,   -128, 127, 0, "T6a i8");
        run_sm(1, 3, 48,   0.0015, 0, 1, 32,  -20000, 20000, 1, "T6b i32+mask");
        run_sm(0, 1, 16,   0.05,   0, 0, 0,   7, 7, 2, "T6c uniform");
        run_sm(1, 1, 32,   0.01,   0, 0, 0,   -3000, -2000, 3, "T6c dominant");
        for (i = 0; i < 32; i++) SQ[i] = -3000;                 // rewrite: one large element
        SQ[5] = 4000;
        put_sm_row(1, 32, B(4) + 3 * 32'h8000);
        begin
            int im, ish, om, osh, vt2;
            real_to_ms(0.01 * LOG2E * 65536.0, im, ish);
            real_to_ms(256.0 / 65536.0, om, osh);
            vt2 = v_done + 1;
            push_v(mk_sm(1, 0, 1, 32, 128, B(4) + 3 * 32'h8000, B(6) + 3 * 32'h1000, 32, ish, osh, 0,
                         im, 0, om, -128, 0));
            wait_vdone(vt2, 100000);
            check_sm_row(32, im, ish, 0, 0.01, om, osh, -128, 256.0, 0, B(6) + 3 * 32'h1000, "T6c dominant 2");
        end
        run_sm(0, 1, 1024, 0.06,   0, 0, 0,   -128, 127, 4, "T6d len 1024");
        run_sm(0, 8, 16,   0.02,   0, 0, 0,   -128, 127, 5, "T6f flat");     // every output mid-range
        $display("T6 done, %0d checked, %0d errors; vs double softmax: %0d of %0d outputs above -128, %0d differ, worst %0d i8 step(s)",
                 checked, errors, sm_live, checked - 1984, sm_diff, sm_worst);
        run_sm_causal(16, 32, 0,  0, 6, "T6g causal diagonal");       // a whole block blocked in early rows
        run_sm_causal(4,  64, 40, 0, 7, "T6g causal q_pos 40");
        run_sm_causal(4,  48, 13, 1, 8, "T6g causal + mask");
        run_sm_causal(2,  32, 31, 0, 9, "T6g causal nothing blocked");
        if (sm_worst > 1) begin $display("FAIL T6: softmax accuracy worse than 1 step"); errors++; end

        // ---- T7: VEC_LUT, T8: VEC_LAYERNORM ----
        run_lut(3, 48, 0,  10, "T7 LUT");
        run_lut(2, 256, 32, 11, "T7 LUT padded pitch");
        run_ln(0, 3, 64,  0,  5,   12, "T8 LN i8");
        run_ln(1, 3, 768, 16, 0,   13, "T8 LN i32 768");
        run_ln(1, 2, 48,  22, -3,  14, "T8 LN saturating");
        run_ln(1, 1, 1024, 8, 0,   15, "T8 LN len 1024");
        $display("T7, T8 done, %0d checked, %0d errors", checked, errors);

        // ---- T6e: bad field (out_type BF16 is not supported) -> ERR_BAD_FIELD ----
        vt = v_done;
        push_v(mk_sm(0, 2, 1, 16, 16, B(4), B(6), 16, 20, 30, 0, 32'h40000000, 0, 32'h40000000, 0, 0));
        repeat (50) @(negedge clk);
        if (!v_err || v_err_code != 8'h11 || v_done != vt) begin
            $display("FAIL T6e: expected ERR_BAD_FIELD (err=%0b code=%h)", v_err, v_err_code); errors++;
        end

        if (m_err) begin $display("FAIL: matrix error %h", m_err_code); errors++; end

        if (errors == 0) $display("TB PASS: %0d values checked", checked);
        else             $display("TB FAIL: %0d errors (%0d checked)", errors, checked);
        $finish;
    end

    initial begin
        #100_000_000;
        $display("TB FAIL: global timeout");
        $finish;
    end

endmodule
