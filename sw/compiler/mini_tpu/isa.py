"""Program format, descriptor encoders and decoder (INTERFACE_SPEC.md §3-§5).

Constants are parsed from hw/interface/*.h, so C and Python share one definition:
MINI_TPU_OP_LOAD in C is C.OP_LOAD here.
"""
import re
import struct
from pathlib import Path
from types import SimpleNamespace

INTERFACE = Path(__file__).resolve().parents[3] / "hw" / "interface"


def _load_constants():
    values = {}
    for name in ("mini_tpu_config.h", "mini_tpu_isa.h", "mini_tpu_regs.h", "mini_tpu_errors.h"):
        text = (INTERFACE / name).read_text(encoding="utf-8")
        for m in re.finditer(r"^#define MINI_TPU_(\w+)\s+(0x[0-9A-Fa-f]+|\d+)u?\b", text, re.M):
            values[m.group(1)] = int(m.group(2), 0)
    return SimpleNamespace(**values)


C = _load_constants()

DESC = C.DESC_SIZE
UNIT_OF = {C.OP_LOAD: 0, C.OP_LOAD_WGT: 0, C.OP_STORE: 3, C.OP_GEMM: 1, C.OP_LOAD_WGT_VM: 1,
           C.OP_VEC_EPILOGUE: 2, C.OP_VEC_ELTWISE: 2, C.OP_VEC_SOFTMAX: 2, C.OP_VEC_LAYERNORM: 2, C.OP_VEC_LUT: 2}
MEM_OPS = (C.OP_LOAD, C.OP_LOAD_WGT, C.OP_STORE)


def arch_word(n=C.TILE_N, units=C.UNITS_MASK, vmem_kb=C.VMEM_KB):
    return n | units << 8 | vmem_kb << 16


def _put(d, fmt, off, *v):
    struct.pack_into("<" + fmt, d, off, *v)


def new(opcode, flags=0):
    d = bytearray(DESC)
    d[C.D_OPCODE], d[C.D_FLAGS] = opcode, flags
    return d


def wait(d, unit, dist):
    """Adds a wait condition (§4.2): all but the newest `dist` commands of `unit`."""
    d[C.D_WAIT_MASK] |= 1 << unit
    _put(d, "H", 0x04 + 2 * unit, dist)
    return d


# ---- memory ops (§5.1-§5.3) ----
def load(region, ddr_off, vmem, rows, row_bytes, ddr_stride, vmem_stride, store=False):
    d = new(C.OP_STORE if store else C.OP_LOAD)
    d[C.D_REGION] = region
    _put(d, "I", C.D_DDR_OFFSET, ddr_off)
    _put(d, "IHHII", C.LD_VMEM_ADDR, vmem, rows, row_bytes, ddr_stride, vmem_stride)
    return d


def store(region, ddr_off, vmem, rows, row_bytes, ddr_stride, vmem_stride):
    return load(region, ddr_off, vmem, rows, row_bytes, ddr_stride, vmem_stride, store=True)


def load_wgt(region, ddr_off, half, k):
    d = new(C.OP_LOAD_WGT)
    d[C.D_REGION] = region
    _put(d, "I", C.D_DDR_OFFSET, ddr_off)
    d[C.LW_HALF] = half
    _put(d, "H", C.LW_K, k)
    return d


# ---- matrix (§5.4, §5.4.1) ----
def gemm(act, act_pitch, m_tiles, k, half):
    d = new(C.OP_GEMM)
    _put(d, "IHHH", C.GM_ACT_ADDR, act, act_pitch, m_tiles, k)
    d[C.GM_WGT_HALF] = half
    return d


def load_wgt_vm(src, src_pitch, k, half, transpose=False):
    d = new(C.OP_LOAD_WGT_VM, C.WV_F_TRANSPOSE if transpose else 0)
    _put(d, "IH", C.WV_SRC_ADDR, src, src_pitch)
    _put(d, "H", C.WV_K, k)
    d[C.WV_WGT_HALF] = half
    return d


# ---- vector (§5.5-§5.7) ----
def epilogue(mode, flags, out_zp, clamp_min, clamp_max, rows, dst_pitch, dst, params, partial=0, partial_pitch=0):
    d = new(C.OP_VEC_EPILOGUE, flags)
    d[C.EP_MODE] = mode
    _put(d, "bbb", C.EP_OUT_ZP, out_zp, clamp_min, clamp_max)
    _put(d, "HHII", C.EP_ROWS, rows, dst_pitch, dst, partial)
    _put(d, "H", C.EP_PARTIAL_PITCH, partial_pitch)
    _put(d, "I", C.EP_PARAMS_ADDR, params)
    return d


def eltwise(op, a_type, b_type, out_type, b_bcast, rows, cols, a, b, out, a_pitch, b_pitch, out_pitch, params):
    d = new(C.OP_VEC_ELTWISE)
    d[C.EW_OP], d[C.EW_TYPES], d[C.EW_OUT_TYPE], d[C.EW_B_BCAST] = op, a_type & 0xF | (b_type & 0xF) << 4, out_type, b_bcast
    _put(d, "HHIII", C.EW_ROWS, rows, cols, a, b, out)
    _put(d, "HHH", C.EW_A_PITCH, a_pitch, b_pitch, out_pitch)
    _put(d, "I", C.EW_PARAMS_ADDR, params)
    return d


def softmax(in_i32, rows, length, in_pitch, in_addr, out, out_pitch, in_mult, in_shift, in_zp, out_mult, out_shift,
            out_zp, mask=0, has_mask=False, causal=False, q_pos=0):
    d = new(C.OP_VEC_SOFTMAX, (C.SM_F_HAS_MASK if has_mask else 0) | (C.SM_F_CAUSAL if causal else 0))
    d[C.SM_IN_TYPE], d[C.SM_OUT_TYPE] = (C.DT_I32 if in_i32 else C.DT_I8), C.DT_I8
    _put(d, "HHHII", C.SM_ROWS, rows, length, in_pitch, in_addr, out)
    _put(d, "HBB", C.SM_OUT_PITCH, out_pitch, in_shift, out_shift)
    _put(d, "Iiii", C.SM_MASK_ADDR, mask, in_mult, in_zp, out_mult)
    _put(d, "b", C.SM_OUT_ZP, out_zp)
    _put(d, "H", C.SM_Q_POS, q_pos)
    return d


def layernorm(in_i32, rows, length, in_pitch, in_addr, out, out_pitch, inv_mult, inv_shift, eps, c_mult, c_shift, out_zp):
    d = new(C.OP_VEC_LAYERNORM)
    d[C.LN_IN_TYPE], d[C.LN_OUT_TYPE] = (C.DT_I32 if in_i32 else C.DT_I8), C.DT_I8
    _put(d, "HHHII", C.LN_ROWS, rows, length, in_pitch, in_addr, out)
    _put(d, "HBB", C.LN_OUT_PITCH, out_pitch, inv_shift, c_shift)
    _put(d, "Ii", C.LN_EPS, eps, inv_mult)
    _put(d, "ib", C.LN_C_MULT, c_mult, out_zp)
    return d


def vec_lut(rows, cols, in_addr, out, in_pitch, out_pitch, table):
    d = new(C.OP_VEC_LUT)
    _put(d, "HHI", C.LU_ROWS, rows, cols, in_addr)
    _put(d, "I", C.LU_OUT_ADDR, out)
    _put(d, "H", C.LU_IN_PITCH, in_pitch)
    _put(d, "H", C.LU_OUT_PITCH, out_pitch)
    _put(d, "I", C.LU_TABLE_ADDR, table)
    return d


# ---- control (§5.8) ----
def nop():
    return new(C.OP_NOP)


def halt():
    return new(C.OP_HALT)


def fence():
    return new(C.OP_FENCE)


def break_(break_id):
    d = new(C.OP_BREAK)
    _put(d, "H", C.BREAK_ID, break_id)
    return d


def next_chunk(offset, length):
    d = new(C.OP_NEXT_CHUNK)
    _put(d, "IH", C.NC_OFFSET, offset, length)
    return d


def loop(count, body_len, strides=None):
    """strides: {region: bytes added per iteration}."""
    d = new(C.OP_LOOP)
    _put(d, "IH", C.LOOP_COUNT, count, body_len)
    for r, s in (strides or {}).items():
        _put(d, "i", 0x20 + 4 * r, s)
    return d


# ---- decoder ----
_FIELDS = {
    C.OP_LOAD: [("vmem", "I", C.LD_VMEM_ADDR), ("rows", "H", C.LD_ROWS), ("row_bytes", "H", C.LD_ROW_BYTES),
                ("ddr_stride", "I", C.LD_DDR_STRIDE), ("vmem_stride", "I", C.LD_VMEM_STRIDE)],
    C.OP_LOAD_WGT: [("half", "B", C.LW_HALF), ("k", "H", C.LW_K)],
    C.OP_GEMM: [("act", "I", C.GM_ACT_ADDR), ("act_pitch", "H", C.GM_ACT_PITCH), ("m_tiles", "H", C.GM_M_TILES),
                ("k", "H", C.GM_K), ("half", "B", C.GM_WGT_HALF)],
    C.OP_LOAD_WGT_VM: [("src", "I", C.WV_SRC_ADDR), ("src_pitch", "H", C.WV_SRC_PITCH), ("k", "H", C.WV_K),
                       ("half", "B", C.WV_WGT_HALF)],
    C.OP_VEC_EPILOGUE: [("mode", "B", C.EP_MODE), ("out_zp", "b", C.EP_OUT_ZP), ("clamp_min", "b", C.EP_CLAMP_MIN),
                        ("clamp_max", "b", C.EP_CLAMP_MAX), ("rows", "H", C.EP_ROWS), ("dst_pitch", "H", C.EP_DST_PITCH),
                        ("dst", "I", C.EP_DST_ADDR), ("partial", "I", C.EP_PARTIAL_ADDR),
                        ("partial_pitch", "H", C.EP_PARTIAL_PITCH), ("params", "I", C.EP_PARAMS_ADDR)],
    C.OP_VEC_ELTWISE: [("op", "B", C.EW_OP), ("types", "B", C.EW_TYPES), ("out_type", "B", C.EW_OUT_TYPE),
                       ("b_bcast", "B", C.EW_B_BCAST), ("rows", "H", C.EW_ROWS), ("cols", "H", C.EW_COLS),
                       ("a", "I", C.EW_A_ADDR), ("b", "I", C.EW_B_ADDR), ("out", "I", C.EW_OUT_ADDR),
                       ("a_pitch", "H", C.EW_A_PITCH), ("b_pitch", "H", C.EW_B_PITCH),
                       ("out_pitch", "H", C.EW_OUT_PITCH), ("params", "I", C.EW_PARAMS_ADDR)],
    C.OP_VEC_SOFTMAX: [("in_type", "B", C.SM_IN_TYPE), ("out_type", "B", C.SM_OUT_TYPE), ("rows", "H", C.SM_ROWS),
                       ("len", "H", C.SM_LEN), ("in_pitch", "H", C.SM_IN_PITCH), ("in_addr", "I", C.SM_IN_ADDR),
                       ("out", "I", C.SM_OUT_ADDR), ("out_pitch", "H", C.SM_OUT_PITCH), ("in_shift", "B", C.SM_IN_SHIFT),
                       ("out_shift", "B", C.SM_OUT_SHIFT), ("mask", "I", C.SM_MASK_ADDR), ("in_mult", "i", C.SM_IN_MULT),
                       ("in_zp", "i", C.SM_IN_ZP), ("out_mult", "i", C.SM_OUT_MULT), ("out_zp", "b", C.SM_OUT_ZP),
                       ("q_pos", "H", C.SM_Q_POS)],
    C.OP_VEC_LAYERNORM: [("in_type", "B", C.LN_IN_TYPE), ("out_type", "B", C.LN_OUT_TYPE), ("rows", "H", C.LN_ROWS),
                         ("len", "H", C.LN_LEN), ("in_pitch", "H", C.LN_IN_PITCH), ("in_addr", "I", C.LN_IN_ADDR),
                         ("out", "I", C.LN_OUT_ADDR), ("out_pitch", "H", C.LN_OUT_PITCH), ("inv_shift", "B", C.LN_INV_SHIFT),
                         ("c_shift", "B", C.LN_C_SHIFT), ("eps", "I", C.LN_EPS), ("inv_mult", "i", C.LN_INV_MULT),
                         ("c_mult", "i", C.LN_C_MULT), ("out_zp", "b", C.LN_OUT_ZP)],
    C.OP_VEC_LUT: [("rows", "H", C.LU_ROWS), ("cols", "H", C.LU_COLS), ("in_addr", "I", C.LU_IN_ADDR), ("out", "I", C.LU_OUT_ADDR),
                   ("in_pitch", "H", C.LU_IN_PITCH), ("out_pitch", "H", C.LU_OUT_PITCH), ("table", "I", C.LU_TABLE_ADDR)],
    C.OP_LOOP: [("count", "I", C.LOOP_COUNT), ("body_len", "H", C.LOOP_BODY_LEN)],
    C.OP_NEXT_CHUNK: [("offset", "I", C.NC_OFFSET), ("len", "H", C.NC_LEN)],
    C.OP_BREAK: [("break_id", "H", C.BREAK_ID)],
}
_FIELDS[C.OP_STORE] = _FIELDS[C.OP_LOAD]


def decode(d):
    """Descriptor bytes -> dict of the common fields and the opcode's payload."""
    op = d[C.D_OPCODE]
    f = {"opcode": op, "flags": d[C.D_FLAGS], "wait_mask": d[C.D_WAIT_MASK],
         "wait_dist": struct.unpack_from("<4H", d, 0x04), "region": d[C.D_REGION],
         "ddr_offset": struct.unpack_from("<I", d, C.D_DDR_OFFSET)[0]}
    for name, fmt, off in _FIELDS.get(op, ()):
        f[name] = struct.unpack_from("<" + fmt, d, off)[0]
    if op == C.OP_LOOP:
        f["strides"] = struct.unpack_from("<8i", d, 0x20)
    return f


def reencode(f):
    """decode() in reverse, through the encoders above. Used to test them."""
    op = f["opcode"]
    if op in (C.OP_LOAD, C.OP_STORE):
        d = load(f["region"], f["ddr_offset"], f["vmem"], f["rows"], f["row_bytes"], f["ddr_stride"], f["vmem_stride"],
                 store=op == C.OP_STORE)
    elif op == C.OP_LOAD_WGT:
        d = load_wgt(f["region"], f["ddr_offset"], f["half"], f["k"])
    elif op == C.OP_GEMM:
        d = gemm(f["act"], f["act_pitch"], f["m_tiles"], f["k"], f["half"])
    elif op == C.OP_LOAD_WGT_VM:
        d = load_wgt_vm(f["src"], f["src_pitch"], f["k"], f["half"], bool(f["flags"] & C.WV_F_TRANSPOSE))
    elif op == C.OP_VEC_EPILOGUE:
        d = epilogue(f["mode"], f["flags"], f["out_zp"], f["clamp_min"], f["clamp_max"], f["rows"], f["dst_pitch"],
                     f["dst"], f["params"], f["partial"], f["partial_pitch"])
    elif op == C.OP_VEC_ELTWISE:
        d = eltwise(f["op"], f["types"] & 0xF, f["types"] >> 4, f["out_type"], f["b_bcast"], f["rows"], f["cols"], f["a"],
                    f["b"], f["out"], f["a_pitch"], f["b_pitch"], f["out_pitch"], f["params"])
    elif op == C.OP_VEC_SOFTMAX:
        d = softmax(f["in_type"] == C.DT_I32, f["rows"], f["len"], f["in_pitch"], f["in_addr"], f["out"], f["out_pitch"],
                    f["in_mult"], f["in_shift"], f["in_zp"], f["out_mult"], f["out_shift"], f["out_zp"], f["mask"],
                    bool(f["flags"] & C.SM_F_HAS_MASK), bool(f["flags"] & C.SM_F_CAUSAL), f["q_pos"])
    elif op == C.OP_VEC_LAYERNORM:
        d = layernorm(f["in_type"] == C.DT_I32, f["rows"], f["len"], f["in_pitch"], f["in_addr"], f["out"], f["out_pitch"],
                      f["inv_mult"], f["inv_shift"], f["eps"], f["c_mult"], f["c_shift"], f["out_zp"])
    elif op == C.OP_VEC_LUT:
        d = vec_lut(f["rows"], f["cols"], f["in_addr"], f["out"], f["in_pitch"], f["out_pitch"], f["table"])
    elif op == C.OP_LOOP:
        d = loop(f["count"], f["body_len"], dict(enumerate(f["strides"])))
    elif op == C.OP_NEXT_CHUNK:
        d = next_chunk(f["offset"], f["len"])
    elif op == C.OP_BREAK:
        d = break_(f["break_id"])
    else:
        d = new(op)
    for u in range(4):
        if f["wait_mask"] >> u & 1:
            wait(d, u, f["wait_dist"][u])
    return d


# ---- program container (§3) ----
def header(arch, regions_mask, first_off, first_len, total_bytes, program_id=0, compiler_hash=0):
    h = bytearray(C.PROG_HDR_SIZE)
    _put(h, "IHH", C.HDR_MAGIC, C.PROG_MAGIC, C.PROG_FORMAT, C.PROG_HDR_SIZE)
    _put(h, "I", C.HDR_ARCH, arch)
    h[C.HDR_REGIONS] = regions_mask
    _put(h, "IH", C.HDR_FIRST_OFF, first_off, first_len)
    _put(h, "II", C.HDR_TOTAL_BYTES, total_bytes, program_id)
    _put(h, "Q", C.HDR_COMPILER_HASH, compiler_hash)
    return h


def build_program(descs, regions_mask=0x0F, arch=None, program_id=0, compiler_hash=0):
    """Header + chunks. Linear descriptor list in; splits into chunks with NEXT_CHUNK.

    A LOOP body must stay inside one chunk (§5.8), so a LOOP moves to the next chunk whole.
    """
    if not descs or descs[-1][C.D_OPCODE] != C.OP_HALT:
        raise ValueError("a program must end with HALT")
    limit = C.CHUNK_MAX_DESCS - 1                           # room for NEXT_CHUNK
    chunks, cur, i = [], [], 0
    while i < len(descs):
        n = 1
        if descs[i][C.D_OPCODE] == C.OP_LOOP:
            n += decode(descs[i])["body_len"]
        if n > limit:
            raise ValueError(f"LOOP at descriptor {i} does not fit in a chunk")
        if len(cur) + n > limit:
            chunks.append(cur)
            cur = []
        cur += descs[i:i + n]
        i += n
    chunks.append(cur)

    off, offs = C.PROG_HDR_SIZE, []
    for j, c in enumerate(chunks):
        offs.append(off)
        off += DESC * (len(c) + (j + 1 < len(chunks)))
    body = bytearray()
    for j, c in enumerate(chunks):
        for d in c:
            body += d
        if j + 1 < len(chunks):
            body += next_chunk(offs[j + 1], len(chunks[j + 1]) + (j + 2 < len(chunks)))
    first_len = len(chunks[0]) + (len(chunks) > 1)
    h = header(arch_word() if arch is None else arch, regions_mask, offs[0], first_len, off, program_id, compiler_hash)
    return bytes(h + body)
