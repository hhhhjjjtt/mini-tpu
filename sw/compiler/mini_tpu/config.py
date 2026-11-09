"""config.toml: the one description of the hardware, the model and the build.

tools/gen_config.py turns [hardware] into hw/interface/mini_tpu_config.h and
hw/rtl/common/mini_tpu_config.svh; the compiler reads the same values through here.
"""
import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
UNITS_MASK = 0x1F           # MINI_TPU_UNITS_MASK: four units, bit 4 = three weight halves


def _pow2(v):
    return v > 0 and v & (v - 1) == 0


@dataclass(frozen=True)
class Hardware:
    name: str
    array_n: int
    vector_lanes: int
    vmem_kb: int
    max_k: int
    max_len: int
    fifo_depth: int

    def __post_init__(self):
        checks = [
            (self.array_n in (8, 16, 32, 64), "array_n: 8, 16, 32 or 64"),
            (self.vector_lanes in (4, 8, 16, 32, 64), "vector_lanes: 4, 8, 16, 32 or 64"),
            (_pow2(self.vmem_kb) and 256 <= self.vmem_kb <= 2048, "vmem_kb: power of two, 256 ... 2048"),
            (_pow2(self.max_k) and 16 <= self.max_k <= 1024, "max_k: power of two, 16 ... 1024"),
            (self.max_len % max(16, self.vector_lanes) == 0 and 16 <= self.max_len <= 1024,
             "max_len: multiple of 16 and of vector_lanes, 16 ... 1024"),
            (2 <= self.fifo_depth <= 64, "fifo_depth: 2 ... 64"),
        ]
        bad = [msg for ok, msg in checks if not ok]
        if bad:
            raise ValueError("config.toml [hardware]: " + "; ".join(bad))

    @property
    def tile_n(self):
        return max(16, self.array_n)                        # GEMM tile: what programs see

    @property
    def vmem_bytes(self):
        return self.vmem_kb << 10

    @property
    def vm_aw(self):
        return (self.vmem_bytes >> 4).bit_length() - 1      # word address bits

    @property
    def vmem_banks(self):
        return self.vmem_kb // 256

    @property
    def free_w(self):
        return self.fifo_depth.bit_length()                 # holds 0 ... depth

    @property
    def wgt_aw(self):
        return (self.max_k * self.tile_n // 16).bit_length() - 1   # word within a weight half

    @property
    def arch_word(self):
        return self.tile_n | UNITS_MASK << 8 | self.vmem_kb << 16

    @property
    def hash(self):
        """Identifies a hardware build. Goes in the program header's compiler_hash."""
        text = json.dumps(asdict(self), sort_keys=True)
        return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


@dataclass(frozen=True)
class Config:
    hardware: Hardware
    model_name: str
    model_path: Path
    calib_path: Path
    seq: int
    calib_tokens: int

    @property
    def calibrated(self):
        """Stage 1 output. Does not depend on the hardware."""
        return ROOT / "sw" / "compiler" / "output" / f"calibrated_{self.model_name}_s{self.seq}.npz"

    @property
    def output(self):
        return ROOT / "sw" / "compiler" / "output" / f"{self.hardware.name}_{self.model_name}_s{self.seq}"


def load(path=None):
    d = tomllib.loads(Path(path or ROOT / "config.toml").read_text(encoding="utf-8"))
    m, b = d["model"], d["build"]
    return Config(Hardware(**d["hardware"]), m["name"], ROOT / m["path"], ROOT / m["calib"], b["seq"], b["calib_tokens"])
