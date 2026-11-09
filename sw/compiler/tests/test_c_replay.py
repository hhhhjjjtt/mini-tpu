"""The KV260 runtime with the Python simulator in place of the board.

mtpu_test.c builds its programs with the C encoders and checks every result against its own
reference math, the same checks that passed on the board. run_prefill.c and generate.c load the
compiled model the way they do on the board. All build unchanged against mtpu_stub.c. Needs a host gcc.
"""
import os
import re
import shutil
import struct
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent / "c_replay"
BUILD = HERE / "build"
ROOT = HERE.parents[3]
RUNTIME = ROOT / "sw" / "runtime"
SEEDS = (1, 2, 3)
VALUES = 42552              # checked per run on the KV260 (sw/runtime/README.md)
ENV = dict(os.environ, REPLAY_CMD=f'"{sys.executable}" ../replay.py')
MODEL_FILES = ("program.bin", "weights.bin", "params.bin", "model.txt", "golden_input.bin", "golden_output.bin")


def build(name, source):
    BUILD.mkdir(exist_ok=True)
    exe = BUILD / f"{name}.exe"
    subprocess.run(["gcc", "-O1", "-w", "-Dusleep=mtpu_stub_usleep", f"-I{RUNTIME / 'libmtpu'}", f"-I{ROOT / 'hw' / 'interface'}",
                    str(source), str(RUNTIME / "libmtpu" / "mtpu_desc.c"), str(HERE / "mtpu_stub.c"),
                    "-lm", "-o", str(exe)], check=True)
    return exe


class CReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exe = build("mtpu_test_sim", RUNTIME / "tests" / "mtpu_test.c")

    def test_all_seeds(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                summary = BUILD / "summary.txt"
                summary.unlink(missing_ok=True)
                r = subprocess.run([str(self.exe), "-f", "x", "-s", str(seed)], cwd=BUILD, env=ENV, capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-1000:])
                m = re.search(r"(\d+) passed, (\d+) failed; (\d+) values checked, (\d+) mismatches", r.stdout)
                passed, failed, values, mismatches = map(int, m.groups())
                self.assertEqual((passed, failed, mismatches), (12, 0, 0), r.stdout)
                self.assertEqual(values, VALUES)
                runs = summary.read_text().splitlines()
                total = lambda key: sum(int(re.search(key + r"=(\d+)", line).group(1)) for line in runs)
                self.assertEqual(total("encode_mismatch"), 0)
                self.assertEqual(total("races"), 0, "\n".join(runs))
                self.assertEqual(total("rules"), 0, "\n".join(runs))
                print(f"seed {seed}: {passed}/12 tests, {values} values, {len(runs)} programs, {total('descs')} descriptors "
                      f"re-encoded, {total('uninit')} uninitialized reads")


class Prefill(unittest.TestCase):
    """The compiled model through run_prefill. Skipped until `make` has written output/."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(HERE.parents[1]))
        from mini_tpu import config
        cls.model = config.load().output
        if not (cls.model / MODEL_FILES[-1]).exists():
            raise unittest.SkipTest("no compiled model: run make")
        cls.exe = build("run_prefill_sim", RUNTIME / "run_prefill.c")

    def run_tool(self, directory):
        (BUILD / "summary.txt").unlink(missing_ok=True)
        return subprocess.run([str(self.exe), "-f", "x", "-d", str(directory)], cwd=BUILD, env=ENV, capture_output=True, text=True)

    def tampered(self, name, offset):
        """A copy of the model with one byte of one file flipped."""
        d = BUILD / "tampered"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir()
        for f in MODEL_FILES:
            shutil.copy(self.model / f, d / f) if f == name else os.link(self.model / f, d / f)
        with open(d / name, "r+b") as f:
            f.seek(offset)
            byte = f.read(1)[0]
            f.seek(offset)
            f.write(bytes([byte ^ 1]))
        return d

    def test_matches(self):
        r = self.run_tool(self.model)
        self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-1000:])
        self.assertIn(" 0 of ", r.stdout)
        self.assertIn("PASSED", r.stdout)
        line = (BUILD / "summary.txt").read_text()
        self.assertIn("encode_mismatch=0", line)
        self.assertIn("races=0", line)
        print(r.stdout.strip().splitlines()[-2].strip())

    def test_detects_wrong_output(self):
        r = self.run_tool(self.tampered("golden_output.bin", 4096))
        self.assertEqual(r.returncode, 1, r.stdout[-2000:] + r.stderr[-1000:])
        self.assertIn("1 of ", r.stdout)
        self.assertIn("FAILED", r.stdout)

    def test_rejects_other_config(self):
        sys.path.insert(0, str(HERE.parents[1]))
        from mini_tpu.isa import C
        r = self.run_tool(self.tampered("program.bin", C.HDR_COMPILER_HASH))
        self.assertEqual(r.returncode, 1, r.stdout[-2000:] + r.stderr[-1000:])
        self.assertIn("program built for config", r.stderr)


class Generate(unittest.TestCase):
    """generate.c: its decode check, and one request. Skipped until `make` has written output/."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(HERE.parents[1]))
        from mini_tpu import config
        cls.model = config.load().output
        if not (cls.model / "golden_decode.bin").exists():
            raise unittest.SkipTest("no compiled model: run make")
        cls.exe = build("generate_sim", RUNTIME / "generate.c")

    def run_tool(self, *args, stdin=None):
        (BUILD / "summary.txt").unlink(missing_ok=True)
        r = subprocess.run([str(self.exe), "-f", "x", "-d", str(self.model), *args], cwd=BUILD, env=ENV, input=stdin,
                           capture_output=True, text=True)
        for line in (BUILD / "summary.txt").read_text().splitlines():
            self.assertIn("encode_mismatch=0", line)
            self.assertIn("races=0", line)
        return r

    def test_decode_check(self):
        """Prefill of 8 tokens, then decode at positions 8 and 9: bit for bit against the reference."""
        r = self.run_tool("-c", "-m", "9")
        self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-1000:])
        self.assertIn("3 positions checked, 0 logits differ", r.stdout)
        print(r.stdout.strip())

    def test_request(self):
        """Greedy, two tokens: the first is the argmax of the reference logits after the prompt."""
        import numpy as np
        ids = np.fromfile(self.model / "golden_ids.bin", dtype="<i4")[:8]
        g = np.fromfile(self.model / "golden_decode.bin", dtype="<i4")
        count = int(g[1])
        rows = dict(zip(g[2:2 + count].tolist(), g[2 + count:].reshape(count, -1)))
        scale = np.fromfile(self.model / "logit_scale.bin", dtype="<f4")
        want = int(np.argmax(rows[7][:scale.size].astype(np.float64) * scale))
        r = self.run_tool(stdin=f"2 0 0 1 0 8 {' '.join(map(str, ids))}\n")
        self.assertEqual(r.returncode, 0, r.stderr[-1000:])
        lines = r.stdout.splitlines()
        self.assertTrue(lines[0].startswith("R 50257 128 1024"), lines)
        self.assertEqual(lines[1], f"T {want}")
        self.assertTrue(lines[2].startswith("T "), lines)
        self.assertTrue(lines[3].startswith("E max 2 "), lines)


if __name__ == "__main__":
    unittest.main()
