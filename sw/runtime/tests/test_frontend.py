"""The board's text frontend on the workstation: gpt2_tokenizer.py must encode like Hugging Face
`tokenizers`, and complete.py must speak generate.c's protocol (against a stand-in generate).

usage: python sw/runtime/tests/test_frontend.py   (needs `tokenizers`; make test runs it)
"""
import re
import subprocess
import sys
import unittest
from pathlib import Path

RUNTIME = Path(__file__).resolve().parents[1]
ROOT = RUNTIME.parents[1]
TOKENIZER = ROOT / "sw" / "compiler" / "model" / "gpt2" / "tokenizer.json"
sys.path.insert(0, str(RUNTIME))

import gpt2_tokenizer

SAMPLES = ["Hello world!", "  several   spaces\n\n\tand tabs  ", "it's I'll we've they'd", "3.14159 and 1,000,000 or 2nd",
           "naïve café — “quotes” déjà vu 😀 日本語のテキスト", "snake_case __init__ a+b=c", "<|endoftext|>After the end",
           "Ünïcödé non-breaking", ""]

# stands in for build/generate: answers one request with the tokens of " jumps over the lazy dog." and the stop reason
FAKE = """
import sys
sys.path.insert(0, {runtime!r})
from gpt2_tokenizer import Tokenizer
tok = Tokenizer({tokenizer!r})
print("R 50257 128 1024", flush=True)
for line in sys.stdin:
    v = line.split()
    n_stop = int(v[4])
    stops = set(map(int, v[5:5 + n_stop]))
    ids = list(map(int, v[6 + n_stop:]))
    assert len(ids) == int(v[5 + n_stop]) and tok.decode(ids) == "The quick brown fox", line
    out = tok.encode(" jumps over the lazy dog. And more")
    n = next(i for i, t in enumerate(out) if t in stops) + 1
    for t in out[:n]:
        print("T", t, flush=True)
    print("E stop", n, "1012.5 98.0", flush=True)
"""


class Tokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer as HF
        cls.hf = HF.from_file(str(TOKENIZER))
        cls.tok = gpt2_tokenizer.Tokenizer(TOKENIZER)
        cls.text = (ROOT / "sw" / "compiler" / "model" / "calib.txt").read_text(encoding="utf-8")[:400000]

    def compare(self):
        for s in SAMPLES + [self.text]:
            self.assertEqual(self.tok.encode(s), self.hf.encode(s).ids, s[:60])
            self.assertEqual(self.tok.decode(self.tok.encode(s)), s)

    def test_matches_hugging_face(self):
        self.compare()
        print(f"{len(self.tok.encode(self.text))} tokens of WikiText match, regex package: {gpt2_tokenizer._WORDS.__class__.__module__}")

    def test_without_regex_package(self):
        """The board may not have `regex`: the plain `re` pattern must give the same tokens here."""
        saved = gpt2_tokenizer._WORDS
        gpt2_tokenizer._WORDS = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+""")
        try:
            self.tok._cache.clear()
            self.compare()
        finally:
            gpt2_tokenizer._WORDS = saved

    def test_special_token(self):
        self.assertEqual(self.tok.encode("<|endoftext|>"), [50256])
        self.assertEqual(self.tok.token_bytes(50256), b"<|endoftext|>")


class Complete(unittest.TestCase):
    def test_protocol(self):
        fake = RUNTIME / "build" / "fake_generate.py"
        fake.parent.mkdir(exist_ok=True)
        fake.write_text(FAKE.format(runtime=str(RUNTIME), tokenizer=str(TOKENIZER)), encoding="utf-8")
        model = TOKENIZER.parent
        r = subprocess.run([sys.executable, str(RUNTIME / "complete.py"), "--model", str(model), "--generate",
                            f'"{sys.executable}" "{fake}"', "The", "quick", "brown", "fox"], capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("The quick brown fox jumps over the lazy dog.\n", r.stdout)
        self.assertIn("tokens, stop; prompt 1012.5 ms, 98.0 ms per token, 10.2 tokens/s]", r.stdout)
        print(r.stdout.strip())


if __name__ == "__main__":
    unittest.main()
