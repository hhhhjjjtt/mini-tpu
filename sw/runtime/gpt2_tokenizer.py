"""GPT-2's byte-level BPE from a Hugging Face tokenizer.json, in plain Python for the KV260.

Uses the `regex` package for GPT-2's exact pre-tokenizer pattern when it is installed. Without
it, an equivalent pattern on `re` (letters [^\\W\\d_], numbers \\d) differs only for rare Unicode
characters such as superscript digits or combining marks.
"""
import json
import re

try:
    import regex
    _WORDS = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
except ImportError:
    _WORDS = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+""")


def _byte_chars():
    """GPT-2's map from each byte to a printable character."""
    keep = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    extra = [b for b in range(256) if b not in keep]
    return {b: chr(c) for b, c in zip(keep + extra, keep + list(range(256, 256 + len(extra))))}


class Tokenizer:
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            t = json.load(f)
        m = t["model"]
        self.encoder = m["vocab"]
        self.decoder = {i: s for s, i in self.encoder.items()}
        merges = [tuple(x.split(" ", 1)) if isinstance(x, str) else tuple(x) for x in m["merges"]]
        self.ranks = {pair: i for i, pair in enumerate(merges)}
        self.byte_char = _byte_chars()
        self.char_byte = {c: b for b, c in self.byte_char.items()}
        self.special = {a["content"]: a["id"] for a in t.get("added_tokens", [])}
        self._split = re.compile("(" + "|".join(map(re.escape, self.special)) + ")") if self.special else None
        self._cache = {}

    def __len__(self):
        return len(self.decoder) + sum(i not in self.decoder for i in self.special.values())

    def _bpe(self, word):
        if word in self._cache:
            return self._cache[word]
        parts = list(word)
        while len(parts) > 1:
            pairs = {(parts[i], parts[i + 1]) for i in range(len(parts) - 1)}
            best = min(pairs, key=lambda p: self.ranks.get(p, len(self.ranks)))
            if best not in self.ranks:
                break
            out, i = [], 0
            while i < len(parts):
                if i + 1 < len(parts) and (parts[i], parts[i + 1]) == best:
                    out.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    out.append(parts[i])
                    i += 1
            parts = out
        self._cache[word] = parts
        return parts

    def encode(self, text):
        ids = []
        for part in self._split.split(text) if self._split else [text]:
            if part in self.special:
                ids.append(self.special[part])
                continue
            for w in _WORDS.findall(part):
                ids.extend(self.encoder[p] for p in self._bpe("".join(self.byte_char[b] for b in w.encode("utf-8"))))
        return ids

    def token_bytes(self, i):
        if i in self.decoder and i not in self.special.values():
            return bytes(self.char_byte[c] for c in self.decoder[i])
        return next(s for s, j in self.special.items() if j == i).encode("utf-8")

    def decode(self, ids):
        return b"".join(self.token_bytes(i) for i in ids).decode("utf-8", errors="replace")
