"""Type the start of a line; GPT-2 on the KV260 finishes it.

Starts build/generate (through sudo unless already root) and talks to it over its stdin and
stdout (protocol in generate.c). Text goes through gpt2_tokenizer.py and model/tokenizer.json.

usage: python3 complete.py [--model DIR] [--tokens N] [--temperature T] [--top-k K]
                           [--stop sentence|line|none] [--seed N] [prompt ...]
Without a prompt it asks for one line at a time; an empty line or Ctrl-D quits. It stops at the
first sentence end or newline (--stop), after --tokens tokens, or at <|endoftext|>.
"""
import argparse
import codecs
import os
import random
import shlex
import subprocess
import sys
from pathlib import Path

from gpt2_tokenizer import Tokenizer

HERE = Path(__file__).resolve().parent


def stop_ids(tok, mode):
    """Tokens that end a completion: the special ones, and by mode those holding a newline or ending a sentence."""
    ids = set(tok.special.values())
    if mode == "none":
        return ids
    for i in range(len(tok)):
        t = tok.token_bytes(i).decode("utf-8", errors="ignore")
        if "\n" in t or (mode == "sentence" and t.rstrip().endswith((".", "!", "?"))):
            ids.add(i)
    return ids


def prompts(given):
    if given:
        yield " ".join(given)
        return
    while True:
        try:
            line = input("> ")
        except EOFError:
            return
        if not line:
            return
        yield line


def main():
    ap = argparse.ArgumentParser(description="GPT-2 on the KV260 finishes your line.")
    ap.add_argument("prompt", nargs="*")
    ap.add_argument("--model", default=str(HERE / "model"), help="compiled model folder (default: model)")
    ap.add_argument("--tokens", type=int, default=50, help="most new tokens (default 50)")
    ap.add_argument("--temperature", type=float, default=0.8, help="0 always takes the most likely token (default 0.8)")
    ap.add_argument("--top-k", type=int, default=40, help="sample from the K most likely tokens; 0 for all (default 40)")
    ap.add_argument("--stop", choices=("sentence", "line", "none"), default="sentence", help="where to stop (default sentence)")
    ap.add_argument("--seed", type=int, help="random seed (default: a new one each run)")
    ap.add_argument("--generate", help=argparse.SUPPRESS)       # replaces build/generate, for tests
    a = ap.parse_args()

    tok = Tokenizer(Path(a.model) / "tokenizer.json")
    stops = sorted(stop_ids(tok, a.stop))
    if a.generate:
        cmd = shlex.split(a.generate)
    else:
        cmd = [str(HERE / "build" / "generate"), "-d", a.model]
        if os.geteuid() != 0:
            cmd = ["sudo"] + cmd
    gen = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    ready = gen.stdout.readline().split()
    if len(ready) != 4 or ready[0] != "R":
        sys.exit("generate did not start (see above); is the bitstream loaded? (make load)")
    ctx = int(ready[3])
    seed = random.randrange(1 << 31) if a.seed is None else a.seed
    bold, plain = ("\033[1m", "\033[0m") if sys.stdout.isatty() else ("", "")

    for text in prompts(a.prompt):
        ids = tok.encode(text)[-(ctx - 1):]
        gen.stdin.write(f"{a.tokens} {a.temperature} {a.top_k} {seed} {len(stops)} {' '.join(map(str, stops))} "
                        f"{len(ids)} {' '.join(map(str, ids))}\n")
        gen.stdin.flush()
        seed += 1
        utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")
        print(text + bold, end="", flush=True)
        while True:
            line = gen.stdout.readline()
            if not line:
                sys.exit(plain + "\ngenerate stopped")
            kind, *rest = line.split()
            if kind == "E":
                reason, n, prefill_ms, ms = rest
                rate = f", {1000 / float(ms):.1f} tokens/s" if float(ms) > 0 else ""
                print(f"{plain}\n[{n} tokens, {reason}; prompt {prefill_ms} ms, {ms} ms per token{rate}]", flush=True)
                break
            i = int(rest[0])
            if i in tok.special.values():
                continue
            s = utf8.decode(tok.token_bytes(i))
            if a.stop != "none":
                s = s.split("\n")[0]
            print(s, end="", flush=True)
    gen.stdin.close()
    gen.wait()


if __name__ == "__main__":
    main()
