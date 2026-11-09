"""Checks the hand-written forward pass before any quantization result is trusted."""
import torch
from gpt2_quant import *
from gpt2_quant import ROOT  # noqa: F401
from transformers import GPT2LMHeadModel

sd = load_gpt2()
ident = lambda s, x, b, k="act": x
ids = torch.randint(0, 50257, (2, 128), device=DEV)
with torch.no_grad():
    hf = GPT2LMHeadModel.from_pretrained("gpt2").to(DEV).eval()(ids).logits
    plain = forward(ids, prepare(sd, Cfg("fp")), Cfg("fp"), ident)
    folded = forward(ids, prepare(sd, Cfg("fold", fold_ln=True)), Cfg("fold", fold_ln=True), ident)
print("max |mine - HF|      ", (plain - hf).abs().max().item())
print("max |folded - plain| ", (folded - plain).abs().max().item())
print("logit scale          ", hf.abs().max().item())

train, test = data()
print("tokens train/test", train.numel(), test.numel())
for S in (256, 1024):
    obs = calibrate(sd, train, S, fold=False, windows=8)
    print(S, evaluate(sd, Cfg("fp"), obs, test, S))
    # a 24-bit grid should be indistinguishable from float, proving the quant path itself
    print(S, evaluate(sd, Cfg("q24", w=24, act=24, qkv=24, resid=24, gelu_in=24), obs, test, S))
