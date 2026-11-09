"""Downloads GPT-2 and a calibration text into this folder. Needs huggingface_hub and datasets."""
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
CALIB_CHARS = 400_000

for name in ("config.json", "model.safetensors", "tokenizer.json"):
    hf_hub_download("gpt2", name, local_dir=HERE / "gpt2")
text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["train"]["text"])
(HERE / "calib.txt").write_text(text[:CALIB_CHARS], encoding="utf-8")
print(f"wrote {HERE / 'gpt2'} and calib.txt")
