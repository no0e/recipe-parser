#!/usr/bin/env bash
# Merge the LoRA adapter into the base model and convert it to a single GGUF file (8-bit) for CPU inference.
set -e
BASE=${BASE:-Qwen/Qwen2.5-1.5B-Instruct}
python - <<PY
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("$BASE", dtype=torch.float16)
PeftModel.from_pretrained(model, "outputs/lora").merge_and_unload().save_pretrained("outputs/merged")
AutoTokenizer.from_pretrained("$BASE").save_pretrained("outputs/merged")
PY
[ -d outputs/llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp outputs/llama.cpp
python outputs/llama.cpp/convert_hf_to_gguf.py outputs/merged --outtype q8_0 --outfile outputs/recipe-parser-q8_0.gguf
ls -lh outputs/recipe-parser-q8_0.gguf
