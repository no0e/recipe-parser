#!/usr/bin/env bash
# Full pipeline on a single GPU (tested on a 16 GB T4): data, baselines, training, evaluation. Run from the repo root.
set -e
pip install -q -r requirements-gpu.txt
python -m data.build_dataset
python -m eval.predict --method rules --out eval/preds/rules.jsonl
python -m eval.predict --method hf --out eval/preds/zeroshot.jsonl
python -m eval.predict --method hf --shots 1 --batch-size 4 --out eval/preds/oneshot.jsonl
python -m train.train_qlora --out outputs/lora
python -m eval.predict --method hf --adapter outputs/lora --out eval/preds/finetuned.jsonl
python -m eval.evaluate eval/preds/rules.jsonl eval/preds/zeroshot.jsonl eval/preds/oneshot.jsonl eval/preds/finetuned.jsonl | tee eval/results.md
