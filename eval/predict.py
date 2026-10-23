"""Write predictions for the test set.

    python -m eval.predict --method rules --out eval/preds/rules.jsonl
    python -m eval.predict --method hf --model Qwen/Qwen2.5-1.5B-Instruct --out eval/preds/zeroshot.jsonl
    python -m eval.predict --method hf --model Qwen/Qwen2.5-1.5B-Instruct --shots 1 --out eval/preds/oneshot.jsonl
    python -m eval.predict --method hf --model Qwen/Qwen2.5-1.5B-Instruct --adapter outputs/lora --out eval/preds/finetuned.jsonl
"""
import argparse
import json
from pathlib import Path

from recipe_parser import rules
from recipe_parser.prompt import messages


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def hf_predict(examples, model_name, adapter, shots, batch_size):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    demo = read_jsonl("data/train.jsonl")[:shots]

    outputs = []
    for b in range(0, len(examples), batch_size):
        batch = examples[b:b + batch_size]
        prompts = [tok.apply_chat_template(messages(ex["page"], shots=demo), tokenize=False, add_generation_prompt=True)
                   for ex in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=1024, do_sample=False)
        outputs += tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"{len(outputs)}/{len(examples)}")
    return outputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["rules", "hf"], required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter")
    p.add_argument("--shots", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--split", default="test")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    examples = read_jsonl(f"data/{args.split}.jsonl")
    if args.method == "rules":
        outputs = [json.dumps(rules.extract(ex["page"])) for ex in examples]
    else:
        outputs = hf_predict(examples, args.model, args.adapter, args.shots, args.batch_size)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for ex, out in zip(examples, outputs):
            f.write(json.dumps({"id": ex["id"], "output": out}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
