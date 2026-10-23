"""QLoRA fine-tuning of a small instruct model on page -> JSON extraction (fits a 16 GB T4).

    python -m train.train_qlora --model Qwen/Qwen2.5-1.5B-Instruct --out outputs/lora
"""
import argparse
import json

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)

from recipe_parser.prompt import messages


def tokenize(ex, tok, max_len):
    """Tokenize prompt + answer; loss is computed on the answer tokens only."""
    prompt = tok.apply_chat_template(messages(ex["page"]), tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(messages(ex["page"], ex["target"]), tokenize=False)
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    ids = tok(full, add_special_tokens=False).input_ids[:max_len]
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(p_ids) + ids[len(p_ids):]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--out", default="outputs/lora")
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--max-len", type=int, default=2048)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules="all-linear"))
    model.print_trainable_parameters()

    load = lambda split: [tokenize(json.loads(l), tok, args.max_len) for l in open(f"data/{split}.jsonl", encoding="utf-8")]
    train, val = load("train"), load("val")
    print(f"max train length: {max(len(x['input_ids']) for x in train)} tokens")

    trainer = Trainer(
        model=model,
        train_dataset=train,
        eval_dataset=val,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True),
        args=TrainingArguments(
            output_dir=args.out, num_train_epochs=args.epochs, learning_rate=2e-4,
            per_device_train_batch_size=2, gradient_accumulation_steps=8, per_device_eval_batch_size=2,
            lr_scheduler_type="cosine", warmup_steps=10, fp16=True,
            logging_steps=10, eval_strategy="steps", eval_steps=50, save_strategy="no", report_to="none",
        ),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)


if __name__ == "__main__":
    main()
