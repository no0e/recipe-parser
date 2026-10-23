---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags: [gguf, qlora, information-extraction, recipes]
language: [en]
---
# recipe-parser (Qwen2.5-1.5B, QLoRA, GGUF q8_0)

Turns a noisy recipe blog page (story, ads, comments) into structured JSON:
`{"title", "servings", "ingredients": [{"quantity", "unit", "name"}], "steps": [{"step", "text", "duration_minutes"}]}`.

Fine-tuned with QLoRA (r=16, 4-bit NF4) on 2,000 synthetic noisy pages built from Epicurious recipes (CC BY-SA 3.0),
then merged and converted to GGUF (8-bit) for CPU inference with llama.cpp.

| method (held-out page layouts, 200 recipes) | ingredient F1 | valid JSON |
|---|---|---|
| rule-based | 0.613 | 1.000 |
| base model, one-shot | 0.179 | 0.810 |
| **this model** | **0.929** | **0.980** |

Code, data generation, evaluation and API: https://github.com/No0e/recipe-parser

Limitation: pages are synthetic; scores measure robustness to noise and unseen layouts, not accuracy on real scraped pages.
