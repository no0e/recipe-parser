# recipe-parser

Paste a messy recipe blog page (life story, ads, comments…) and get back a clean, structured recipe:
ingredients as `quantity / unit / name`, numbered steps with durations. The recipe can then be rescaled to any
number of servings and turned into a merged shopping list.

The extraction is done by a **small LLM (Qwen2.5-1.5B-Instruct) fine-tuned with QLoRA**, compared against
honest baselines, and served on CPU as an API on **Google Cloud Run**.

## Results

Test set: 200 recipes rendered with page layouts (section headers, bullets, ads, story sentences) **never seen
during training or while writing the rules**.

| method | valid JSON | ingredient F1 | name F1 | servings acc. | step count acc. | duration acc. |
|---|---|---|---|---|---|---|
| rule-based (regex + headers) | 1.000 | 0.613 | 0.613 | 0.550 | 0.355 | 0.518 |
| Qwen2.5-1.5B zero-shot | 0.760 | 0.107 | 0.468 | 0.745 | 0.550 | 0.405 |
| Qwen2.5-1.5B one-shot | 0.810 | 0.179 | 0.542 | 0.810 | 0.740 | 0.315 |
| **Qwen2.5-1.5B + QLoRA** | **0.980** | **0.929** | **0.935** | **0.970** | **0.980** | **0.940** |

- *ingredient F1*: exact match on the `(quantity, unit, name)` triple; *name F1*: name only.
- The rules are near-perfect (0.999 F1) on pages using the training layouts but break on new ones; prompting alone
  gets the gist but not the exact format (canonical units, numeric quantities). Fine-tuning gets both.
- Training: 2,000 examples, 1 epoch, LoRA r=16 on all linear layers, 4-bit NF4 base, ~70 min on a single T4. A second training run gave 0.938 ingredient F1 (vs 0.929 above): results are stable across runs.

## How it works

```
noisy page ──► fine-tuned LLM ──► Recipe JSON ──► scale(servings) ──► shopping_list()
                (extraction only)                  (plain Python: unit conversion, merging)
```

Only the extraction uses the model. Scaling and shopping-list merging are deterministic Python
(`recipe_parser/kitchen.py`): no reason to let a model do arithmetic.

**Data.** No labelled "blog page → JSON" dataset exists, so it is generated (`data/build_dataset.py`):
recipes from the [Epicurious dataset](https://huggingface.co/datasets/Hieu-Pham/kaggle_food_recipes) (CC BY-SA 3.0)
are parsed into the gold JSON, then rendered as fake blog pages with random quantity styles (`½`, `0.5`, `one`),
unit spellings, stories, ads and comments. The test split uses held-out templates.

**Limitation.** Gold ingredient fields come from a deterministic parse of the source lines, and the pages are
synthetic. The scores measure robustness to noise and unseen layouts, not accuracy on real scraped pages.

## Repository

```
recipe_parser/   schema, units & ingredient parsing, rule baseline, prompt, scaling/shopping list
data/            dataset generation
train/           QLoRA training, GGUF export
eval/            predictions + metrics
app/             FastAPI API + Gradio demo (/demo)
```

## Reproduce

```bash
pip install -r requirements-gpu.txt
bash scripts_onyxia.sh          # dataset, baselines, training, evaluation (GPU)
bash train/export_gguf.sh       # merge LoRA + convert to GGUF (8-bit) for CPU serving
```

## API

```bash
MODEL_PATH=outputs/recipe-parser-q8_0.gguf uvicorn app.main:app --port 8080
```

| endpoint | input | output |
|---|---|---|
| `POST /parse` | `{"text": "<page>"}` | recipe JSON |
| `POST /scale` | `{"recipe": {...}, "servings": 6}` | rescaled recipe |
| `POST /shopping-list` | `[recipe, recipe, ...]` | merged ingredients |
| `GET /demo` | | Gradio UI |

Speed on CPU (llama-bench, q8_0, 14 threads): 126 tok/s prompt processing, 23 tok/s generation, i.e. roughly
35 s for a typical recipe. llama-cpp-python is compiled with AVX2 in the image: the generic prebuilt wheel was ~6x slower.

Deploy to Cloud Run (the image downloads the GGUF model from the Hugging Face Hub at build time):

```bash
gcloud run deploy recipe-parser --source . --region europe-west1 --memory 4Gi --cpu 8 --concurrency 1 --timeout 300 --allow-unauthenticated
```
