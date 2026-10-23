"""Build the train/val/test sets.

Source: Epicurious recipes (Kaggle, CC BY-SA 3.0) mirrored on the HF Hub.
Each clean recipe gives the gold JSON; we then render it as a noisy "blog page"
(life story, ads, newsletter prompts, comments, random formatting). The test set
uses *held-out* headers, bullets, ad texts and story sentences, so a method that
memorised the training templates is penalised.

    python -m data.build_dataset --n-train 2000 --n-val 100 --n-test 200
"""
import argparse
import ast
import json
import random
import re
from pathlib import Path

import pandas as pd

from recipe_parser.units import UNITS, format_qty, parse_ingredient, split_line, step_duration

URL = "https://huggingface.co/api/datasets/Hieu-Pham/kaggle_food_recipes/parquet/default/train/0.parquet"
OUT = Path(__file__).parent

POOLS = {
    "train": {
        "ing_header": ["Ingredients", "INGREDIENTS", "Ingredients:", "What You'll Need"],
        "step_header": ["Instructions", "Directions", "INSTRUCTIONS", "Method:"],
        "bullet": ["- ", "• ", "", "* "],
        "step_fmt": ["{i}. ", "Step {i}: ", "{i}) ", ""],
        "servings": ["Serves {n}", "Yield: {n} servings", "Servings: {n}"],
        "ads": ["ADVERTISEMENT", "Subscribe to our newsletter for weekly recipes!",
                "This post may contain affiliate links.", "Pin this recipe for later!",
                "Shop the kitchen tools I use in my Amazon storefront."],
        "story": [
            "I still remember the first time my grandmother made {title} for us.",
            "This recipe has been on repeat in our house all autumn long.",
            "Honestly, I wasn't sure about adding {ing} at first, but trust me on this one.",
            "We just got back from a trip and I was craving something comforting.",
            "My kids ask for this at least once a week, which says a lot.",
            "If you've been following the blog for a while, you know I love easy dinners.",
            "The secret really is good {ing}, so don't skimp on it.",
            "I tested this recipe five times before I was happy with it.",
            "It's the kind of dish that makes the whole kitchen smell amazing.",
            "Scroll down for the full printable recipe card!",
        ],
        "tail": ["Notes", "Nutrition: 420 kcal | 18g fat | 32g carbs",
                 "Did you make this recipe? Leave a comment below!",
                 "Comments (12)\nJenny: Made this last night, so good!!"],
    },
    "test": {
        "ing_header": ["You Will Need", "Shopping list for this recipe", "INGREDIENT LIST"],
        "step_header": ["How To Make It", "Preparation", "Let's Cook"],
        "bullet": ["▢ ", "o ", "→ "],
        "step_fmt": ["STEP {i} — ", "({i}) ", "#{i} "],
        "servings": ["Makes {n} portions", "Feeds {n} people"],
        "ads": ["Sponsored content", "Get 20% off your first meal kit with code YUM20",
                "Follow me on Instagram @cozykitchen", "Ad - continue reading below"],
        "story": [
            "Last summer, on a rainy afternoon at the lake house, I first tried {title}.",
            "People always ask me how I keep {ing} from going soggy.",
            "This one is dedicated to my neighbor Carol, who brings us soup every winter.",
            "Pro tip: read the whole recipe before you start cooking.",
            "Let me tell you, this disappeared in about ten minutes.",
        ],
        "tail": ["Recipe notes: keeps 3 days in the fridge.",
                 "Rate this recipe ★★★★★", "Reader questions\nTom: Can I freeze it? Reply: Yes!"],
    },
}


def load_recipes(seed: int) -> list[dict]:
    df = pd.read_parquet(URL).dropna(subset=["Title", "Ingredients", "Instructions"])
    recipes = []
    for row in df.sample(frac=1, random_state=seed).itertuples():
        lines = [l for l in ast.literal_eval(row.Ingredients) if l.strip()]
        steps = [s.strip() for s in row.Instructions.split("\n") if s.strip()]
        # keep short recipes so the JSON fits a small model's context comfortably
        if not (3 <= len(lines) <= 12 and 1 <= len(steps) <= 6 and len(row.Instructions) < 1200):
            continue
        ingredients = [parse_ingredient(l) for l in lines]
        if sum(i["quantity"] is not None for i in ingredients) < 0.7 * len(ingredients):
            continue
        recipes.append({"title": row.Title.strip(), "lines": lines, "steps": steps, "ingredients": ingredients})
    return recipes


def render_line(line: str, rng: random.Random) -> str:
    """Re-write an ingredient line with a random quantity style and unit spelling."""
    qty, unit, rest = split_line(line)
    parts = []
    if qty is not None:
        parts.append(format_qty(qty, rng.choice(["frac", "unicode", "decimal", "word"])))
    if unit:
        parts.append(rng.choice(UNITS[unit]))
    return " ".join(parts + [rest])


def render_page(r: dict, servings: int, pool: dict, rng: random.Random) -> str:
    fill = lambda s: s.format(title=r["title"].lower(), ing=rng.choice(r["ingredients"])["name"])
    out = [r["title"] + rng.choice(["", " | The Cozy Kitchen", " (Easy & Delicious!)"]), rng.choice(["Jump to Recipe", "Print", ""])]
    for _ in range(rng.randint(1, 3)):
        out.append(" ".join(fill(s) for s in rng.sample(pool["story"], rng.randint(2, 4))))
        if rng.random() < 0.5:
            out.append(rng.choice(pool["ads"]))
    out += ["", r["title"], rng.choice(pool["servings"]).format(n=servings),
            f"Prep Time: {rng.randint(5, 30)} mins | Cook Time: {rng.randint(10, 90)} mins", ""]

    bullet = rng.choice(pool["bullet"])
    out.append(rng.choice(pool["ing_header"]))
    lines = [bullet + render_line(l, rng) for l in r["lines"]]
    if rng.random() < 0.3:
        lines.insert(rng.randint(1, len(lines)), rng.choice(pool["ads"]))
    out += lines + ["", rng.choice(pool["step_header"])]

    fmt = rng.choice(pool["step_fmt"])
    out += [fmt.format(i=i) + s for i, s in enumerate(r["steps"], 1)]
    out += ["", *rng.sample(pool["tail"], rng.randint(1, 2))]
    return "\n".join(out)


def make_example(r: dict, split: str, rng: random.Random) -> dict:
    servings = rng.choice([2, 4, 4, 6, 8])
    target = {
        "title": r["title"],
        "servings": servings,
        "ingredients": r["ingredients"],
        "steps": [{"step": i, "text": s, "duration_minutes": step_duration(s)} for i, s in enumerate(r["steps"], 1)],
    }
    pool = POOLS["test" if split == "test" else "train"]
    return {"page": render_page(r, servings, pool, rng), "target": target}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-val", type=int, default=100)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    recipes = load_recipes(args.seed)
    print(f"{len(recipes)} usable recipes")
    sizes = {"test": args.n_test, "val": args.n_val, "train": args.n_train}
    start = 0
    for split, n in sizes.items():  # disjoint recipes across splits
        with open(OUT / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for i, r in enumerate(recipes[start:start + n]):
                ex = {"id": f"{split}-{i}", **make_example(r, split, rng)}
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        start += n
        print(f"{split}: {n} examples")


if __name__ == "__main__":
    main()
