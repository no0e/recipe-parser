"""Score prediction files against the gold test set and print a markdown table.

    python -m eval.evaluate eval/preds/*.jsonl
"""
import sys
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from eval.predict import read_jsonl
from recipe_parser.prompt import parse_output
from recipe_parser.schema import Recipe


def f1(pred: list, gold: list) -> float:
    tp = sum((Counter(pred) & Counter(gold)).values())
    if not pred or not gold:
        return float(pred == gold)
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r) if tp else 0.0


def ing_key(i, full=True):
    name = i.name.strip().lower()
    return (round(i.quantity, 2) if i.quantity is not None else None, i.unit, name) if full else name


def score(gold: dict, output: str) -> dict:
    g = Recipe(**gold)
    try:
        p = Recipe(**parse_output(output))
    except (TypeError, ValidationError):
        return {"valid_json": 0, "ingredient_f1": 0, "name_f1": 0, "servings_acc": 0, "steps_count_acc": 0, "duration_acc": 0}
    durations = [ps.duration_minutes == gs.duration_minutes for ps, gs in zip(p.steps, g.steps)]
    return {
        "valid_json": 1,
        "ingredient_f1": f1([ing_key(i) for i in p.ingredients], [ing_key(i) for i in g.ingredients]),
        "name_f1": f1([ing_key(i, False) for i in p.ingredients], [ing_key(i, False) for i in g.ingredients]),
        "servings_acc": float(p.servings == g.servings),
        "steps_count_acc": float(len(p.steps) == len(g.steps)),
        "duration_acc": sum(durations) / len(g.steps),
    }


def main(paths):
    gold = {ex["id"]: ex["target"] for ex in read_jsonl("data/test.jsonl")}
    rows = []
    for path in paths:
        scores = [score(gold[pr["id"]], pr["output"]) for pr in read_jsonl(path)]
        rows.append((Path(path).stem, {k: sum(s[k] for s in scores) / len(scores) for k in scores[0]}))
    keys = list(rows[0][1])
    print("| method | " + " | ".join(keys) + " |")
    print("|---" * (len(keys) + 1) + "|")
    for name, avg in rows:
        print(f"| {name} | " + " | ".join(f"{avg[k]:.3f}" for k in keys) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
