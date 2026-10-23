"""Prompt shared by training, evaluation and the API."""
import json
import re

SYSTEM = (
    "You extract recipes from messy blog pages. Ignore stories, ads and comments. "
    'Answer with JSON only: {"title": str, "servings": int|null, '
    '"ingredients": [{"quantity": float|null, "unit": str|null, "name": str}], '
    '"steps": [{"step": int, "text": str, "duration_minutes": int|null}]}. '
    "Units are canonical (cup, tbsp, tsp, oz, lb, g, kg, ml, l, pinch, clove, can, ...); "
    "names are lowercase without preparation notes; step texts are copied verbatim."
)


def messages(page: str, target: dict | None = None, shots: list[dict] = ()) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM}]
    for ex in shots:
        msgs += [{"role": "user", "content": ex["page"]},
                 {"role": "assistant", "content": json.dumps(ex["target"], ensure_ascii=False)}]
    msgs.append({"role": "user", "content": page})
    if target is not None:
        msgs.append({"role": "assistant", "content": json.dumps(target, ensure_ascii=False)})
    return msgs


def parse_output(text: str) -> dict | None:
    """Return the first JSON object in the model output, or None."""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None
