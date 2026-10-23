"""Rule-based baseline: section headers + regexes, written by looking at training pages only."""
import re

from .units import parse_ingredient, split_line, step_duration

ING_HEADER = re.compile(r"^(ingredients?|what you'?ll need)\s*:?$", re.I)
STEP_HEADER = re.compile(r"^(instructions|directions|method|steps|preparation)\s*:?$", re.I)
SERVINGS = re.compile(r"\b(serves|servings|yield|makes)\b\D{0,15}(\d+)", re.I)
BULLET = re.compile(r"^[^\w(]+\s*")
STEP_PREFIX = re.compile(r"^\W*(step\s*)?\d+\s*[^\w\s]?\s*", re.I)


def is_noise(line: str) -> bool:
    return line.isupper() or "!" in line or any(w in line.lower() for w in ("subscribe", "affiliate", "amazon", "pin this"))


def extract(page: str) -> dict:
    lines = [l.strip() for l in page.splitlines()]
    title = re.split(r" \| | \(", lines[0])[0].strip()
    m = SERVINGS.search(page)

    ing_i = next((i for i, l in enumerate(lines) if ING_HEADER.match(l)), None)
    step_i = next((i for i, l in enumerate(lines) if STEP_HEADER.match(l)), None)
    if ing_i is None:  # fallback: first block of lines starting with a quantity
        ing_i = next((i - 1 for i, l in enumerate(lines) if split_line(BULLET.sub("", l))[0] is not None and len(l) < 80), len(lines))

    ingredients = []
    for i, l in enumerate(lines[ing_i + 1:], ing_i + 1):
        if not l or i == step_i:
            break
        if not is_noise(l):
            ingredients.append(parse_ingredient(BULLET.sub("", l)))

    steps = []
    start = step_i if step_i is not None else ing_i + 1 + len(ingredients)
    for l in lines[start + 1:]:
        if not l and steps:
            break
        if l and not is_noise(l):
            text = STEP_PREFIX.sub("", l)
            steps.append({"step": len(steps) + 1, "text": text, "duration_minutes": step_duration(text)})

    return {"title": title, "servings": int(m.group(2)) if m else None, "ingredients": ingredients, "steps": steps}
