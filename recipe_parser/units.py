"""Units, quantity parsing/formatting and a deterministic ingredient-line parser."""
import re
from fractions import Fraction

# canonical unit -> surface forms (first one is the canonical spelling)
UNITS = {
    "cup": ["cup", "cups"],
    "tbsp": ["tbsp", "Tbsp.", "tablespoon", "tablespoons", "tbs"],
    "tsp": ["tsp", "tsp.", "teaspoon", "teaspoons"],
    "oz": ["oz", "oz.", "ounce", "ounces"],
    "lb": ["lb", "lb.", "lbs", "pound", "pounds"],
    "g": ["g", "gram", "grams"],
    "kg": ["kg", "kilogram", "kilograms"],
    "ml": ["ml", "milliliter", "milliliters"],
    "l": ["liter", "liters", "litre"],
    "pint": ["pint", "pints"],
    "quart": ["quart", "quarts", "qt."],
    "pinch": ["pinch", "pinches"],
    "clove": ["clove", "cloves"],
    "can": ["can", "cans"],
    "slice": ["slice", "slices"],
    "stick": ["stick", "sticks"],
    "sprig": ["sprig", "sprigs"],
    "bunch": ["bunch", "bunches"],
    "package": ["package", "packages", "pkg."],
}
ALIASES = {a.lower().rstrip("."): c for c, forms in UNITS.items() for a in forms}

# conversion to a base unit, used to merge shopping-list lines
TO_BASE = {
    "tsp": ("ml", 4.93), "tbsp": ("ml", 14.79), "cup": ("ml", 236.6), "pint": ("ml", 473.2),
    "quart": ("ml", 946.4), "ml": ("ml", 1), "l": ("ml", 1000),
    "g": ("g", 1), "kg": ("g", 1000), "oz": ("g", 28.35), "lb": ("g", 453.6),
}

UNICODE_FRAC = {"½": "1/2", "¼": "1/4", "¾": "3/4", "⅓": "1/3", "⅔": "2/3", "⅛": "1/8",
                "⅜": "3/8", "⅝": "5/8", "⅞": "7/8", "⅕": "1/5"}
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12}

NUM = r"(?:\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?)"
QTY_RE = re.compile(rf"^({NUM})(?:\s*(?:-|–|to)\s*{NUM})?\s*")


def normalize(text: str) -> str:
    for u, a in UNICODE_FRAC.items():
        text = re.sub(rf"(\d){u}", rf"\1 {a}", text).replace(u, a)
    return text.replace("⁄", "/")


def parse_number(s: str) -> float:
    return float(sum(Fraction(p) for p in s.split()))


def format_qty(q: float, style: str = "frac") -> str:
    if style == "decimal" or q != q:
        return f"{q:g}"
    if style == "word" and q in WORDS.values():
        return next(w for w, v in WORDS.items() if v == q)
    whole, frac = int(q), Fraction(q - int(q)).limit_denominator(8)
    if frac == 0:
        return str(whole)
    f = f"{frac.numerator}/{frac.denominator}"
    if style == "unicode":
        f = {v: k for k, v in UNICODE_FRAC.items()}.get(f, f)
        return f"{whole}{f}" if whole else f
    return f"{whole} {f}" if whole else f


def split_line(line: str) -> tuple[float | None, str | None, str]:
    """'2¾ tsp. kosher salt, divided' -> (2.75, 'tsp', 'kosher salt, divided')."""
    line = normalize(line.strip())
    qty = None
    first = line.split(" ", 1)[0].lower()
    if first in WORDS:
        qty, line = float(WORDS[first]), line[len(first):].lstrip()
    elif m := QTY_RE.match(line):
        qty, line = round(parse_number(m.group(1)), 3), line[m.end():]
    line = re.sub(r"^\([^)]*\)\s*", "", line)  # "1 (14-oz.) can" -> "can"
    unit = None
    word = line.split(" ", 1)[0]
    if qty is not None and word.lower().rstrip(".") in ALIASES:
        unit, line = ALIASES[word.lower().rstrip(".")], line[len(word):].lstrip()
    if line.lower().startswith("of "):
        line = line[3:]
    return qty, unit, line.strip()


def clean_name(rest: str) -> str:
    """Drop parentheses and anything after the first comma: the 'core' ingredient name."""
    rest = re.sub(r"\([^)]*\)", "", rest)
    return re.split(r",|;| plus | or ", rest)[0].strip(" .-").lower()


def parse_ingredient(line: str) -> dict:
    qty, unit, rest = split_line(line)
    return {"quantity": qty, "unit": unit, "name": clean_name(rest)}


DUR_RE = re.compile(rf"({NUM})(?:\s*(?:-|–|to)\s*{NUM})?\s*(minutes?|mins?|hours?|hrs?)\b", re.I)


def step_duration(text: str) -> int | None:
    """Total explicit time mentioned in a step, in minutes."""
    total = 0.0
    for m in DUR_RE.finditer(normalize(text)):
        total += parse_number(m.group(1)) * (60 if m.group(2).lower().startswith("h") else 1)
    return round(total) or None
