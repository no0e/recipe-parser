"""Deterministic post-processing: scaling servings and merging a shopping list."""
from collections import defaultdict

from .schema import Ingredient, Recipe
from .units import TO_BASE


def scale(recipe: Recipe, servings: int) -> Recipe:
    if not recipe.servings:
        raise ValueError("recipe has no base servings, cannot scale")
    factor = servings / recipe.servings
    ingredients = [
        i.model_copy(update={"quantity": round(i.quantity * factor, 2) if i.quantity else i.quantity})
        for i in recipe.ingredients
    ]
    return recipe.model_copy(update={"servings": servings, "ingredients": ingredients})


def shopping_list(recipes: list[Recipe]) -> list[Ingredient]:
    """Merge ingredients across recipes: same name + compatible units are summed."""
    groups = defaultdict(list)  # (name, base unit or raw unit) -> ingredients
    for r in recipes:
        for i in r.ingredients:
            base = TO_BASE.get(i.unit, (i.unit, 1))[0]
            groups[(i.name, base)].append(i)

    merged = []
    for (name, base), items in groups.items():
        if any(i.quantity is None for i in items):
            merged.append(Ingredient(name=name, unit=items[0].unit,
                                     quantity=None if len(items) > 1 else items[0].quantity))
            continue
        if items[0].unit not in TO_BASE:
            merged.append(Ingredient(name=name, unit=items[0].unit, quantity=sum(i.quantity for i in items)))
            continue
        total = sum(i.quantity * TO_BASE[i.unit][1] for i in items)
        # express the total in the largest unit used in the inputs
        unit = max({i.unit for i in items}, key=lambda u: TO_BASE[u][1])
        merged.append(Ingredient(name=name, unit=unit, quantity=round(total / TO_BASE[unit][1], 2)))
    return merged
