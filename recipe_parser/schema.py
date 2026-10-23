from pydantic import BaseModel


class Ingredient(BaseModel):
    quantity: float | None = None
    unit: str | None = None
    name: str


class Step(BaseModel):
    step: int
    text: str
    duration_minutes: int | None = None


class Recipe(BaseModel):
    title: str
    servings: int | None = None
    ingredients: list[Ingredient]
    steps: list[Step]
