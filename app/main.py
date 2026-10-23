"""HTTP API (FastAPI) + a small Gradio demo at /demo, serving the fine-tuned GGUF model on CPU.

    MODEL_PATH=outputs/recipe-parser-q8_0.gguf uvicorn app.main:app --port 8080
"""
import json
import os

import gradio as gr
from fastapi import FastAPI, HTTPException
from llama_cpp import Llama
from pydantic import BaseModel, ValidationError

from recipe_parser.kitchen import scale, shopping_list
from recipe_parser.prompt import messages, parse_output
from recipe_parser.schema import Ingredient, Recipe

llm = Llama(model_path=os.environ.get("MODEL_PATH", "/models/model.gguf"),
            n_ctx=4096, n_threads=int(os.environ.get("N_THREADS", 4)), verbose=False)
app = FastAPI(title="recipe-parser")


class ParseRequest(BaseModel):
    text: str


class ScaleRequest(BaseModel):
    recipe: Recipe
    servings: int


def extract(text: str) -> Recipe:
    out = llm.create_chat_completion(messages(text), temperature=0, max_tokens=1024)
    try:
        return Recipe(**parse_output(out["choices"][0]["message"]["content"]))
    except (TypeError, ValidationError):
        raise HTTPException(422, "model output is not a valid recipe")


@app.post("/parse", response_model=Recipe)
def parse(req: ParseRequest):
    return extract(req.text)


@app.post("/scale", response_model=Recipe)
def scale_recipe(req: ScaleRequest):
    return scale(req.recipe, req.servings)


@app.post("/shopping-list", response_model=list[Ingredient])
def shopping(recipes: list[Recipe]):
    return shopping_list(recipes)


@app.get("/health")
def health():
    return {"status": "ok"}


def demo_fn(text, servings):
    recipe = extract(text)
    if servings and recipe.servings:
        recipe = scale(recipe, int(servings))
    items = [f"- {i.quantity or ''} {i.unit or ''} {i.name}".replace("  ", " ") for i in shopping_list([recipe])]
    return recipe.model_dump(), "\n".join(items)


demo = gr.Interface(
    demo_fn,
    [gr.Textbox(lines=18, label="Paste a recipe blog page"), gr.Number(label="Servings (optional)", precision=0)],
    [gr.JSON(label="Structured recipe"), gr.Markdown(label="Shopping list")],
    title="recipe-parser",
    description="Fine-tuned Qwen2.5-1.5B (QLoRA) that turns noisy recipe pages into structured JSON.",
)
app = gr.mount_gradio_app(app, demo, path="/demo")
