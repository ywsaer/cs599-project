"""菜谱数据加载服务。"""

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

from src.config import get_config


class Ingredient(BaseModel):
    name: str = ""
    category: str = "other"


class RecipeStep(BaseModel):
    order: int = 0
    content: str = ""
    beginner_tip: str = ""
    risk_tip: str = ""


class RecipeIngredient(BaseModel):
    ingredient: str = ""
    amount: str = ""
    required: bool = True


class Recipe(BaseModel):
    id: int = 0
    name: str = ""
    description: str = ""
    difficulty: str = "easy"
    time_minutes: int = 30
    beginner_friendly: bool = True
    cuisine: str = "家常菜"
    seasons: list[str] = Field(default_factory=list)
    ingredients: list[RecipeIngredient] = Field(default_factory=list)
    steps: list[RecipeStep] = Field(default_factory=list)
    common_failures: list[str] = Field(default_factory=list)
    substitutions: dict[str, str] = Field(default_factory=dict)
    safety_notes: list[str] = Field(default_factory=list)


def _parse_recipe(raw: dict) -> Recipe:
    return Recipe(
        id=raw.get("id", 0),
        name=raw.get("name", ""),
        description=raw.get("description", ""),
        difficulty=raw.get("difficulty", "easy"),
        time_minutes=raw.get("time_minutes", 30),
        beginner_friendly=raw.get("beginner_friendly", True),
        cuisine=raw.get("cuisine", "家常菜"),
        seasons=raw.get("seasons", []),
        ingredients=[
            RecipeIngredient(
                ingredient=i.get("ingredient", ""),
                amount=i.get("amount", ""),
                required=i.get("required", True),
            )
            for i in raw.get("ingredients", [])
        ],
        steps=[
            RecipeStep(
                order=s.get("order", 0),
                content=s.get("content", ""),
                beginner_tip=s.get("beginner_tip", ""),
                risk_tip=s.get("risk_tip", ""),
            )
            for s in raw.get("steps", [])
        ],
        common_failures=raw.get("common_failures", []),
        substitutions=raw.get("substitutions", {}),
        safety_notes=raw.get("safety_notes", []),
    )


@lru_cache
def load_recipes() -> list[Recipe]:
    config = get_config()
    path = Path(config.recipes_path)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [_parse_recipe(item) for item in data]
