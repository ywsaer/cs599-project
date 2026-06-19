"""从菜谱数据构建语义分块，用于 RAG 检索。"""

from hashlib import sha256
from enum import StrEnum
from pydantic import BaseModel, Field

from src.recipe_service import Recipe, RecipeIngredient


class ChunkType(StrEnum):
    OVERVIEW = "overview"
    INGREDIENTS = "ingredients"
    STEP = "step"
    BEGINNER_TIP = "beginner_tip"
    FAILURE = "failure"
    SUBSTITUTION = "substitution"
    SAFETY = "safety"


class SourceChunk(BaseModel):
    recipe_id: int = 0
    recipe_name: str = ""
    chunk_type: ChunkType = ChunkType.OVERVIEW
    content: str = ""
    score: float = 0.0
    metadata: dict = Field(default_factory=dict)


CHUNK_SCHEMA_VERSION = 1


def build_recipe_chunks(recipes: list[Recipe]) -> list[SourceChunk]:
    chunks: list[SourceChunk] = []
    for recipe in recipes:
        chunks.append(_overview(recipe))
        chunks.append(_ingredients(recipe))
        chunks.extend(_steps(recipe))
        chunks.extend(_beginner_tips(recipe))
        chunks.extend(_failures(recipe))
        chunks.extend(_substitutions(recipe))
        chunks.extend(_safety_chunks(recipe))
    return chunks


def _make(recipe: Recipe, chunk_type: ChunkType, content: str, key: str) -> SourceChunk:
    chunk_id = f"recipe:{recipe.id}:{key}:v{CHUNK_SCHEMA_VERSION}"
    return SourceChunk(
        recipe_id=recipe.id,
        recipe_name=recipe.name,
        chunk_type=chunk_type,
        content=content,
        metadata={
            "chunk_id": chunk_id,
            "content_hash": sha256(content.encode()).hexdigest(),
            "difficulty": recipe.difficulty,
            "beginner_friendly": recipe.beginner_friendly,
            "time_minutes": recipe.time_minutes,
            "cuisine": recipe.cuisine,
            "ingredients": _names(recipe),
        },
    )


def _names(recipe: Recipe) -> list[str]:
    return [i.ingredient for i in recipe.ingredients]


def _overview(recipe: Recipe) -> SourceChunk:
    content = "\n".join([
        f"菜谱：{recipe.name}",
        f"类型：菜谱概览",
        f"简介：{recipe.description}",
        f"难度：{recipe.difficulty}",
        f"预计耗时：{recipe.time_minutes} 分钟",
        f"新手友好：{'是' if recipe.beginner_friendly else '否'}",
        f"菜系：{recipe.cuisine}",
    ])
    return _make(recipe, ChunkType.OVERVIEW, content, "overview")


def _ingredients(recipe: Recipe) -> SourceChunk:
    required = [i for i in recipe.ingredients if i.required]
    optional = [i for i in recipe.ingredients if not i.required]

    def fmt(items):
        return "、".join(f"{i.ingredient}（{i.amount}）" if i.amount else i.ingredient for i in items) or "无"

    content = "\n".join([
        f"菜谱：{recipe.name}",
        f"类型：食材清单",
        f"必需食材：{fmt(required)}",
        f"可选食材：{fmt(optional)}",
    ])
    return _make(recipe, ChunkType.INGREDIENTS, content, "ingredients")


def _steps(recipe: Recipe) -> list[SourceChunk]:
    chunks = []
    for step in recipe.steps:
        lines = [f"菜谱：{recipe.name}", f"类型：制作步骤", f"步骤 {step.order}：{step.content}"]
        if step.beginner_tip:
            lines.append(f"新手提示：{step.beginner_tip}")
        if step.risk_tip:
            lines.append(f"风险提示：{step.risk_tip}")
        chunks.append(_make(recipe, ChunkType.STEP, "\n".join(lines), f"step:{step.order}"))
    return chunks


def _beginner_tips(recipe: Recipe) -> list[SourceChunk]:
    chunks = []
    for step in recipe.steps:
        if not step.beginner_tip:
            continue
        content = "\n".join([
            f"菜谱：{recipe.name}", f"类型：新手提示",
            f"步骤 {step.order}：{step.beginner_tip}",
        ])
        chunks.append(_make(recipe, ChunkType.BEGINNER_TIP, content, f"beginner_tip:{step.order}"))
    return chunks


def _failures(recipe: Recipe) -> list[SourceChunk]:
    chunks = []
    for i, failure in enumerate(recipe.common_failures, start=1):
        content = "\n".join([f"菜谱：{recipe.name}", f"类型：失败排查", f"常见问题 {i}：{failure}"])
        chunks.append(_make(recipe, ChunkType.FAILURE, content, f"failure:{i}"))
    return chunks


def _substitutions(recipe: Recipe) -> list[SourceChunk]:
    chunks = []
    for i, (ingredient, substitute) in enumerate(recipe.substitutions.items(), start=1):
        content = "\n".join([
            f"菜谱：{recipe.name}", f"类型：食材替代",
            f"原食材：{ingredient}", f"替代方案：{substitute}",
        ])
        chunks.append(_make(recipe, ChunkType.SUBSTITUTION, content, f"substitution:{i}"))
    return chunks


def _safety_chunks(recipe: Recipe) -> list[SourceChunk]:
    chunks = []
    for i, note in enumerate(recipe.safety_notes, start=1):
        content = "\n".join([f"菜谱：{recipe.name}", f"类型：安全提醒", f"安全事项 {i}：{note}"])
        chunks.append(_make(recipe, ChunkType.SAFETY, content, f"safety:{i}"))
    return chunks
