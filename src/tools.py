"""工具定义模块 — 使用 @tool 装饰器（与 LangChain 生态对齐）。"""

from langchain_core.tools import tool

from src.rag.retriever import get_retriever
from src.recipe_service import load_recipes


# ====== 工具 1：菜谱知识检索 ======

@tool
def search_recipes(query: str) -> str:
    """
    搜索菜谱知识库，返回相关菜谱的做法步骤、技巧、失败排查或替代方案。
    当用户问"怎么做"、"为什么失败"、"需要注意什么"、"有什么替代"时调用此工具。

    参数 query: 用户的具体问题，如'番茄炒蛋怎么做'、'可乐鸡翅为什么太甜'
    """
    retriever = get_retriever()
    retriever.ensure_indexed()
    chunks = retriever.search(query, top_k=4)

    if not chunks:
        return "知识库中没有找到相关信息。建议换一种问法或提供更具体的菜名。"

    lines = ["检索到以下相关资料："]
    for i, chunk in enumerate(chunks, start=1):
        lines.append(f"\n[来源 {i}] {chunk.recipe_name} / {chunk.chunk_type.value}")
        lines.append(chunk.content)
    return "\n".join(lines)


# ====== 工具 2：基于食材推荐菜谱 ======

@tool(description="根据用户提供的食材推荐合适的菜谱。")
def recommend_by_ingredients(ingredients: str) -> str:
    """
    根据用户提供的食材推荐合适的菜谱。
    当用户提供食材列表或问'今天吃什么'、'推荐一道菜'、'今晚吃什么'时调用此工具。

    参数 ingredients: 用户拥有的食材，用逗号分隔，如'鸡蛋, 番茄, 土豆'
    """
    user_ingredients = [i.strip() for i in ingredients.replace("，", ",").split(",") if i.strip()]
    if not user_ingredients:
        return "请提供至少一种食材，例如'鸡蛋, 番茄'。"

    results: list[dict] = []
    for recipe in load_recipes():
        required = [i.ingredient for i in recipe.ingredients if i.required]
        matched = [i for i in required if any(
            i in ui or ui in i for ui in user_ingredients
        )]
        missing = [i for i in required if i not in matched]
        match_ratio = len(matched) / len(required) if required else 0.0
        score = match_ratio * 60

        if recipe.beginner_friendly:
            score += 15
        if recipe.difficulty == "easy":
            score += 10
        elif recipe.difficulty == "hard":
            score -= 15
        if matched:
            score += len(matched) * 5

        results.append({
            "recipe_name": recipe.name,
            "score": score,
            "matched": matched,
            "missing": missing,
            "difficulty": recipe.difficulty,
            "time": recipe.time_minutes,
            "beginner_friendly": recipe.beginner_friendly,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    top = results[:3]
    top = [r for r in top if r["score"] > 0]

    if not top:
        return f"用 {', '.join(user_ingredients[:5])} 没有找到匹配的菜谱。建议补充更多食材。"

    lines = ["根据你已有的食材，推荐如下："]
    for i, r in enumerate(top, start=1):
        missing_str = "、".join(r["missing"]) if r["missing"] else "无"
        matched_str = "、".join(r["matched"]) if r["matched"] else "无"
        beginner_note = "（新手友好）" if r["beginner_friendly"] else ""
        lines.append(
            f"{i}. {r['recipe_name']}{beginner_note}：难度 {r['difficulty']}，"
            f"约 {r['time']} 分钟。"
            f"匹配食材：{matched_str}。还需准备：{missing_str}。"
        )
    return "\n".join(lines)


# ====== 工具 3：获取用户画像 ======

# 预定义的用户画像数据
_USER_PROFILES = {
    "demo_user": {
        "role_description": (
            "一位刚学会开火的厨房新人，会做番茄炒蛋和酸辣土豆丝这两道拿手菜，"
            "对炒、煮、蒸有一点概念但火候还掌握不好。喜欢鸡蛋、土豆、鸡翅、番茄这些常见食材，"
            "看到'红烧'、'干煸'这类词就会紧张。每顿饭希望控制在半小时以内搞定，"
            "太难太复杂的菜暂时不敢碰。"
        ),
        "liked_ingredients": ["鸡蛋", "土豆", "鸡翅", "番茄"],
        "max_time_minutes": 30,
        "avoid_difficulty": "hard",
    },
    "novice_user": {
        "role_description": (
            "几乎零基础的厨房小白，连切菜都还不太利索。最能驾驭的食材就是鸡蛋、番茄、土豆这三样，"
            "讨厌油炸（怕油溅）、大火爆炒（怕糊锅）、复杂肉菜（处理不来生肉）。"
            "最多接受 20 分钟的烹饪时间，超过这个时间会觉得太麻烦。"
            "只敢做简单级别的菜，看到'中等难度'就觉得压力山大。"
        ),
        "liked_ingredients": ["鸡蛋", "番茄", "土豆"],
        "max_time_minutes": 20,
        "avoid_difficulty": "medium",
    },
    "expert_user": {
        "role_description": (
            "下厨多年、手艺娴熟的老饕，家里锅具调料一应俱全。偏爱牛腩、鲜鱼、鸡翅、虾、五花肉这类有质感的食材，"
            "不排斥任何烹饪风格，油炸爆炒慢炖红烧都信手拈来。享受烹饪过程，不介意花一两个小时慢慢做一道大菜，"
            "挑战中等甚至高难度的菜谱反而让他更有成就感。"
        ),
        "liked_ingredients": ["牛腩", "鲜鱼", "鸡翅", "虾", "五花肉"],
        "max_time_minutes": 120,
        "avoid_difficulty": "",
    },
}


@tool
def get_user_profile(user_id: str = "demo_user") -> str:
    """
    获取用户的烹饪偏好画像（以生动的人物角色描述）。
    在推荐菜谱前务必调用此工具，以便做出个性化推荐。

    参数 user_id: 用户标识，如 'demo_user'、'novice_user'、'expert_user'。默认为 'demo_user'。
    """
    profile = _USER_PROFILES.get(user_id, _USER_PROFILES["demo_user"])

    return (
        f"=== 当前用户的画像 ===\n"
        f"{profile['role_description']}\n"
        f"偏好的食材：{'、'.join(profile['liked_ingredients'])}\n"
        f"能接受的烹饪时间上限：{profile['max_time_minutes']} 分钟\n"
        f"不推荐的难度级别：{profile['avoid_difficulty'] or '无限制'}\n"
        f"========================\n"
        f"请根据以上画像特点来调整回答的详细程度、措辞风格和推荐逻辑。"
    )


# ====== 工具列表 ======

TOOLS = [search_recipes, recommend_by_ingredients, get_user_profile]
