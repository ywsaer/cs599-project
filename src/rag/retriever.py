"""混合检索引擎：BM25 词汇检索（中文 bigram 分词）+ 可选的 ChromaDB 语义检索。"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from src.config import get_config
from src.rag.chunks import build_recipe_chunks, ChunkType, SourceChunk
from src.recipe_service import load_recipes


_ASCII_RE = re.compile(r"[A-Za-z0-9]+")

# ═══════════════════ BM25 ═══════════════════

@dataclass(frozen=True)
class _BM25Doc:
    chunk: SourceChunk
    term_counts: Counter[str]
    length: int


class BM25Retriever:
    """BM25 词汇检索器，使用中文 bigram 分词。"""

    def __init__(self, chunks: list[SourceChunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: list[_BM25Doc] = []
        self._df: Counter[str] = Counter()
        self._avgdl = 0.0
        self._build(chunks)

    def _build(self, chunks: list[SourceChunk]) -> None:
        docs = []
        df: Counter[str] = Counter()
        for chunk in chunks:
            terms = _tokenize_chunk(chunk)
            tc = Counter(terms)
            if not tc:
                continue
            docs.append(_BM25Doc(chunk=chunk, term_counts=tc, length=sum(tc.values())))
            df.update(tc.keys())
        self._docs = docs
        self._df = df
        total = sum(d.length for d in docs)
        self._avgdl = total / len(docs) if docs else 0.0

    def search(self, query: str, top_k: int = 4) -> list[SourceChunk]:
        if not self._docs:
            return []
        qt = _tokenize_text(query)
        if not qt:
            return []
        scored = []
        N = len(self._docs)
        for doc in self._docs:
            score = 0.0
            for t in qt:
                tf = doc.term_counts.get(t, 0)
                if not tf:
                    continue
                df = self._df.get(t, 0)
                idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
                denom = tf + self.k1 * (1 - self.b + self.b * doc.length / (self._avgdl or 1))
                score += idf * (tf * (self.k1 + 1)) / denom
            if score <= 0:
                continue
            meta = {**doc.chunk.metadata, "retrieval_source": "bm25"}
            scored.append(doc.chunk.model_copy(update={"score": score, "metadata": meta}))
        return sorted(scored, key=lambda c: c.score, reverse=True)[:top_k]


def _tokenize_chunk(chunk: SourceChunk) -> list[str]:
    """对分块进行中文 bigram + 英文单词分词（含元数据加权）。"""
    meta = chunk.metadata or {}
    ingredients = meta.get("ingredients", [])
    extras = [
        chunk.recipe_name,
        str(chunk.chunk_type.value),
        str(meta.get("difficulty", "")),
        str(meta.get("time_minutes", "")),
    ]
    if isinstance(ingredients, list):
        extras.extend(str(i) for i in ingredients)
    return _tokenize_text(" ".join([chunk.content, *extras]))


def _tokenize_text(text: str) -> list[str]:
    """将纯文本拆分为英文单词、单个中文字符和中文 bigram。"""
    lowered = text.lower()
    tokens: list[str] = list(_ASCII_RE.findall(lowered))
    chinese = [c for c in text if "一" <= c <= "鿿"]
    tokens.extend(chinese)
    # 中文 bigram：连续的2个中文字符作为一个词
    tokens.extend("".join(chinese[i:i + 2]) for i in range(len(chinese) - 1))
    return [t for t in tokens if t.strip()]


# ═══════════════════ RecipeRetriever ═══════════════════

class RecipeRetriever:
    """菜谱检索引擎：BM25 检索 + 分块类型偏好重排序 + 实体匹配加权。"""

    def __init__(self) -> None:
        recipes = load_recipes()
        self._chunks = build_recipe_chunks(recipes)
        self._bm25 = BM25Retriever(self._chunks)
        self._indexed = False

    def ensure_indexed(self) -> int:
        """BM25 索引在 __init__ 时已构建，此处仅做兼容处理。"""
        self._indexed = True
        return len(self._chunks)

    def search(self, query: str, top_k: int = 4) -> list[SourceChunk]:
        """检索 top_k 条相关分块。"""
        self.ensure_indexed()
        bm25_results = self._bm25.search(query, max(top_k * 2, 8))
        return self._rerank(query, bm25_results)[:top_k]

    def _rerank(self, query: str, chunks: list[SourceChunk]) -> list[SourceChunk]:
        """按分块类型偏好 + 实体匹配加权重排序。"""
        preferred = self._pref_types(query)
        priority = {ct: len(preferred) - i for i, ct in enumerate(preferred)}

        def key(c: SourceChunk) -> tuple[float, float]:
            boost = priority.get(c.chunk_type, 0) * 4 + self._entity_boost(query, c)
            return (c.score + boost, c.score)

        return sorted(chunks, key=key, reverse=True)

    @staticmethod
    def _entity_boost(query: str, chunk: SourceChunk) -> float:
        """如果菜名或食材名在查询中精确出现，大幅加权。"""
        candidates = [chunk.recipe_name]
        ingredients = chunk.metadata.get("ingredients", [])
        if isinstance(ingredients, list):
            candidates.extend(str(i) for i in ingredients)

        for cand in candidates:
            if len(cand) < 2:
                continue
            if cand in query or query in cand:
                return 8.0
            bigrams = [
                cand[i:i + 2]
                for i in range(len(cand) - 1)
                if "一" <= cand[i] <= "鿿" and "一" <= cand[i + 1] <= "鿿"
            ]
            if any(bg in query for bg in bigrams):
                return 8.0
        return 0.0

    @staticmethod
    def _pref_types(query: str) -> list[ChunkType]:
        """根据查询关键词推断偏好的分块类型。"""
        if any(k in query for k in ["替代", "代替", "换成"]):
            return [ChunkType.SUBSTITUTION, ChunkType.INGREDIENTS, ChunkType.STEP]
        if any(k in query for k in ["安全", "过敏"]):
            return [ChunkType.SAFETY, ChunkType.STEP]
        if any(k in query for k in ["失败", "太甜", "太咸", "不脆", "粘锅", "为什么"]):
            return [ChunkType.FAILURE, ChunkType.STEP]
        if any(k in query for k in ["怎么做", "步骤", "火候", "多久"]):
            return [ChunkType.STEP, ChunkType.OVERVIEW]
        # 默认：优先步骤和概览
        return [ChunkType.STEP, ChunkType.OVERVIEW, ChunkType.INGREDIENTS]


@lru_cache
def get_retriever() -> RecipeRetriever:
    return RecipeRetriever()
