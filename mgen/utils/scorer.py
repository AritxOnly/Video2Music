from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import re


@dataclass(frozen=True)
class SemComponents:
    emb_cos: float          # raw cosine in [-1,1]
    emb_score: float        # mapped to [0,1]
    kw_jaccard: float       # [0,1]
    sem_score: float        # [0,1]


class Scorer:
    """
    评分逻辑集中管理：
    - 语义：BGE cosine + keyword jaccard (deterministic)
    - 同时产出可解释的 components，便于 debug / 训练监督
    """

    def __init__(
        self,
        sem_alpha: float = 0.75,   # emb vs keyword 融合权重
        emb_map: str = "cos_to_01",# cosine -> [0,1] 的映射方式
    ):
        self.sem_alpha = float(sem_alpha)
        self.emb_map = emb_map

    # ---------- public: semantic ----------

    def score_sem(
        self,
        query: str,
        visual_summary: str,
        track_style: str = "",
        track_tags: Optional[List[str]] = None,
        retriever_match_score: Optional[float] = None,  # raw cosine [-1,1]
        extra_text: str = "",  # 可选：title/artist等
    ) -> Tuple[float, SemComponents]:
        """
        返回 (sem_score, components)

        - retriever_match_score：来自 VectorMusicRetriever.match_score（raw cosine）
        - kw_jaccard：query+visual_summary tokens 与 track(style+tags+extra_text) tokens 的 jaccard
        """
        track_tags = track_tags or []

        q_text = f"{query} {visual_summary}".strip()
        d_text = f"{track_style} {' '.join(track_tags)} {extra_text}".strip()

        # 1) embedding part
        emb_cos = float(retriever_match_score) if retriever_match_score is not None else 0.0
        emb_score = self._map_cosine_to_01(emb_cos)

        # 2) keyword part (deterministic)
        kw_j = self._jaccard(self._tokenize(q_text), self._tokenize(d_text))

        # 3) fusion
        sem = self.sem_alpha * emb_score + (1.0 - self.sem_alpha) * kw_j
        sem = self._clamp01(sem)

        return sem, SemComponents(
            emb_cos=emb_cos,
            emb_score=emb_score,
            kw_jaccard=kw_j,
            sem_score=sem,
        )

    # ---------- helpers ----------

    def _map_cosine_to_01(self, cos: float) -> float:
        # retriever 的 util.cos_sim ∈ [-1,1]
        if math.isnan(cos):
            cos = 0.0
        if self.emb_map == "cos_to_01":
            return self._clamp01((cos + 1.0) / 2.0)
        # 你也可以改成 sigmoid 映射，但那会引入超参；先保持线性可解释
        return self._clamp01((cos + 1.0) / 2.0)

    @staticmethod
    def _tokenize(text: str) -> set:
        # 中文/英文都用最稳妥的：中文按字/词都不理想，这里用“字母数字词 + 单字中文”混合策略
        # 目标是可复现、轻量；不是追求最强 NLP。
        if not text:
            return set()

        text = text.lower()
        tokens = set()

        # 英文/数字词
        for m in re.finditer(r"[a-z0-9]+", text):
            tokens.add(m.group(0))

        # 中文单字（够用来做 overlap）
        for ch in text:
            if "\u4e00" <= ch <= "\u9fff":
                tokens.add(ch)

        return tokens

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _clamp01(x: float) -> float:
        return max(0.0, min(1.0, float(x)))