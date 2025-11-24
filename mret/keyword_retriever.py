from typing import List
from vsem.model import SegmentSemantics
from mgen.model import MusicTrack
from .interface import MusicRetrieverInterface

class KeywordMusicRetriever(MusicRetrieverInterface):
    """
    最简单版：“伪 RAG”：
    - 把 segment 的 mood/activity/tags 拼成一个 query string
    - 在 MusicTrack 的 style/mood/metadata 里做匹配
    - 打一个粗略得分，top_k 返回
    """

    def retrieve_for_segment(
        self,
        segment: SegmentSemantics,
        tracks: List[MusicTrack],
        top_k: int = 3,
    ) -> List[MusicTrack]:
        query = f"{segment.mood} {segment.activity} {' '.join(segment.tags)}"
        query = query.lower()

        scored: List[tuple[float, MusicTrack]] = []

        for t in tracks:
            score = 0.0
            text = f"{t.style} {t.mood} {t.metadata}".lower()
            if any(k in text for k in query.split()):
                score += 1.0
            # 可以继续根据 energy 匹配加分
            scored.append((score, t))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for s, t in scored[:top_k] if s > 0]