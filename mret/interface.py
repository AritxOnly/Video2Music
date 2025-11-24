from typing import Protocol, List
from vsem.model import SegmentSemantics, VideoSemantics
from mgen.model import MusicTrack

class MusicRetrieverInterface(Protocol):
    """给定视频语义（全局/片段），从曲库里找候选 BGM。"""

    def retrieve_for_segment(
        self,
        segment: SegmentSemantics,
        tracks: List[MusicTrack],
        top_k: int = 3,
    ) -> List[MusicTrack]:
        ...