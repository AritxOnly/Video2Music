from __future__ import annotations

from typing import Protocol, runtime_checkable, List, Optional

from vlm.model import TimelineEvent
from .model import MusicTrack, SegmentMusicPlan, MGenOptions


@runtime_checkable
class MusicLibraryInterface(Protocol):
    """音乐资产来源（本地/远程都可以）"""

    def list_tracks(self, style: Optional[str] = None) -> List[MusicTrack]:
        ...


@runtime_checkable
class MusicArrangerInterface(Protocol):
    """把视频时间线 + 音乐库 → 编排计划"""

    def arrange(
        self,
        timeline: List[TimelineEvent],
        library: MusicLibraryInterface,
        options: Optional[MGenOptions] = None,
    ) -> List[SegmentMusicPlan]:
        ...


@runtime_checkable
class MusicGeneratorInterface(Protocol):
    """
    预留给 Suno / MusicGen 这种生成式模型用：
    输入 prompt + 时长 → 输出一段音频文件。
    """

    def generate(self, prompt: str, duration_sec: float) -> str:
        """返回生成好的本地音频文件路径"""
        ...