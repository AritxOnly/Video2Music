from dataclasses import dataclass, field
from typing import List, Dict, Any
from vlm.model import TimelineEvent

@dataclass
class SegmentSemantics:
    start_sec: float
    end_sec: float
    mood: str                 # 结构输出里的 mood / label
    activity: str             # 简短动作描述（如 "滑板跳跃"）
    tags: List[str] = field(default_factory=list)
    text_summary: str = ""    # 合成的一小段 summary
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoSemantics:
    """整个视频的统一语义视图。"""
    global_tags: List[str]
    segments: List[SegmentSemantics]
    extra: Dict[str, Any] = field(default_factory=dict)