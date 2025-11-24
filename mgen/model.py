from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

@dataclass
class MusicTrack:
    """一首可用的 BGM 资产。"""
    id: str
    path: str                     # 本地文件路径，后面 ffmpeg 用
    style: str                    # 如 'lofi', 'edm', 'cinematic'
    mood: str                     # 如 'calm', 'excited'
    energy: float                 # 0.0 ~ 1.0，大致强度
    bpm: Optional[float] = None
    duration_sec: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentMusicPlan:
    """一个视频片段对应的音乐编排计划。"""
    start_sec: float
    end_sec: float
    track_id: str
    offset_sec: float = 0.0       # 从 BGM 第几秒开始用
    fade_in_sec: float = 0.5
    fade_out_sec: float = 0.5
    gain_db: float = -3.0         # 该段音乐音量调整
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MGenOptions:
    """整体配乐控制参数。"""
    preferred_style: Optional[str] = None   # 用户选的风格，如 'lofi', 'edm'
    global_gain_db: float = -5.0
    crossfade_sec: float = 0.3
    extra: Dict[str, Any] = field(default_factory=dict)