from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import asdict

class TaskType(str, Enum):
    QA = "qa"                        # 问答
    TAGGING = "tagging"              # 标签、分类
    STRUCTURE = "structure"          # 结构分析（分段、Hook、高潮等）
    DETECTION = "detection"          # 事件 / 物体检测


@dataclass
class VideoInput:
    """统一表示一段视频，无论是本地、URL、还是已经抽好的帧。"""
    path: Optional[str] = None          # 本地路径
    url: Optional[str] = None           # 远程 URL
    fps: Optional[float] = None
    duration_sec: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisOptions:
    """任务级参数：你之后想做音乐结构分析、配乐等都往这加字段。"""
    task: TaskType
    prompt: Optional[str] = None           # VLM 指令，如“分析段落结构”
    max_tokens: int = 512
    temperature: float = 0.2
    need_timeline: bool = False
    language: str = "en"                   # 返回语言


@dataclass
class TimelineEvent:
    """时间轴上的结构事件，用于后续 agent 对齐（比如 hook/verse/chorus）。"""
    start_sec: float
    end_sec: float
    label: str                   # 如 "intro", "verse", "chorus", "drop"
    description: str = ""


@dataclass
class BeatInfo:
    time_sec: float
    strength: float


@dataclass
class VLMResult:
    """统一的结构化输出：文本 + 时间轴 + 模型内部信息。"""
    raw_text: str
    timeline: List[TimelineEvent] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    beats: List[BeatInfo] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)  # e.g. logprobs, model_name, cost
    
def vlm_result_to_dict(r: VLMResult) -> dict:
    """
    把 VLMResult（含内部 TimelineEvent / BeatInfo）转成纯 dict，
    方便 json.dump。
    """
    return asdict(r)