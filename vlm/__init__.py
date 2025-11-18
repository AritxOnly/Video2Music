from .model import (
    TaskType,
    VideoInput,
    AnalysisOptions,
    TimelineEvent,
    BeatInfo,
    VLMResult,
)

from .interface import VLMInterface

# 各个后端实现
from .qwen.interface import QwenVLWebInterface

__all__ = [
    # 模型基础类型
    "TaskType",
    "VideoInput",
    "AnalysisOptions",
    "TimelineEvent",
    "BeatInfo",
    "VLMResult",
    # 抽象接口
    "VLMInterface",
    # 具体实现
    "QwenVLWebInterface",
    # "OpenAIVLInterface",
    # "LocalVLInterface",
]

from .factory import get_vlm, VLMBuiltinName

__all__.extend(["get_vlm", "VLMBuiltinName"])