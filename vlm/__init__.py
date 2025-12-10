from .model import (
    TaskType,
    VideoInput,
    AnalysisOptions,
    TimelineEvent,
    BeatInfo,
    VLMResult,
    vlm_result_to_dict,
)

from .interface import VLMInterface

# 各个后端实现
from .qwen.interface import QwenVLWebInterface
from .qwen_seg.interface import QwenVLSegmentWebInterface

from .utils.sampler import global_sampler

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
    "QwenVLSegmentWebInterface",
    # 工具函数
    "global_sampler",
]

from .factory import get_vlm, VLMBuiltinName

__all__.extend(["get_vlm", "VLMBuiltinName"])