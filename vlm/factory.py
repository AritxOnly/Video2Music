from typing import Literal

from .interface import VLMInterface
from .qwen.interface import QwenVLWebInterface
from .sample.interface import SampleVLMOutputInterface
# from .openai.interface import OpenAIVLInterface
# from .local.interface import LocalVLInterface

VLMBuiltinName = Literal["qwen-web", "qwen-seg-web", "sample", ]  # 后面有别的可以往里加

def get_vlm(name: VLMBuiltinName, **kwargs) -> VLMInterface:
    """
    统一创建 VLM 实例的工厂。
    kwargs 直接传给对应实现的 __init__。
    """
    if name == "qwen-web":
        return QwenVLWebInterface(**kwargs)
    if name == "qwen-seg-web":
        from .qwen_seg.interface import QwenVLSegmentWebInterface
        return QwenVLSegmentWebInterface(**kwargs)
    if name == "sample":
        return SampleVLMOutputInterface(**kwargs)

    raise ValueError(f"Unknown VLM backend: {name}")