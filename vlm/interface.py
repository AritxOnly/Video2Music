from typing import Protocol, runtime_checkable
from vlm.model import *

@runtime_checkable
class VLMInterface(Protocol):
    """视频语言模型的统一接口，所有后端都实现这个协议即可。"""

    def analyze(self, video: VideoInput, options: AnalysisOptions) -> VLMResult:
        ...

    def get_backend_name(self) -> str:
        ...
        