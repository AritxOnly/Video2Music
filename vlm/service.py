from vlm.interface import VLMInterface
from vlm.model import *

class VLMService:
    def __init__(self, vlm: VLMInterface):
        self._vlm = vlm

    def analyze_structure(
        self,
        video: VideoInput,
        prompt: Optional[str] = None,
        language: str = "en",
    ) -> VLMResult:
        """给上层 agent 用的“结构分析”封装。"""
        options = AnalysisOptions(
            task=TaskType.STRUCTURE,
            prompt=prompt or "Analyze the high-level structure of this video.",
            need_timeline=True,
            language=language,
        )
        return self._vlm.analyze(video, options)

    def caption(
        self,
        video: VideoInput,
        language: str = "en",
        prompt: Optional[str] = None,
    ) -> VLMResult:
        options = AnalysisOptions(
            task=TaskType.CAPTION,
            prompt=prompt or f"Generate a detailed caption in {language}.",
            language=language,
        )
        return self._vlm.analyze(video, options)