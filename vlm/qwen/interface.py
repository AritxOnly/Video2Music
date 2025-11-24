import os
from typing import Any

from dashscope import MultiModalConversation
from pathlib import Path

from vlm.service import VLMInterface
from vlm.model import *
from .utils import *


class QwenVLWebInterface(VLMInterface):
    """
    使用 DashScope MultiModalConversation 的 Qwen-VL 实现。
    支持本地 file:// 路径和远程 URL 视频输入。
    只负责“看视频 + 输出文本”，不在这里解析 timeline / tags / beats。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen2.5-vl-72b-instruct",
    ):
        # 优先用参数，其次 DASHSCOPE_API_KEY / QWEN_API_KEY
        self._api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        if not self._api_key:
            raise EnvironmentError("缺少 DashScope API Key，请设置 DASHSCOPE_API_KEY 或 QWEN_API_KEY")
        self._model = model

    def get_backend_name(self) -> str:
        return self._model

    def analyze(self, video: VideoInput, options: AnalysisOptions) -> VLMResult:
        """
        使用 DashScope MultiModalConversation 调用 Qwen-VL。
        支持：
        - video.url: 远程 URL
        - video.path: 本地文件（转成 file:// 形式）
        """

        # 1. 解析视频输入 → file:// 或 URL
        if video.url:
            video_source = video.url
        elif video.path:
            abs_path = Path(video.path).expanduser().absolute()
            video_source = f"file://{abs_path}"
        else:
            raise ValueError("VideoInput 必须至少提供 url 或 path 之一。")

        fps = int(video.fps or 1)

        # 2. 构造任务 prompt（只管结构/标签/QA/检测）
        if options.prompt is not None:
            base_prompt = options.prompt
        else:
            if options.task == TaskType.QA:
                base_prompt = (
                    "你是一个严格的“视频解读助手”，只能基于画面和声音回答问题，"
                    "禁止臆测看不见的内容。"
                )
            elif options.task == TaskType.TAGGING:
                base_prompt = (
                    "请生成简洁的标签，描述视频的内容、场景、色彩风格、镜头运动和情绪。"
                    "使用短语，每行一个标签。"
                )
            elif options.task == TaskType.STRUCTURE:
                base_prompt = (
                    "请分析视频的时间结构，把视频划分为若干片段。"
                    "对每个片段给出：start_time（秒）、end_time（秒）、description（中文描述）、mood（情绪）。"
                    "请严格输出一个 JSON 对象，包含 segments 数组。"
                )
            elif options.task == TaskType.DETECTION:
                base_prompt = (
                    "请描述视频中出现的重要人物、物体和关键事件。"
                    "如果可以，请大致指出它们出现在视频中的时间范围（用秒表示）。"
                )
            else:
                raise NotImplementedError(f"Task {options.task} 不由 QwenVLWebInterface 支持。")

        # 语言控制
        lang = (options.language or "").lower()
        if lang.startswith("zh"):
            base_prompt += " 请用中文回答。"
        elif lang.startswith("en"):
            base_prompt += " Please answer in English."

        # 时间线提示
        if options.need_timeline:
            base_prompt += " 请尽量明确给出以秒为单位的时间范围。"

        # 3. 构造 DashScope MultiModalConversation 消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "video": video_source,
                        "fps": fps,
                    },
                    {
                        "text": base_prompt,
                    },
                ],
            }
        ]

        # 4. 调用 DashScope
        response = MultiModalConversation.call(
            api_key=self._api_key,
            model=self._model,
            messages=messages,
        )

        # 5. 解析返回内容（尽量稳妥）
        output = response.get("output", {}) or {}
        choices = output.get("choices", []) or []
        if not choices:
            raise ValueError(f"DashScope 返回中没有 choices，原始响应: {response}")

        msg = choices[0].get("message", {}) or {}
        content = msg.get("content")

        if isinstance(content, list):
            # 常见格式：[{ "text": "..." }]
            text_piece = ""
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    text_piece = part["text"]
                    break
            answer_text = text_piece
        elif isinstance(content, dict) and "text" in content:
            answer_text = content["text"]
        elif isinstance(content, str):
            answer_text = content
        else:
            answer_text = str(content)
        
        timeline = []   
        if options.task == TaskType.STRUCTURE:
            timeline = parse_timeline_from_structure(answer_text)

        return VLMResult(
            raw_text=answer_text,
            timeline=timeline,
            tags=[],
            beats=[],
            extra={
                "backend": self._model,
                "task": options.task.value,
                "video_metadata": {**video.metadata, "fps": fps},
                "video_source": video_source,
                "raw_response": response,
            },
        )