import os
from typing import Any

from openai import OpenAI
from pathlib import Path

from vlm.service import VLMInterface
from vlm.model import *


class QwenVLWebInterface(VLMInterface):
    def __init__(
        self,
        api_key: str | None = None,
        model: str = 'qwen3-vl-plus',
        base_url: str = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    ):
        self._client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url=base_url
        )
        self._model = model
    
    def get_backend_name(self) -> str:
        return self._model

    def analyze(self, video: VideoInput, options: AnalysisOptions) -> VLMResult:
        """
        使用阿里云 DashScope 的 OpenAI 兼容接口调用 Qwen-VL 模型。

        支持：
        - video.url: 远程视频地址
        - video.path: 本地视频路径，会转成 file:// 形式传给 Qwen

        不做的事：
        - 不在这里解析时间线 / 标签 / 节拍，只返回 raw_text 和 extra，
          timeline/tags/beats 由上层 Agent 在需要时从 raw_text 里二次解析。
        """

        # 1. 解析视频输入 → 统一成一个 media_url
        media_url: str | None = None

        if video.url:
            media_url = video.url
        elif video.path:
            abs_path = Path(video.path).expanduser().absolute()
            media_url = f"file://{abs_path}"
        else:
            raise ValueError("VideoInput 必须至少提供 url 或 path 之一。")
        
        fps = int(video.fps or 1)

        # 2. 构造任务 prompt
        if options.prompt is not None:
            base_prompt = options.prompt
        else:
            if options.task == TaskType.CAPTION:
                base_prompt = "Generate a detailed caption for this video."
            elif options.task == TaskType.QA:
                base_prompt = "Answer questions about the content of this video."
            elif options.task == TaskType.TAGGING:
                base_prompt = (
                    "Generate concise tags describing the content and style of this video."
                )
            elif options.task == TaskType.STRUCTURE:
                base_prompt = (
                    "Analyze the high-level structure of this video. "
                    "Identify segments such as intro, hook, verse, chorus, drop, and outro."
                )
            elif options.task == TaskType.DETECTION:
                base_prompt = "Describe important objects and events that appear in this video."
            else:
                base_prompt = "Analyze the content of this video."

        # 语言控制
        lang = (options.language or "").lower()
        if lang.startswith("zh"):
            base_prompt += " 请用中文回答。"
        elif lang.startswith("en"):
            base_prompt += " Please answer in English."

        # 时间线提示
        if options.need_timeline:
            base_prompt += (
                " If possible, also roughly indicate the time ranges of key segments in seconds."
            )

        user_content: list[dict[str, Any]] = [
            {'text': base_prompt},
            {'video': media_url, "fps": fps},
        ]

        messages = [
            {
                "role": "user",
                "content": user_content,
            }
        ]

        # 4. 调用接口
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=options.max_tokens,
            temperature=options.temperature,
        )

        # 5. 解析返回内容（兼容多种 content 形式）
        msg = completion.choices[0].message
        content = msg.content

        answer_text: str

        # 兼容：Qwen 可能直接返回 string，也可能是 list[{"type": "text", "text": "..."}]
        if isinstance(content, str):
            answer_text = content
        elif isinstance(content, list):
            texts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    t = part.get("text")
                else:
                    # openai-python v1 里可能是对象，有 .text 属性
                    t = getattr(part, "text", None)
                if t:
                    texts.append(t)
            answer_text = "\n".join(texts) if texts else ""
        else:
            # 最兜底
            answer_text = str(content)

        # 6. 封装为统一结果
        return VLMResult(
            raw_text=answer_text,
            timeline=[],
            tags=[],
            beats=[],
            extra={
                "backend": self._model,
                "task": options.task.value,
                "usage": getattr(completion, "usage", None),
                "video_metadata": video.metadata,
                "media_url": media_url,
            },
        )