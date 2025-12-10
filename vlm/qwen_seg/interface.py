import os
import json
from typing import Optional, List, Any
from pathlib import Path
from dashscope import MultiModalConversation

from vlm.service import VLMInterface
from vlm.model import *
from vlm.utils.sampler import global_sampler

class QwenVLSegmentWebInterface(VLMInterface):
    """
    Qwen-VL 分段版实现 (Keyframe-based).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen2.5-vl-72b-instruct",
    ):
        self._api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        if not self._api_key:
            raise EnvironmentError("Missing DashScope API Key")
        self._model = model

    def get_backend_name(self) -> str:
        return f"{self._model}-segment-optimized"

    def analyze(self, video: VideoInput, options: AnalysisOptions) -> VLMResult:
        if not video.path:
            raise ValueError("QwenVLSegmentWebInterface only supports local file paths.")

        # 1. 采样
        image_paths = global_sampler.sample(video.path)
        
        # 2. 构造 Payload
        content_payload = []
        for img_path in image_paths:
            content_payload.append({"image": f"file://{img_path}"})

        # 3. 构造 Prompt
        base_prompt = ""
        if options.prompt:
            base_prompt = options.prompt
        else:
            if options.task == TaskType.STRUCTURE or options.task == TaskType.TAGGING:
                base_prompt = (
                    "这是一段视频连续的关键帧截图。请分析这些画面，生成一个 JSON 对象。"
                    "你需要提取以下字段："
                    "1. mood: 视频片段的情绪（如：紧张、欢快、忧伤）。"
                    "2. activity: 画面中的主要动作或事件（简短，如：火箭发射、两人争吵）。"
                    "3. tags: 3-5个描述画面内容、光影、风格的标签列表。"
                    "4. description: 一段简短的中文剧情描述。"
                    "请直接输出 JSON，不要包含 Markdown 格式或其他废话。"
                )
            elif options.task == TaskType.DETECTION:
                 base_prompt = "请列出这些画面中出现的关键物体或人物。"
            else:
                base_prompt = "请描述这些画面的内容。"

        # 【GPT修正】更稳健的语言判定
        lang = (options.language or "").lower()
        if lang.startswith("en"):
            base_prompt += " Please output in English."
        
        content_payload.append({"text": base_prompt})

        # 4. 调用 API
        messages = [{"role": "user", "content": content_payload}]

        # 【GPT修正】透传 options 参数，保持行为一致性
        response = MultiModalConversation.call(
            api_key=self._api_key,
            model=self._model,
            messages=messages,
            max_tokens=options.max_tokens or 1024, # 默认给够 token
            temperature=options.temperature if options.temperature is not None else 0.1, # 降低随机性
        )

        # 5. 解析结果
        output = response.get("output", {}) or {}
        choices = output.get("choices", []) or []
        if not choices:
            return VLMResult(raw_text="", timeline=[], tags=[], beats=[])

        # 【GPT修正】复用稳健的解析逻辑
        msg = choices[0].get("message", {}) or {}
        content = msg.get("content")
        
        answer_text = ""
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    answer_text += part["text"]
        elif isinstance(content, dict) and "text" in content:
            answer_text = content["text"]
        else:
            answer_text = str(content)

        clean_text = answer_text.replace("```json", "").replace("```", "").strip()

        parsed_data = {}
        try:
            parsed_data = json.loads(clean_text)
        except:
            parsed_data = {"description": clean_text, "mood": "neutral", "tags": []}

        # 【GPT修正】使用更新后的 dataclass 定义
        # 注意：start/end 依然设为 0，等待 pipeline 注入绝对时间
        segment_event = TimelineEvent(
            start_sec=0.0,
            end_sec=0.0, 
            label=parsed_data.get("mood", "neutral"),
            description=parsed_data.get("description", ""),
            # 将 tags 和 activity 放入 extra
            extra={
                "tags": parsed_data.get("tags", []), 
                "activity": parsed_data.get("activity", "")
            }
        )

        return VLMResult(
            raw_text=answer_text,
            timeline=[segment_event], 
            tags=parsed_data.get("tags", []),
            beats=[],
            extra={
                "backend": self._model,
                "input_frames_count": len(image_paths),
                "raw_response": response
            }
        )