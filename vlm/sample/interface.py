from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

from vlm.model import *
from vlm.interface import VLMInterface


class SampleVLMOutputInterface(VLMInterface):
    """
    一个“假” VLM：不真正调用模型，而是读取 vlm/test/sample.json，
    按 VLMResult 结构返回，方便本地调试 pipeline。
    """

    def __init__(self, json_path: str | Path | None = None, **kwargs) -> None:
        if json_path is None:
            # 默认指向 vlm/test/sample.json
            base_dir = Path(__file__).resolve().parent
            json_path = base_dir / "index.json"
        self._json_path = Path(json_path)

    def get_backend_name(self) -> str:
        return "sample"

    def analyze(self, video: VideoInput, options: AnalysisOptions) -> VLMResult:
        with open(self._json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 如果 sample.json 里按任务类型分层（如 {"structure": {...}, "tagging": {...}}），
        # 则优先取当前 task 对应的那一层。
        if isinstance(data, dict) and options.task.value in data:
            data = data[options.task.value]

        # 下面把 JSON 映射到 VLMResult
        raw_text: str = data.get("raw_text", "")

        # timeline
        timeline: List[TimelineEvent] = []
        for item in data.get("timeline", []):
            if not isinstance(item, dict):
                continue
            start = item.get("start_sec", item.get("start", 0.0))
            end = item.get("end_sec", item.get("end", 0.0))
            label = item.get("label", "")
            desc = item.get("description", "")
            timeline.append(
                TimelineEvent(
                    start_sec=float(start),
                    end_sec=float(end),
                    label=str(label),
                    description=str(desc),
                )
            )

        # tags
        tags = data.get("tags", [])
        if isinstance(tags, str):
            # 兼容 "tag1, tag2" 这种写法
            tags = [t.strip() for t in tags.replace("，", ",").split(",") if t.strip()]

        # beats（可选）
        beats: List[BeatInfo] = []
        for b in data.get("beats", []):
            if not isinstance(b, dict):
                continue
            t = b.get("time_sec", b.get("time", 0.0))
            s = b.get("strength", 1.0)
            beats.append(BeatInfo(time_sec=float(t), strength=float(s)))

        # 其余字段丢到 extra 里
        extra_keys = {"raw_text", "timeline", "tags", "beats"}
        extra: Dict[str, Any] = {k: v for k, v in data.items() if k not in extra_keys}

        return VLMResult(
            raw_text=raw_text,
            timeline=timeline,
            tags=list(tags),
            beats=beats,
            extra=extra,
        )