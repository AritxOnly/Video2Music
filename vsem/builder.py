import json
from typing import List, Any

from vlm.model import VLMResult, TimelineEvent
from .model import VideoSemantics, SegmentSemantics


def _from_timeline_event(ev: TimelineEvent) -> SegmentSemantics:
    mood = (ev.label or "").strip() if hasattr(ev, "label") else ""
    # 尝试从 description 抽一条简短 activity
    desc = (ev.description or "").strip() if hasattr(ev, "description") else ""
    activity = desc
    if "，" in activity or "。" in activity:
        # 只取第一句 / 第一小段
        activity = activity.replace("。", "。|").replace("，", "，|").split("|")[0]

    return SegmentSemantics(
        start_sec=float(getattr(ev, "start_sec", 0.0)),
        end_sec=float(getattr(ev, "end_sec", 0.0)),
        mood=mood,
        activity=activity or mood,
        tags=[],
        text_summary=desc or mood,
        extra={
            "timeline_label": mood,
            "timeline_desc": desc,
        },
    )


def _from_raw_json(raw_text: str) -> List[SegmentSemantics]:
    """
    兜底：如果 timeline 为空，再尝试从 raw_text 里 parse JSON。
    兼容几种常见字段名：start/start_sec、end/end_sec、label/mood/description/tags...
    """
    segments: List[SegmentSemantics] = []
    if not raw_text:
        return segments

    try:
        data: Any = json.loads(raw_text)
    except Exception:
        return segments

    if isinstance(data, dict) and "segments" in data:
        raw_segments = data["segments"]
    elif isinstance(data, list):
        raw_segments = data
    else:
        return segments

    for item in raw_segments:
        if not isinstance(item, dict):
            continue

        start = float(
            item.get("start_sec")
            or item.get("start")
            or 0.0
        )
        end = float(
            item.get("end_sec")
            or item.get("end")
            or 0.0
        )
        if end <= start:
            continue

        mood = (
            item.get("mood")
            or item.get("label")
            or ""
        )
        desc = (
            item.get("description")
            or item.get("summary")
            or ""
        )
        activity = (
            item.get("activity")
            or item.get("action")
            or desc
            or mood
        )

        tags = item.get("tags") or []
        if isinstance(tags, str):
            # "tag1, tag2, tag3" 这种
            tags = [
                t.strip()
                for t in tags.replace("，", ",").split(",")
                if t.strip()
            ]

        seg = SegmentSemantics(
            start_sec=start,
            end_sec=end,
            mood=str(mood),
            activity=str(activity),
            tags=list(tags),
            text_summary=str(desc or activity or mood),
            extra={"raw": item},
        )
        segments.append(seg)

    return segments


def build_from_vlm(
    structure_result: VLMResult,
    tagging_result: VLMResult | None = None,
) -> VideoSemantics:
    """
    统一入口：
    1. 优先用 structure_result.timeline 里的 TimelineEvent
    2. 若 timeline 为空，再尝试从 raw_text 解析 JSON
    3. tagging_result.raw_text 按行拆成 global_tags
    """
    segments: List[SegmentSemantics] = []

    # 1) 优先使用结构化时间线
    timeline = getattr(structure_result, "timeline", None) or []
    for ev in timeline:
        segments.append(_from_timeline_event(ev))

    # 2) 如果 timeline 为空，再从 raw_text JSON 兜底
    if not segments:
        segments = _from_raw_json(structure_result.raw_text or "")

    # 3) Tagging 的 raw_text：按行拆 tag
    global_tags: List[str] = []
    raw_tags_text = ""
    if tagging_result is not None and tagging_result.raw_text:
        raw_tags_text = tagging_result.raw_text
        for line in tagging_result.raw_text.splitlines():
            t = line.strip("-• \t")
            if t:
                global_tags.append(t)

    return VideoSemantics(
        global_tags=global_tags,
        segments=segments,
        extra={
            "raw_structure": structure_result.raw_text or "",
            "raw_tags": raw_tags_text,
        },
    )