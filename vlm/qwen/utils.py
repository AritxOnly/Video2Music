import json
from typing import List
from vlm.model import TimelineEvent

def parse_timeline_from_structure(raw: str) -> List[TimelineEvent]:
    # 去掉 ```json ... ``` 包裹
    raw = raw.strip()
    if raw.startswith("```"):
        # ```json\n...\n``` 这种
        raw = raw.strip("`")
        # 简单找第一个 { 和最后一个 }
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start:end+1]

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return []

    segs = obj.get("segments") or []
    timeline: List[TimelineEvent] = []

    for seg in segs:
        try:
            start = float(seg.get("start_time", 0.0))
            end = float(seg.get("end_time", 0.0))
            label = seg.get("mood", "") or "segment"
            desc = seg.get("description", "") or ""
            timeline.append(
                TimelineEvent(
                    start_sec=start,
                    end_sec=end,
                    label=label,
                    description=desc,
                )
            )
        except Exception:
            continue

    return timeline