import json
from typing import List
from vlm.model import VLMResult, TimelineEvent
from .model import VideoSemantics, SegmentSemantics

# TODO 
def build_from_vlm(
    structure_result: VLMResult,
    tagging_result: VLMResult | None = None,
) -> VideoSemantics:
    # 1. 解析 structure_result.raw_text 里的 JSON → segments
    #    已有 TimelineEvent 也可以作为输入
    segments: List[SegmentSemantics] = []

    # 这里略掉具体 parse，逻辑大概是：
    # - 每个 JSON segment → SegmentSemantics
    # - mood 放到 mood
    # - description 放到 text_summary / activity

    # 2. Tagging 的 raw_text：拆成 tag 列表
    global_tags: List[str] = []
    if tagging_result is not None:
        for line in tagging_result.raw_text.splitlines():
            t = line.strip("-• \t")
            if t:
                global_tags.append(t)

    return VideoSemantics(
        global_tags=global_tags,
        segments=segments,
        extra={
            "raw_structure": structure_result.raw_text,
            "raw_tags": tagging_result.raw_text if tagging_result else "",
        },
    )