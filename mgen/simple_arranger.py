from __future__ import annotations

from typing import List, Optional, Any

from .interface import MusicArrangerInterface, MusicLibraryInterface
from .model import SegmentMusicPlan, MGenOptions, MusicTrack


def _infer_energy_from_label(label: str) -> float:
    label = (label or "").lower()
    if any(k in label for k in ["excited", "刺激", "激烈", "high", "高潮"]):
        return 0.8
    if any(k in label for k in ["calm", "平静", "温柔", "轻松"]):
        return 0.3
    if any(k in label for k in ["sad", "悲伤", "忧郁"]):
        return 0.4
    # 默认中等
    return 0.5


def _pick_best_track(
    tracks: List[MusicTrack],
    target_energy: float,
) -> Optional[MusicTrack]:
    if not tracks:
        return None
    # 选 energy 差距最小的
    return min(tracks, key=lambda t: abs(t.energy - target_energy))


def _get_event_times(ev: Any) -> tuple[float, float]:
    start = getattr(ev, "start_sec", None)
    end = getattr(ev, "end_sec", None)

    # 兜底：有的模型可能用 start/end 命名
    if start is None:
        start = getattr(ev, "start", 0.0)
    if end is None:
        end = getattr(ev, "end", 0.0)

    return float(start or 0.0), float(end or 0.0)


def _get_event_text(ev: Any) -> tuple[str, str]:
    """
    返回 (label_like, desc_like)：
    TimelineEvent: label / description
    SegmentSemantics: mood / (activity or text_summary)
    其他类型尽量从常见字段里拼。
    """
    label = getattr(ev, "label", None)
    desc = getattr(ev, "description", None)

    if label is None:
        label = getattr(ev, "mood", "")  # SegmentSemantics
    if desc is None:
        desc = getattr(ev, "activity", "") or getattr(ev, "text_summary", "")

    return str(label or ""), str(desc or "")


class SimpleRuleArranger(MusicArrangerInterface):
    """
    非生成式的规则编排器：
    - 对每个时间线事件估计 energy
    - 在指定 style 的 track 里选最接近 energy 的一首
    - 生成 SegmentMusicPlan，后续交给 ffmpeg 切/拼/淡入淡出

    timeline 既可以是 List[TimelineEvent]，也可以是 List[SegmentSemantics]。
    """

    def arrange(
        self,
        timeline: List[Any],
        library: MusicLibraryInterface,
        options: Optional[MGenOptions] = None,
    ) -> List[SegmentMusicPlan]:
        if options is None:
            options = MGenOptions()

        style = options.preferred_style
        tracks = library.list_tracks(style=style)

        plans: List[SegmentMusicPlan] = []

        for ev in timeline:
            start, end = _get_event_times(ev)
            if end <= start:
                continue

            label, desc = _get_event_text(ev)
            energy_src = f"{label} {desc}".strip()
            target_energy = _infer_energy_from_label(energy_src)

            track = _pick_best_track(tracks, target_energy)
            if track is None:
                continue

            plan = SegmentMusicPlan(
                start_sec=start,
                end_sec=end,
                track_id=track.id,
                offset_sec=0.0,                 # 先简单从头播
                fade_in_sec=options.crossfade_sec,
                fade_out_sec=options.crossfade_sec,
                gain_db=options.global_gain_db,
                metadata={
                    "event_label": label,
                    "event_desc": desc,
                    "target_energy": target_energy,
                    "track_energy": track.energy,
                    "track_style": track.style,
                },
            )
            plans.append(plan)

        return plans