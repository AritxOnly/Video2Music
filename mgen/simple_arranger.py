from __future__ import annotations

from typing import List, Optional

from vlm.model import TimelineEvent
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
    best = min(tracks, key=lambda t: abs(t.energy - target_energy))
    return best


class SimpleRuleArranger(MusicArrangerInterface):
    """
    非生成式的规则编排器：
    - 对每个 TimelineEvent 估计 energy
    - 在指定 style 的 track 里选最接近 energy 的一首
    - 生成 SegmentMusicPlan，后续交给 ffmpeg 切/拼/淡入淡出
    """

    def arrange(
        self,
        timeline: List[TimelineEvent],
        library: MusicLibraryInterface,
        options: Optional[MGenOptions] = None,
    ) -> List[SegmentMusicPlan]:
        if options is None:
            options = MGenOptions()

        style = options.preferred_style
        tracks = library.list_tracks(style=style)

        plans: List[SegmentMusicPlan] = []

        for ev in timeline:
            target_energy = _infer_energy_from_label(ev.label or ev.description)
            track = _pick_best_track(tracks, target_energy)
            if track is None:
                continue

            start = ev.start_sec
            end = ev.end_sec
            if end <= start:
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
                    "event_label": ev.label,
                    "event_desc": ev.description,
                    "target_energy": target_energy,
                    "track_energy": track.energy,
                    "track_style": track.style,
                },
            )
            plans.append(plan)

        return plans