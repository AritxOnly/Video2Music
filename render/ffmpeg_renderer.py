from typing import List
from mgen.model import SegmentMusicPlan, MusicTrack

def render_with_bgm(
    video_path: str,
    plans: List[SegmentMusicPlan],
    track_map: dict[str, MusicTrack],
    output_path: str,
) -> None:
    """
    按照 plans 从 track_map 里的 mp3/wav 切片、淡入淡出、混到 video 上。
    内部用 ffmpeg subprocess 搞定。
    """
    ...