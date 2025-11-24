from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

from vlm import (
    get_vlm,
    VideoInput,
    AnalysisOptions,
    TaskType,
)
from vsem.builder import build_from_vlm
from mgen import (
    JsonMusicLibrary,
    SimpleRuleArranger,
    MGenOptions,
)
from render.ffmpeg_renderer import render_with_bgm


def run_video2music(
    backend: str,
    api_key: str,
    video_path: str,
    tracks_json: str,
    output_path: str,
    lang: str = "zh",
    music_style: Optional[str] = None,
) -> Dict[str, Any]:
    """
    最简单的一条龙：
    video → VLM(结构+标签) → VideoSemantics → mgen 编排 → ffmpeg 合成
    """

    # 0. 绝对路径
    video_abs = str(Path(video_path).expanduser().absolute())
    tracks_json_abs = str(Path(tracks_json).expanduser().absolute())
    output_abs = str(Path(output_path).expanduser().absolute())

    # 1. VLM：结构 + 标签
    vlm = get_vlm(backend, api_key=api_key)

    vinput = VideoInput(path=video_abs)

    struct_res = vlm.analyze(
        vinput,
        AnalysisOptions(
            task=TaskType.STRUCTURE,
            language=lang,
            need_timeline=True,
        ),
    )

    tag_res = vlm.analyze(
        vinput,
        AnalysisOptions(
            task=TaskType.TAGGING,
            language=lang,
        ),
    )

    # 2. 语义汇总：结构 + 标签 → VideoSemantics
    semantics = build_from_vlm(struct_res, tag_res)

    # 3. 音乐库 + 简单编排
    lib = JsonMusicLibrary(tracks_json_abs)
    arranger = SimpleRuleArranger()

    mopts = MGenOptions(
        preferred_style=music_style,
        # 这些先用默认，后面你可以在 CLI 里暴露
        global_gain_db=-6.0,
        crossfade_sec=0.3,
    )

    plans = arranger.arrange(
        timeline=semantics.segments,
        library=lib,
        options=mopts,
    )

    # 4. 渲染：按计划把 BGM 混到视频上
    track_map = {t.id: t for t in lib.list_tracks(style=music_style) or lib.list_tracks()}

    render_with_bgm(
        video_path=video_abs,
        plans=plans,
        track_map=track_map,
        output_path=output_abs,
    )

    return {
        "semantics": semantics,
        "plans": plans,
        "video_in": video_abs,
        "video_out": output_abs,
    }