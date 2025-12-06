from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Dict, Any

from pipeline.video2music import run_video2music


def main() -> None:
    parser = argparse.ArgumentParser(description="Video2Music pipeline (sample backend)")

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="输入视频路径（用于 ffmpeg 合成画面）",
    )

    parser.add_argument(
        "--tracks",
        type=str,
        default="mgen/tracks.example.json",
        help="音乐库 JSON 路径（默认 mgen/tracks.example.json）",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/sample_with_bgm.mp4",
        help="输出视频路径（默认 outputs/sample_with_bgm.mp4）",
    )

    parser.add_argument(
        "--style",
        type=str,
        default=None,
        help="可选：指定音乐风格，如 lofi / edm / cinematic，对应 tracks.json 里的 style 字段",
    )

    parser.add_argument(
        "--lang",
        type=str,
        default="zh",
        help="VLM 语言标签（对 sample backend 其实无所谓，默认 zh）",
    )

    args = parser.parse_args()

    video_path = str(Path(args.video).expanduser())
    tracks_json = str(Path(args.tracks).expanduser())
    output_path = str(Path(args.output).expanduser())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # sample backend 不需要 api_key，这里传空字符串即可
    result: Dict[str, Any] = run_video2music(
        backend="sample",
        api_key="",
        video_path=video_path,
        tracks_json=tracks_json,
        output_path=output_path,
        lang=args.lang,
        music_style=args.style,
    )

    print("=== Pipeline Finished ===")
    print("Video in :", result["video_in"])
    print("Video out:", result["video_out"])
    print(f"Segments : {len(result['semantics'].segments)}")
    print(f"Plans    : {len(result['plans'])}")


if __name__ == "__main__":
    main()