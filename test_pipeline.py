from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Dict, Any

# 确保引用路径正确
from pipeline.video2music import run_video2music

def main() -> None:
    parser = argparse.ArgumentParser(description="Video2Music pipeline (Qwen-Segment backend)")

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
        help="VLM 语言标签（建议 zh）",
    )
    
    # 新增：API Key 参数
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="DashScope API Key (如果不传，将尝试读取环境变量 DASHSCOPE_API_KEY)",
    )
    
    parser.add_argument(
        "--fusion_strategy",
        type=str,
        default="ai",
        choices=["ai", "rule"],
        help="视频语义融合策略：ai或 rule，默认 ai",
    )

    args = parser.parse_args()

    # 0. 路径与Key准备
    video_path = str(Path(args.video).expanduser())
    tracks_json = str(Path(args.tracks).expanduser())
    output_path = str(Path(args.output).expanduser())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 优先使用命令行参数，其次使用环境变量
    api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY")
    
    fusion_strategy = args.fusion_strategy
    
    if not api_key:
        print("\n[Error] 必须提供 API Key 才能运行 Qwen-VL。")
        print("请使用 --api_key 参数或设置环境变量 DASHSCOPE_API_KEY。\n")
        return

    # 1. 运行 Pipeline
    # backend 参数这里其实只是个名字，因为我们在 pipeline 里已经硬编码了 QwenVLSegmentWebInterface
    try:
        result: Dict[str, Any] = run_video2music(
            backend="qwen-segment", 
            api_key=api_key,
            video_path=video_path,
            tracks_json=tracks_json,
            output_path=output_path,
            lang=args.lang,
            music_style=args.style,
            fusion_strategy=fusion_strategy,
            use_cache=True,  # 默认开启缓存
        )

        print("\n=== Pipeline Finished Successfully ===")
        print(f"Video in : {result['video_in']}")
        print(f"Video out: {result['video_out']}")
        
        semantics = result['semantics']
        print(f"Segments : {len(semantics.segments)} (Semantics extracted)")
        
        # 打印一下具体的语义片段，方便你第一时间确认效果
        for i, seg in enumerate(semantics.segments):
            print(f"  [{i}] {seg.start_sec:.1f}s-{seg.end_sec:.1f}s | Mood: {seg.mood} | {seg.text_summary[:15]}...")
            
        print(f"Plans    : {len(result['plans'])} music cues generated.")
        
    except Exception as e:
        print(f"\n[Fatal Error] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()