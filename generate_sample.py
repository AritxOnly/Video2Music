from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from vlm import (
    get_vlm,
    VideoInput,
    AnalysisOptions,
    TaskType,
)
from vlm.model import vlm_result_to_dict


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="生成 vlm/test/sample.json 用的脚本")

    parser.add_argument(
        "--backend",
        type=str,
        default="qwen-web",
        help="真实 VLM 后端，例如 qwen-web",
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="视频路径或 URL",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="zh",
        help="返回语言: zh / en",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="vlm/test/sample.json",
        help="sample.json 输出路径",
    )

    args = parser.parse_args()

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "找不到 DASHSCOPE_API_KEY，请在 .env 中加入：\n"
            "DASHSCOPE_API_KEY=你的key"
        )

    vlm = get_vlm(args.backend, api_key=api_key)

    video_arg = args.video
    if video_arg.startswith("http://") or video_arg.startswith("https://"):
        video = VideoInput(url=video_arg)
    else:
        video = VideoInput(path=str(Path(video_arg).absolute()))

    # 1) structure（一定要开 need_timeline）
    struct_opts = AnalysisOptions(
        task=TaskType.STRUCTURE,
        prompt=None,
        language=args.lang,
        need_timeline=True,
    )
    print(f"使用 {args.backend} 执行 STRUCTURE ...")
    struct_res = vlm.analyze(video, struct_opts)

    # 2) tagging
    tag_opts = AnalysisOptions(
        task=TaskType.TAGGING,
        prompt=None,
        language=args.lang,
        need_timeline=False,
    )
    print(f"使用 {args.backend} 执行 TAGGING ...")
    tag_res = vlm.analyze(video, tag_opts)

    # 3) 打包写入 sample.json
    sample = {
        "structure": vlm_result_to_dict(struct_res),
        "tagging": vlm_result_to_dict(tag_res),
    }

    import json

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)

    print(f"\n已写入 sample.json 到：{out_path}")


if __name__ == "__main__":
    main()