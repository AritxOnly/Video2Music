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

def main():
    # 1. 加载 .env
    load_dotenv()

    parser = argparse.ArgumentParser(description="VLM 视频分析脚本")

    parser.add_argument(
        "--backend",
        type=str,
        default="qwen-web",
        help="选择后端，例如 qwen-web",
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="视频路径或视频URL",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="structure",
        choices=[t.value for t in TaskType],
        help="任务类型: qa / tagging / structure / detection",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="自定义提示词，不传则走默认 prompt",
    )

    parser.add_argument(
        "--lang",
        type=str,
        default="zh",
        help="返回语言: zh / en",
    )

    parser.add_argument(
        "--timeline",
        action="store_true",
        help="是否需要时间线输出",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="可选：将结果写入 JSON 文件",
    )

    args = parser.parse_args()

    # 2. 环境变量取 API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "找不到 DASHSCOPE_API_KEY，请在 .env 中加入：\n"
            "DASHSCOPE_API_KEY=你的key"
        )

    # 3. 构造 VLM 实例
    vlm = get_vlm(args.backend, api_key=api_key)

    # 4. 构造 VideoInput（自动判断：URL 还是路径）
    video_arg = args.video
    if video_arg.startswith("http://") or video_arg.startswith("https://"):
        video = VideoInput(url=video_arg)
    else:
        video = VideoInput(path=str(Path(video_arg).absolute()))

    # 5. 构造分析选项
    options = AnalysisOptions(
        task=TaskType(args.task),
        prompt=args.prompt,
        language=args.lang,
        need_timeline=args.timeline,
    )

    # 6. 调用模型
    print(f"使用 {args.backend} 后端执行任务 {args.task} ...")
    result = vlm.analyze(video, options)

    # 7. 处理输出
    print("\n=== 模型输出（raw_text） ===")
    print(result.raw_text)
    
    print("\n=== parse后输出 ===")
    for seg in result.timeline:
        print(seg.start_sec, seg.end_sec, seg.label, seg.description)

    if args.output:
        import json
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result.__dict__, f, ensure_ascii=False, indent=2)
        print(f"\n已写入结果到：{out_path}")

if __name__ == "__main__":
    main()