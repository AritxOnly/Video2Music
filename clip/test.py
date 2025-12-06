from cut import cut_video
from detect import detect_shot_changes
from argparse import ArgumentParser

def main():
    arg_parser = ArgumentParser(description="视频镜头切割脚本")
    arg_parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="输入视频文件路径",
    )
    arg_parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录，用于保存切割后的视频片段",
    )
    video_path = arg_parser.parse_args().video
    output_dir = arg_parser.parse_args().output_dir
    
    change_points = detect_shot_changes(video_path=video_path)
    
    cut_video(video_path=video_path, shot_changes=change_points, output_dir=output_dir)
    
    print(f"检测到的镜头切换点（帧数）: {change_points}")

if __name__ == '__main__':
    main()