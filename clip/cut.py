import subprocess
import cv2
import os

def cut_video(video_path: str, shot_changes: list, output_dir: str) -> None:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # 确保是 int
    shot_changes = sorted(int(f) for f in shot_changes)

    # 构造完整边界：开头0 + 所有切换点 + 结尾
    boundaries = [0] + shot_changes + [int(total_frames)]

    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(boundaries) - 1):
        start_frame = boundaries[i]
        end_frame = boundaries[i + 1]

        # 转成秒
        start_time = start_frame / fps
        end_time = end_frame / fps

        output_path = f"{output_dir}/clip_{i}.mp4"
        command = [
            'ffmpeg',
            '-i', video_path,
            '-ss', str(start_time),
            '-to', str(end_time),   # 绝对时间：相对整条视频的时间
            '-c:v', 'libx264', '-c:a', 'aac', '-strict', 'experimental',
            '-y', output_path
        ]

        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            err = result.stderr.decode('utf-8')
            if err:
                print(f"ffmpeg 错误: {err}")
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode('utf-8')
            if err:
                print(f"ffmpeg 错误: {err}")

    cap.release()