import subprocess
import cv2

def cut_video(video_path: str, shot_changes: list, output_dir: str) -> None:
    """
    根据镜头切换点切割视频并保存为单独的文件
    :param video_path: 视频文件路径
    :param shot_changes: 镜头切换点的帧数
    :param output_dir: 输出目录
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    for i, start_frame in enumerate(shot_changes):
        # 计算时间戳（单位：秒）
        start_time = start_frame / fps
        end_time = shot_changes[i + 1] / fps if i + 1 < len(shot_changes) else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        
        # 使用ffmpeg切割视频
        output_path = f"{output_dir}/clip_{i}.mp4"
        command = [
            'ffmpeg',
            '-i', video_path,
            '-ss', str(start_time),
            '-to', str(end_time),
            '-c:v', 'libx264', '-c:a', 'aac', '-strict', 'experimental',
            '-y', output_path
        ]
        
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            # 这里不打印标准输出
            error_message = result.stderr.decode('utf-8')
            if error_message:
                print(f"ffmpeg 错误: {error_message}")
        except subprocess.CalledProcessError as e:
            error_message = e.stderr.decode('utf-8')
            if error_message:
                print(f"ffmpeg 错误: {error_message}")

    cap.release()