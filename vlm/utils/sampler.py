import cv2
import os
import shutil
from pathlib import Path
from typing import List
import numpy as np
import uuid

class VideoKeyframeSampler:
    """
    视频关键帧提取器。
    策略：
    - < 2s: 提取中间 1 帧
    - 2s - 10s: 提取 首、中、尾 3 帧
    - > 10s: 每 3 秒提取 1 帧 (或根据 Token 预算调整)
    """

    def __init__(self, temp_dir: str = ".temp_frames", max_frames: int = 12):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.max_frames = max_frames

    def sample(self, video_path: str) -> List[str]:
        """
        输入视频路径，返回提取出的图片绝对路径列表。
        """
        path_obj = Path(video_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(path_obj))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        # === 策略核心 ===
        target_indices = []
        
        if duration < 2.0:
            # 极短片段：取中间
            target_indices = [frame_count // 2]
        elif duration < 10.0:
            # 短片段：首、中、尾
            target_indices = [
                0, 
                frame_count // 2, 
                max(0, frame_count - 1)
            ]
        else:
            # 长片段：每 3 秒一帧 (降采样，既保证覆盖又不浪费Token)
            step = int(fps * 3) 
            target_indices = list(range(0, frame_count, step))
            # 确保最后一帧被包含，防止漏掉结尾动作
            if target_indices[-1] < frame_count - 1:
                target_indices.append(frame_count - 1)
                
        if len(target_indices) > self.max_frames:
            target_indices = np.linspace(0, frame_count - 1, self.max_frames, dtype=int).tolist()
            target_indices = sorted(set(target_indices))

        # === 执行抽取 ===
        image_paths = []
        # 为当前任务创建一个唯一的子目录，避免冲突
        task_id = str(uuid.uuid4())[:8]
        save_dir = self.temp_dir / task_id
        save_dir.mkdir(exist_ok=True)

        for i, idx in enumerate(target_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # 命名格式: 001.jpg
                file_name = f"{i:03d}.jpg"
                save_path = save_dir / file_name
                # 压缩一下质量，节省传输带宽和 Token (DashScope会自动处理分辨率，但jpg压缩有好处)
                cv2.imwrite(str(save_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                image_paths.append(str(save_path.absolute()))
        
        cap.release()
        return image_paths

    def cleanup(self):
        """清理临时文件"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

# 全局单例方便调用，或者在 Interface 里实例化
global_sampler = VideoKeyframeSampler()