import cv2
import numpy as np

def detect_shot_changes(video_path: str, threshold: float = 0.7) -> list:
    """
    检测视频中的镜头切换点
    :param video_path: 视频文件路径
    :param threshold: 相似度阈值，用于判断镜头切换
    :return: 切换点的帧数列表
    """
    cap = cv2.VideoCapture(video_path)
    prev_hist = None
    shot_changes = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 将图像转换为灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 计算直方图
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist /= hist.sum()  # 归一化

        # 比较当前帧和前一帧的直方图差异
        if prev_hist is not None:
            # 计算直方图的相似度（使用相关系数）
            similarity = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if similarity < threshold:  # 小于阈值时认为发生了镜头切换
                shot_changes.append(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        prev_hist = hist

    cap.release()
    return shot_changes

if __name__ == "__main__":
    video_path = 'your_video.mp4'
    shot_changes = detect_shot_changes(video_path)
    print(f"镜头切换点: {shot_changes}")