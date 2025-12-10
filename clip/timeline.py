def generate_timeline(shot_changes: list, fps: float, total_frames: int) -> list:
    """
    根据镜头切换点生成时间线。
    :param shot_changes: 镜头切换点的帧数列表（不含0，不含最后一帧）
    :param fps: 视频的帧率
    :param total_frames: 视频总帧数
    :return: 时间线列表，每个元素是一个字典（包含开始时间、结束时间和标签）
    """
    timeline = []

    # 构造完整边界：0 + 所有切换点 + 总帧数
    shot_changes = sorted(int(f) for f in shot_changes)
    boundaries = [0] + shot_changes + [int(total_frames)]

    # 相邻边界成段：[b0,b1], [b1,b2], ...
    for i in range(len(boundaries) - 1):
        start_frame = boundaries[i]
        end_frame = boundaries[i + 1]

        start_time = start_frame / fps
        end_time = end_frame / fps

        filename = f"clip_{i}.mp4"          # 对齐 cut_video 里的命名
        label = f"镜头 {i + 1}"
        description = f"描述 {i + 1}，镜头内容分析"

        event = {
            "start_sec": start_time,
            "end_sec": end_time,
            "filename": filename,
            "label": label,
            "description": description
        }
        timeline.append(event)

    return timeline