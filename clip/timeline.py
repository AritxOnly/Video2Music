def generate_timeline(shot_changes: list, fps: float) -> list:
    """
    根据镜头切换点生成时间线。
    :param shot_changes: 镜头切换点的帧数列表
    :param fps: 视频的帧率
    :return: 时间线列表，每个元素是一个字典（包含开始时间、结束时间和标签）
    """
    timeline = []
    
    # 遍历镜头切换点
    for i in range(len(shot_changes) - 1):
        start_frame = shot_changes[i]
        end_frame = shot_changes[i + 1]
        
        # 计算每个镜头的起始和结束时间
        start_time = start_frame / fps
        end_time = end_frame / fps
        
        # 在这里添加标签或者描述
        filename = f"clip_{i}.mp4"
        label = f"镜头 {i + 1}"
        description = f"描述 {i + 1}，镜头内容分析"
        
        # 构建时间线事件
        event = {
            "start_sec": start_time,
            "end_sec": end_time,
            "filename": filename,
            "label": label,
            "description": description
        }
        timeline.append(event)
    
    return timeline