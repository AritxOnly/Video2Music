import math
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any

# 假设你的 MusicTrack 是个 namedtuple 或简单对象，或者直接用字典
# 这里为了兼容之前的 pipeline，我们假设 track_map 的 value 是一个有 filepath 属性的对象
# 如果没有 model 定义，我们可以容错处理

def _db_to_linear(gain_db: float) -> float:
    """把 dB 转线性倍率，ffmpeg 的 volume 需要这个。"""
    return float(10 ** (gain_db / 20.0))

def render_with_bgm(
    video_path: str,
    plans: List[Dict[str, Any]], # 这里改用 Dict 以匹配 Arranger 的输出
    track_map: Dict[str, Any],   # value 包含 filepath 即可
    output_path: str,
    keep_original_audio: bool = False # 新增：是否保留原视频声音
) -> None:
    """
    按照 plans 从 track_map 里的 mp3/wav 切片、淡入淡出、混到 video 上。
    """
    video_path = str(Path(video_path).expanduser().absolute())
    output_path = str(Path(output_path).expanduser().absolute())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 如果没有计划，直接 copy 原视频
    if not plans:
        shutil.copyfile(video_path, output_path)
        return

    # 确保按时间顺序
    plans = sorted(plans, key=lambda p: p.get('start_time', 0.0))

    # 构建 ffmpeg 命令
    # -y 覆盖输出
    cmd: List[str] = ["ffmpeg", "-y", "-i", video_path]

    # === Step 1: 构造输入列表 ===
    # 输入 0 是视频
    # 输入 1..N 是 BGM
    
    valid_plans = []
    
    for plan in plans:
        track_id = plan.get('track_id')
        track = track_map.get(track_id)
        
        if track is None:
            continue

        # 兼容不同类型的 track 对象 (dict 或 object)
        audio_path = getattr(track, 'filepath', None) or track.get('filepath')
        if not audio_path:
            continue
            
        audio_path = str(Path(audio_path).expanduser().absolute())
        
        # 计算需要播放的时长
        start_sec = plan.get('start_time', 0.0)
        end_sec = plan.get('end_time', 0.0)
        dur = max(0.0, end_sec - start_sec)
        
        if dur <= 0:
            continue
            
        # 获取音乐内部的起始点 (例如从副歌开始)
        source_start = plan.get('source_start', 0.0)

        # 记录有效 plan，供后面 filter 使用
        valid_plans.append({
            "plan": plan,
            "dur": dur,
            "input_index": len(valid_plans) + 1 # 视频是0，所以从1开始
        })

        # 【关键修改】使用 -ss 定位到音乐内部的开始位置 (source_start)
        # -t 限制读取的时长 (dur)
        cmd.extend([
            "-ss", f"{source_start:.3f}",
            "-t", f"{dur:.3f}",
            "-i", audio_path,
        ])

    if not valid_plans:
        shutil.copyfile(video_path, output_path)
        return

    # === Step 2: 构造 Filter Complex ===
    filter_parts: List[str] = []
    bgm_output_labels = [] 

    for item in valid_plans:
        idx = item['input_index']
        plan = item['plan']
        dur = item['dur']
        
        # 输入标签
        in_label = f"[{idx}:a]"
        
        # 参数准备
        gain_db = plan.get('volume_db', -6.0)
        volume = _db_to_linear(gain_db)
        
        # 淡入淡出计算
        fade_in = plan.get('fade_in', 0.5)
        fade_out = plan.get('fade_out', 0.5)
        
        # 安全检查：fade 不能超过时长的一半
        fade_in = max(0.0, min(fade_in, dur / 2.0))
        fade_out = max(0.0, min(fade_out, dur / 2.0))
        fade_out_start = max(0.0, dur - fade_out)

        # 标签命名
        tag_processed = f"a{idx}_proc"
        tag_delayed = f"a{idx}_final"

        # 1. 音量 + 淡入淡出
        chain_process = (
            f"{in_label}"
            f"volume={volume:.4f},"
            f"afade=t=in:st=0:d={fade_in:.3f},"
            f"afade=t=out:st={fade_out_start:.3f}:d={fade_out:.3f}"
            f"[{tag_processed}]"
        )
        filter_parts.append(chain_process)

        # 2. 延迟 (Adelay) - 把音乐推到视频的时间轴位置
        # adelay 需要毫秒，格式 "L|R"
        delay_ms = int(plan.get('start_time', 0.0) * 1000)
        chain_delay = f"[{tag_processed}]adelay={delay_ms}|{delay_ms}[{tag_delayed}]"
        filter_parts.append(chain_delay)

        bgm_output_labels.append(f"[{tag_delayed}]")

    # === Step 3: 混音 ===
    # 将所有处理好的 BGM 混在一起
    mix_bgm_label = "[bgm_mix_all]"
    filter_parts.append(
        f"{''.join(bgm_output_labels)}"
        f"amix=inputs={len(bgm_output_labels)}:normalize=0{mix_bgm_label}"
    )
    
    final_audio_label = mix_bgm_label

    # (可选) 如果要保留原声
    if keep_original_audio:
        final_audio_label = "[final_mix]"
        # 将原视频音频 [0:a] 和 BGM [bgm_mix_all] 混合
        # inputs=2, duration=first (以视频长度为准)
        filter_parts.append(
            f"[0:a]{mix_bgm_label}amix=inputs=2:duration=first:normalize=0{final_audio_label}"
        )

    # === Step 4: 组装命令 ===
    filter_complex_str = ";".join(filter_parts)
    
    cmd.extend([
        "-filter_complex", filter_complex_str,
        "-map", "0:v",              # 使用原视频画面
        "-map", final_audio_label,  # 使用处理后的音频
        "-c:v", "copy",             # 视频流直接复制，不转码 (快！)
        "-c:a", "aac",              # 音频重编码
        "-shortest",                # 以最短流为准 (通常是视频)
        output_path,
    ])

    print(f"    [Render] Executing FFmpeg with {len(valid_plans)} audio tracks...")
    # print(" ".join(cmd)) # Debug用，打印完整命令

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f"    [Error] FFmpeg failed with code {e.returncode}")
        raise e