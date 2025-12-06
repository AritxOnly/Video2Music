import math
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict

from mgen.model import SegmentMusicPlan, MusicTrack


def _db_to_linear(gain_db: float) -> float:
    """把 dB 转线性倍率，ffmpeg 的 volume 需要这个。"""
    return float(10 ** (gain_db / 20.0))


def render_with_bgm(
    video_path: str,
    plans: List[SegmentMusicPlan],
    track_map: Dict[str, MusicTrack],
    output_path: str,
) -> None:
    """
    按照 plans 从 track_map 里的 mp3/wav 切片、淡入淡出、混到 video 上。
    实现策略（单次 ffmpeg 调用）：
    - 输入 0: 原视频（含原声）
    - 输入 1..N: 各个 BGM 音频文件
    - 对每个 BGM:
      * -ss / -t 先在读入时从 offset_sec 开始截出对应长度
      * volume 调整到 gain_db
      * afade in/out
      * adelay 把片段推到 start_sec 所在的时间线上
    - 所有 BGM 轨用 amix 叠加，再与原视频的音频混合（这里默认覆盖原声，想保留原声可以再做一次 amix）
    """
    video_path = str(Path(video_path).expanduser().absolute())
    output_path = str(Path(output_path).expanduser().absolute())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 如果没有计划，直接 copy 原视频
    if not plans:
        shutil.copyfile(video_path, output_path)
        return

    # 确保按时间顺序
    plans = sorted(plans, key=lambda p: p.start_sec)

    # 构建 ffmpeg 命令
    cmd: List[str] = ["ffmpeg", "-y", "-i", video_path]

    # 先把所有 BGM 作为额外输入塞进去
    # 我们这里按 SegmentMusicPlan，一个 plan 一个音频输入
    for plan in plans:
        track = track_map.get(plan.track_id)
        if track is None:
            # 找不到就跳过
            continue

        audio_path = str(Path(track.path).expanduser().absolute())
        dur = max(0.0, plan.end_sec - plan.start_sec)
        if dur <= 0:
            continue

        # 对每个输入使用 -ss/-t，只保留需要的那一段
        # 注意：这里把 -ss 放在 -i 前面是“输入级别的 seek”，速度更快
        cmd.extend([
            "-ss", f"{max(plan.offset_sec, 0.0):.3f}",
            "-t", f"{dur:.3f}",
            "-i", audio_path,
        ])

    # 构造 filter_complex
    filter_parts: List[str] = []
    bgm_input_indices = []  # 记录真正加入 filter 的输入索引

    # 输入 0 是视频；音频输入从 1 开始
    ff_idx = 1
    for plan in plans:
        track = track_map.get(plan.track_id)
        if track is None:
            continue

        dur = max(0.0, plan.end_sec - plan.start_sec)
        if dur <= 0:
            continue

        in_label = f"[{ff_idx}:a]"
        ff_idx += 1

        # 线性音量
        volume = _db_to_linear(plan.gain_db)

        # 防止 fade 长度超过片段本身
        fade_in = max(0.0, min(plan.fade_in_sec, dur / 2.0))
        fade_out = max(0.0, min(plan.fade_out_sec, dur / 2.0))

        # 淡出开始时间 = 片段长度 - fade_out
        fade_out_start = max(0.0, dur - fade_out)

        tag_vol = f"a{ff_idx}_vol"
        tag_fi = f"a{ff_idx}_fi"
        tag_fo = f"a{ff_idx}_fo"
        tag_delayed = f"a{ff_idx}_d"

        # volume + fade in + fade out
        chain = (
            f"{in_label}"
            f"volume={volume:.4f},"
            f"afade=t=in:st=0:d={fade_in:.3f},"
            f"afade=t=out:st={fade_out_start:.3f}:d={fade_out:.3f}"
            f"[{tag_fo}]"
        )
        filter_parts.append(chain)

        # adelay 需要毫秒，且对每个声道都指定
        delay_ms = int(plan.start_sec * 1000)
        chain_delay = f"[{tag_fo}]adelay={delay_ms}|{delay_ms}[{tag_delayed}]"
        filter_parts.append(chain_delay)

        bgm_input_indices.append(f"[{tag_delayed}]")

    if not bgm_input_indices:
        # 所有 plan 都被 skip 了，就直接复制视频
        shutil.copyfile(video_path, output_path)
        return

    # 把所有延迟后的 BGM 混成一个轨道
    mix_in = "".join(bgm_input_indices)
    mix_label = "[bgm_mix]"
    filter_parts.append(f"{mix_in}amix=inputs={len(bgm_input_indices)}:normalize=0{mix_label}")

    # 如果想保留原声，可以在这里再加一条 amix 把 [0:a] 和 [bgm_mix] 叠加。
    # 目前简单起见：只用 bgm_mix 作为最终音轨。
    filter_complex = ";".join(filter_parts)

    cmd.extend([
        "-filter_complex", filter_complex,
        # 视频直接 copy
        "-map", "0:v:0",
        "-map", mix_label,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ])

    # 调 ffmpeg
    subprocess.run(cmd, check=True)