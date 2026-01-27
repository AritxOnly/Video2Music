from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, List
import math
import numpy as np


@dataclass(frozen=True)
class FlowDebug:
    """用于日志/调参：保存中间量"""
    used_librosa: bool
    resample_hz: float
    n_points: int
    pearson_corr: float


def compute_flow_score(
    audio_path: str,
    visual_energy: Sequence[float],
    movement_start: float,
    movement_end: float,
    play_start: float = 0.0,
    visual_times: Optional[Sequence[float]] = None,
    resample_hz: float = 1.0,
    sr: int = 22050,
    hop_length: int = 512,
) -> Tuple[float, FlowDebug]:
    """
    计算 S_flow：视频 visual_energy 与音频 rms_energy 的相关性（Pearson）
    - 将二者重采样到同一时间栅格（默认 1Hz）
    - corr ∈ [-1,1]，再映射到 [0,1]：score = (corr+1)/2

    时间对齐：
    - video_time = [movement_start, movement_end]
    - music_time = video_time - movement_start + play_start
    """
    v = np.asarray(list(visual_energy), dtype=np.float32).flatten()
    if len(v) < 2:
        return 0.0, FlowDebug(False, float(resample_hz), 0, 0.0)

    # 读音频并算 rms 曲线
    y, used_librosa = _load_audio_mono(audio_path, sr=sr)
    if y is None or len(y) == 0:
        return 0.0, FlowDebug(used_librosa, float(resample_hz), 0, 0.0)

    rms, rms_times = _compute_rms(y, sr=sr, hop_length=hop_length)
    if rms is None or rms_times is None or len(rms) < 2:
        return 0.0, FlowDebug(used_librosa, float(resample_hz), 0, 0.0)

    # 构造共同时间栅格（video 时间轴）
    t0 = float(movement_start)
    t1 = float(movement_end)
    if t1 <= t0:
        return 0.0, FlowDebug(used_librosa, float(resample_hz), 0, 0.0)

    step = 1.0 / float(resample_hz)
    grid_video = np.arange(t0, t1, step, dtype=np.float32)
    if len(grid_video) < 2:
        return 0.0, FlowDebug(used_librosa, float(resample_hz), 0, 0.0)

    # 1) resample visual_energy 到 grid_video
    v_rs = _resample_visual_to_grid(v, visual_times, grid_video, t0=t0, t1=t1)

    # 2) resample audio rms 到 grid_video（先把 grid_video 映射到 music_time）
    grid_music = (grid_video - t0) + float(play_start)
    a_rs = np.interp(grid_music, rms_times.astype(np.float32), rms.astype(np.float32)).astype(np.float32)

    # 去除常量序列导致的 NaN
    corr = _pearson_corr(v_rs, a_rs)
    score = (corr + 1.0) / 2.0
    score = _clamp01(score)

    return score, FlowDebug(
        used_librosa=used_librosa,
        resample_hz=float(resample_hz),
        n_points=int(len(grid_video)),
        pearson_corr=float(corr),
    )


# -----------------------
# internal helpers
# -----------------------

def _load_audio_mono(audio_path: str, sr: int):
    try:
        import librosa  # type: ignore
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        return y.astype(np.float32, copy=False), True
    except Exception:
        pass

    try:
        import soundfile as sf  # type: ignore
        y, file_sr = sf.read(audio_path, dtype="float32", always_2d=False)
        if y is None:
            return None, False
        if y.ndim == 2:
            y = np.mean(y, axis=1)
        if int(file_sr) != int(sr):
            y = _resample_linear(y, float(file_sr), float(sr))
        return y.astype(np.float32, copy=False), False
    except Exception:
        return None, False


def _compute_rms(y: np.ndarray, sr: int, hop_length: int):
    """
    优先 librosa.feature.rms，否则用帧 RMS fallback。
    返回：
    - rms: [n_frames]
    - times: [n_frames]（秒）
    """
    try:
        import librosa  # type: ignore
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
        # 为了稳定，归一化到 [0,1]（只在段内比较也行；这里全局归一）
        mx = float(np.max(rms)) if len(rms) else 0.0
        if mx > 1e-8:
            rms = rms / mx
        return rms.astype(np.float32, copy=False), times.astype(np.float32, copy=False)
    except Exception:
        pass

    # fallback：固定窗长，逐帧 RMS
    win_length = 2048
    if len(y) < win_length:
        return None, None
    n_frames = 1 + (len(y) - win_length) // hop_length
    rms_list = []
    for i in range(n_frames):
        start = i * hop_length
        frame = y[start:start + win_length]
        rms_list.append(float(np.sqrt(np.mean(frame * frame) + 1e-12)))
    rms = np.asarray(rms_list, dtype=np.float32)
    mx = float(np.max(rms)) if len(rms) else 0.0
    if mx > 1e-8:
        rms = rms / mx
    times = (np.arange(len(rms), dtype=np.float32) * (hop_length / float(sr)))
    return rms, times


def _resample_visual_to_grid(v: np.ndarray, visual_times: Optional[Sequence[float]], grid: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """
    将 visual_energy 对齐到 grid（video time）
    - 若 visual_times 给了：用 np.interp 做确定性插值
    - 若没给：假设 v 是等间隔采样并覆盖 [t0,t1]，做线性映射到 grid
    """
    v = v.astype(np.float32, copy=False).flatten()
    if visual_times is not None and len(visual_times) == len(v) and len(v) >= 2:
        vt = np.asarray(list(visual_times), dtype=np.float32).flatten()
        return np.interp(grid, vt, v).astype(np.float32)

    # fallback：认为 v 覆盖 [t0,t1] 等间隔
    if len(v) < 2:
        return np.zeros_like(grid, dtype=np.float32)
    vt = np.linspace(t0, t1, num=len(v), dtype=np.float32)
    return np.interp(grid, vt, v).astype(np.float32)


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32).flatten()
    y = np.asarray(y, dtype=np.float32).flatten()
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return 0.0
    x0 = x - float(np.mean(x))
    y0 = y - float(np.mean(y))
    denom = float(np.linalg.norm(x0) * np.linalg.norm(y0))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(x0, y0) / denom)


def _resample_linear(y: np.ndarray, sr_from: float, sr_to: float) -> np.ndarray:
    if sr_from <= 0 or sr_to <= 0 or len(y) == 0:
        return y
    ratio = sr_to / sr_from
    n_to = int(round(len(y) * ratio))
    x_from = np.linspace(0.0, 1.0, num=len(y), dtype=np.float32)
    x_to = np.linspace(0.0, 1.0, num=n_to, dtype=np.float32)
    return np.interp(x_to, x_from, y).astype(np.float32)


def _clamp01(x: float) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    return max(0.0, min(1.0, float(x)))