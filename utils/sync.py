from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict, Any
import math
import numpy as np


@dataclass(frozen=True)
class SyncDebug:
    """用于日志/调参：保存中间量，便于复现与排查"""
    used_librosa: bool
    window_s: float
    num_cuts: int
    per_cut_maxima: List[float]
    onset_norm_max: float


def compute_sync_score(
    audio_path: str,
    cut_times_video: Sequence[float],
    movement_start: float,
    movement_end: float,
    play_start: float = 0.0,
    window_s: float = 0.5,
    sr: int = 22050,
    hop_length: int = 512,
) -> Tuple[float, SyncDebug]:
    """
    计算 S_sync：在每个 cut 点附近的 onset strength 最大值均值

    输入说明：
    - cut_times_video：视频时间轴上的 cut 时间点（秒）
    - movement_start/movement_end：当前 movement 在视频时间轴的起止
    - play_start：音乐片段在音频文件中的起始偏移（秒）
    - 映射关系：music_time = (t_cut - movement_start) + play_start

    输出：
    - sync_score ∈ [0,1]（越大表示越多 cut 点附近有节奏事件响应）
    """
    cut_times = _sanitize_cut_times(cut_times_video, movement_start, movement_end)
    if len(cut_times) == 0:
        # 没有切点：给中性偏低（你也可以给 0.0，看你策略）
        return 0.0, SyncDebug(False, window_s, 0, [], 0.0)

    # 读音频并计算 onset envelope（优先 librosa，其次 fallback）
    y, used_librosa = _load_audio_mono(audio_path, sr=sr)
    if y is None or len(y) == 0:
        return 0.0, SyncDebug(used_librosa, window_s, len(cut_times), [0.0]*len(cut_times), 0.0)

    onset_env, onset_times = _compute_onset_envelope(y, sr=sr, hop_length=hop_length)
    if onset_env is None or onset_times is None or len(onset_env) == 0:
        return 0.0, SyncDebug(used_librosa, window_s, len(cut_times), [0.0]*len(cut_times), 0.0)

    # 归一化到 [0,1]，确保可比较、可复现
    onset_max = float(np.max(onset_env)) if len(onset_env) > 0 else 0.0
    if onset_max > 1e-8:
        onset_norm = onset_env / onset_max
    else:
        onset_norm = onset_env

    per_cut = []
    for t_cut in cut_times:
        music_t = (float(t_cut) - float(movement_start)) + float(play_start)
        t0, t1 = music_t - window_s, music_t + window_s
        # 在 onset_times 上选窗
        mask = (onset_times >= t0) & (onset_times <= t1)
        if not np.any(mask):
            per_cut.append(0.0)
        else:
            per_cut.append(float(np.max(onset_norm[mask])))

    score = float(np.mean(per_cut)) if len(per_cut) > 0 else 0.0
    score = _clamp01(score)

    dbg = SyncDebug(
        used_librosa=used_librosa,
        window_s=float(window_s),
        num_cuts=len(cut_times),
        per_cut_maxima=per_cut,
        onset_norm_max=float(np.max(onset_norm)) if len(onset_norm) else 0.0,
    )
    return score, dbg


# -----------------------
# internal helpers
# -----------------------

def _sanitize_cut_times(cut_times: Sequence[float], t_start: float, t_end: float) -> List[float]:
    out: List[float] = []
    for t in cut_times or []:
        try:
            tf = float(t)
        except Exception:
            continue
        if t_start <= tf <= t_end:
            out.append(tf)
    # 去重并排序（确定性）
    out = sorted(set(out))
    return out


def _load_audio_mono(audio_path: str, sr: int) -> Tuple[Optional[np.ndarray], bool]:
    """
    优先 librosa 读取（支持 mp3 等），否则 soundfile（通常不支持 mp3）。
    """
    # try librosa
    try:
        import librosa  # type: ignore
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        return y.astype(np.float32, copy=False), True
    except Exception:
        pass

    # fallback: soundfile (更轻，但常见只支持 wav/flac/ogg)
    try:
        import soundfile as sf  # type: ignore
        y, file_sr = sf.read(audio_path, dtype="float32", always_2d=False)
        if y is None:
            return None, False
        if y.ndim == 2:
            y = np.mean(y, axis=1)
        # 若采样率不匹配，不做复杂重采样：用简单线性插值（确定性）
        if int(file_sr) != int(sr):
            y = _resample_linear(y, float(file_sr), float(sr))
        return y.astype(np.float32, copy=False), False
    except Exception:
        return None, False


def _compute_onset_envelope(y: np.ndarray, sr: int, hop_length: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    优先 librosa 的 onset_strength（更稳），否则使用简化版 spectral flux（确定性、可复现）。
    """
    # librosa onset strength
    try:
        import librosa  # type: ignore
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
        return onset_env.astype(np.float32, copy=False), times.astype(np.float32, copy=False)
    except Exception:
        pass

    # fallback：用 STFT 幅度谱的差分（spectral flux）
    # 参数固定，确保确定性
    n_fft = 2048
    win_length = 2048

    # padding + frame
    y = np.asarray(y, dtype=np.float32)
    if len(y) < win_length:
        return None, None

    # STFT
    stft = _stft_mag(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    # flux = sum(ReLU(diff))
    diff = np.diff(stft, axis=1)
    diff = np.maximum(diff, 0.0)
    flux = np.sum(diff, axis=0)
    flux = flux.astype(np.float32, copy=False)

    # 对齐时间轴：flux 比 frame 少 1
    times = (np.arange(len(flux), dtype=np.float32) * (hop_length / float(sr)))
    return flux, times


def _stft_mag(y: np.ndarray, n_fft: int, hop_length: int, win_length: int) -> np.ndarray:
    """
    极简 STFT 幅度谱实现（确定性）
    输出 shape: [n_fft//2+1, n_frames]
    """
    y = np.asarray(y, dtype=np.float32)
    window = np.hanning(win_length).astype(np.float32)

    n_frames = 1 + (len(y) - win_length) // hop_length
    mags = []
    for i in range(n_frames):
        start = i * hop_length
        frame = y[start:start + win_length] * window
        spec = np.fft.rfft(frame, n=n_fft)
        mag = np.abs(spec).astype(np.float32)
        mags.append(mag)
    return np.stack(mags, axis=1)


def _resample_linear(y: np.ndarray, sr_from: float, sr_to: float) -> np.ndarray:
    """
    线性插值重采样（确定性），避免引入 scipy 依赖。
    """
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