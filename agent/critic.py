from __future__ import annotations

from typing import Dict
import math

from agent.state import ActionType
from utils.flow import compute_flow_score
from utils.harmonic import harmonic_distance, harmonic_score_from_distance
from utils.sync import compute_sync_score


class Critic:
    """
    Critic 只做“确定性、可复现”的聚合打分与诊断标签生成：
    ΔJ = w_sem*S_sem + w_harm*S_harm + w_sync*S_sync + w_flow*S_flow - C_ops

    关键原则：
    - 任何重计算（onset/rms/key/chroma）都不在这里做，只读取 track.meta / movement 的预计算结果
    - 缺失特征时采用“保守低分 + 明确 failure tag”，避免 silent fail
    """

    def __init__(self):
        # 权重配置（你可以后续做超参搜索 / 学习）
        self.weights = {"sem": 1.0, "harm": 0.8, "sync": 0.6, "flow": 0.5}

        # 操作开销配置：用于惩罚过度试错/过度生成
        self.cost_table = {
            ActionType.CONTINUE: 0.0,
            ActionType.REQUERY: 0.1,
            ActionType.SEARCH: 0.1,
            ActionType.GENERATE_SUNO: 0.5,
            ActionType.TRIM: 0.05,
            ActionType.SPLIT: 0.2,
            ActionType.MERGE: 0.1,
            ActionType.SHIFT_ALIGN: 0.1,
        }

        # Failure Tags 阈值
        self.thresholds = {"sem": 0.4, "harm": 0.3, "sync": 0.3, "flow": 0.2}

    # ---------------- public API ----------------

    def evaluate(self, state, action_type, movement, track) -> float:
        """
        计算单步得分 ΔJ，并更新 state.failure_tags 与 state.accumulated_cost
        """
        s_sem = self._score_sem(track)
        s_harm = self._score_harm(state, track)
        s_sync = self._score_sync(movement, track)
        s_flow = self._score_flow(movement, track)

        cost = float(self.cost_table.get(action_type, 0.1))

        step_score = (
            self.weights["sem"] * s_sem
            + self.weights["harm"] * s_harm
            + self.weights["sync"] * s_sync
            + self.weights["flow"] * s_flow
            - cost
        )

        # 诊断标签
        state.failure_tags = self._detect_failure_tags(
            s_sem=s_sem,
            s_harm=s_harm,
            s_sync=s_sync,
            s_flow=s_flow,
            movement=movement,
            track=track,
        )

        # 累计 cost
        state.accumulated_cost = float(getattr(state, "accumulated_cost", 0.0)) + cost

        return float(step_score)

    # ---------------- scoring helpers ----------------

    @staticmethod
    def _clamp01(x: float) -> float:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return 0.0
        return max(0.0, min(1.0, float(x)))

    def _score_sem(self, track) -> float:
        """
        S_sem：语义匹配度
        - 由 MusicService + utils.scorer 预先写入 track.meta['sem_score'] ∈ [0,1]
        - 缺失时给保守低分（不是 0.5），并由 failure tags 指出缺失
        """
        meta = getattr(track, "meta", {}) or {}
        if "sem_score" not in meta:
            return 0.0
        return self._clamp01(meta.get("sem_score", 0.0))

    def _score_sync(self, movement, track) -> float:
        meta = getattr(track, "meta", {}) or {}
        if "sync_score" in meta:
            return self._clamp01(meta.get("sync_score", 0.0))

        # 缺失则现场算（并缓存）
        filepath = meta.get("filepath", "")
        cut_times = getattr(movement, "cut_times", None)
        if cut_times is None and hasattr(movement, "meta") and isinstance(movement.meta, dict):
            cut_times = movement.meta.get("cut_times", [])

        if not filepath or not cut_times:
            return 0.0

        play_start = float(getattr(track, "play_start", meta.get("play_start", 0.0)) or 0.0)
        score, dbg = compute_sync_score(
            audio_path=filepath,
            cut_times_video=cut_times,
            movement_start=float(getattr(movement, "start_time", 0.0)),
            movement_end=float(getattr(movement, "end_time", 0.0)),
            play_start=play_start,
            window_s=0.5,
        )
        meta["sync_score"] = float(score)
        meta["sync_debug"] = {
            "used_librosa": getattr(dbg, "used_librosa", None),
            "window_s": getattr(dbg, "window_s", None),
            "num_cuts": getattr(dbg, "num_cuts", None),
            "onset_norm_max": getattr(dbg, "onset_norm_max", None),
        }
        track.meta = meta  # 确保写回（若 Track.meta 是可写字段）
        return self._clamp01(score)

    def _score_flow(self, movement, track) -> float:
        meta = getattr(track, "meta", {}) or {}
        if "flow_score" in meta:
            return self._clamp01(meta.get("flow_score", 0.0))

        filepath = meta.get("filepath", "")
        if not filepath:
            return 0.0

        # visual_energy 从 movement 读
        visual_energy = getattr(movement, "visual_energy", None)
        if visual_energy is None and hasattr(movement, "meta") and isinstance(movement.meta, dict):
            visual_energy = movement.meta.get("visual_energy", None)

        if visual_energy is None:
            return 0.0

        visual_times = getattr(movement, "visual_energy_times", None)
        if visual_times is None and hasattr(movement, "meta") and isinstance(movement.meta, dict):
            visual_times = movement.meta.get("visual_energy_times", None)

        play_start = float(getattr(track, "play_start", meta.get("play_start", 0.0)) or 0.0)
        score, dbg = compute_flow_score(
            audio_path=filepath,
            visual_energy=visual_energy,
            visual_times=visual_times,
            movement_start=float(getattr(movement, "start_time", 0.0)),
            movement_end=float(getattr(movement, "end_time", 0.0)),
            play_start=play_start,
            resample_hz=1.0,
        )
        meta["flow_score"] = float(score)
        meta["flow_debug"] = {
            "used_librosa": getattr(dbg, "used_librosa", None),
            "resample_hz": getattr(dbg, "resample_hz", None),
            "n_points": getattr(dbg, "n_points", None),
            "pearson_corr": getattr(dbg, "pearson_corr", None),
        }
        track.meta = meta
        return self._clamp01(score)

    def _score_harm(self, state, track) -> float:
        """
        S_harm：和声连续性（先用“纯五度圈距离”版本，确定性、可复用）
        - dist = harmonic_distance(prev_key, cur_key) ∈ [0,6]
        - score = 1 - dist/6

        说明：
        - 你论文里的 key_conf/chroma 混合版后续再加；
        - 目前 library 自带 key 时，这个就足够用来做“避免听感崩坏”的约束。
        """
        prev_track = self._get_prev_track(state)
        if prev_track is None:
            return 1.0

        cur_meta = getattr(track, "meta", {}) or {}
        prev_meta = getattr(prev_track, "meta", {}) or {}

        cur_key = cur_meta.get("key", None)
        prev_key = prev_meta.get("key", None)

        dist = harmonic_distance(prev_key, cur_key)  # 解析失败 -> 6
        return self._clamp01(harmonic_score_from_distance(dist))

    @staticmethod
    def _get_prev_track(state):
        prev_idx = int(getattr(state, "current_movement_index", 0)) - 1
        if prev_idx < 0:
            return None
        assigned = getattr(state, "assigned_tracks", {}) or {}
        return assigned.get(prev_idx, None)

    # ---------------- failure tags ----------------

    def _detect_failure_tags(self, s_sem, s_harm, s_sync, s_flow, movement, track) -> Dict[str, bool]:
        tags: Dict[str, bool] = {}

        # 低分 tag
        if s_sem < self.thresholds["sem"]:
            tags["SEM_LOW"] = True
        if s_harm < self.thresholds["harm"]:
            tags["HARM_BAD"] = True
        if s_sync < self.thresholds["sync"]:
            tags["SYNC_BAD"] = True
        if s_flow < self.thresholds["flow"]:
            tags["FLOW_BAD"] = True

        # 片段长度相关 tag
        dur = float(getattr(movement, "end_time", 0.0)) - float(getattr(movement, "start_time", 0.0))
        if dur > 30.0 and float(getattr(movement, "visual_variance", 0.0)) > 0.8:
            tags["TOO_LONG_VAR"] = True
        if dur < 5.0:
            tags["TOO_SHORT_FRAGMENT"] = True

        # 缺失特征 tag（用于驱动 agent 决策，而不是悄悄给默认值）
        meta = getattr(track, "meta", {}) or {}
        if "sem_score" not in meta:
            tags["MISSING_SEM_SCORE"] = True
        if "sync_score" not in meta:
            tags["MISSING_SYNC_SCORE"] = True
        if "flow_score" not in meta:
            tags["MISSING_FLOW_SCORE"] = True
        if "key" not in meta:
            tags["MISSING_KEY"] = True

        return tags