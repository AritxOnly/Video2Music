from __future__ import annotations

from typing import Dict, List, Optional
import math

from agent.state import ActionType
from utils.flow import compute_flow_score
from utils.harmonic import harmonic_distance, harmonic_score_from_distance
from utils.sync import compute_sync_score


class Critic:
    """
    Critic 负责计算确定性、可复现的聚合打分与诊断标签生成：
    ΔJ = w_sem*S_sem + w_harm*S_harm + w_sync*S_sync + w_flow*S_flow - C_ops
    
    关键原则：
    1. 任何重计算（onset/rms/key/chroma）都不在这里做，只读取预计算结果
    2. 缺失特征时采用"保守低分 + 明确 failure tag"
    3. 加入"曲长越界(OOB)"惩罚，防止无限CONTINUE死循环
    4. 加入"重复搜索"惩罚，防止搜索同一首歌
    """

    def __init__(self):
        # 权重配置
        self.weights = {"sem": 1.0, "harm": 0.8, "sync": 0.6, "flow": 0.5}

        # 操作开销配置
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
        
        # 惩罚因子
        self.penalty_factors = {
            "oob": 2.0,  # 曲长越界惩罚因子
            "repeat_search": 0.5,  # 重复搜索惩罚
        }

    # ---------------- public API ----------------

    def evaluate(self, state, action_type, movement, track, movement_idx: int) -> float:
        """
        计算单步得分 ΔJ，并更新 state.failure_tags 与 state.accumulated_cost
        关键修改：加入曲长越界(OOB)惩罚和重复搜索惩罚
        """
        
        # 基础打分
        s_sem = self._score_sem(track)
        s_harm = self._score_harm(state, track, movement_idx)
        s_sync = self._score_sync(movement, track)
        s_flow = self._score_flow(movement, track)
        
        base_cost = float(self.cost_table.get(action_type, 0.1))
        extra_cost = 0.0
        oob_penalty_applied = False
        
        # 关键修改1：曲长越界(OOB)惩罚
        track_dur = self._get_track_duration(track)
        movement_dur = float(getattr(movement, "end_time", 0.0)) - float(getattr(movement, "start_time", 0.0))
        
        if track_dur > 0 and movement_dur > 0:
            source_start = self._estimate_source_start(state, movement_idx, track)
            overflow = (source_start + movement_dur) - track_dur
            
            if overflow > 0:
                # 指数级暴击惩罚：一旦越界就重罚
                extra_cost += (overflow * 0.5) + self.penalty_factors['oob']
                oob_penalty_applied = True
                
                # 关键：只要超时，强制判定为 Flow/Sync 失败
                s_sync = 0.0
                s_flow = 0.0
                s_sem *= 0.5  # 语义匹配度也打折
                s_harm *= 0.7  # 和声连续性也打折
                
                # 打印调试信息
                print(f"    [Critic] PUNISH: Track {getattr(track, 'id', 'unknown')} "
                      f"overflows by {overflow:.2f}s! Extra Cost: {extra_cost}")

        # 关键修改2：重复搜索惩罚
        if action_type == ActionType.SEARCH and track:
            track_id = getattr(track, 'id', None)
            if track_id and self._is_redundant_search(state, track_id, movement_idx):
                print(f"    [Critic] PUNISH: Redundant SEARCH for same track {getattr(track, 'id', 'unknown')}")
                extra_cost += self.penalty_factors["repeat_search"]
        
        cost = base_cost + extra_cost
        
        # 计算总分
        step_score = (
            self.weights["sem"] * s_sem
            + self.weights["harm"] * s_harm
            + self.weights["sync"] * s_sync
            + self.weights["flow"] * s_flow
            - cost
        )

        # 诊断标签
        tags = self._detect_failure_tags(
            s_sem=s_sem,
            s_harm=s_harm,
            s_sync=s_sync,
            s_flow=s_flow,
            movement=movement,
            track=track,
            oob_penalty_applied=oob_penalty_applied,
            movement_idx=movement_idx,
            state=state,
        )
        
        # 关键修改3：如果曲长越界，添加相关标签
        if oob_penalty_applied:
            tags["TRACK_OOB"] = True
            tags["TRACK_OVERFLOW"] = True
            # 计算剩余可用时间
            if track_dur > 0:
                source_start = self._estimate_source_start(state, movement_idx, track)
                remaining = track_dur - source_start
                if remaining < 5.0:  # 剩余小于5秒
                    tags["TRACK_REMAIN_TOO_SHORT"] = True
        
        # 更新failure tags
        state.failure_tags = tags
        
        # 累计 cost
        state.accumulated_cost = float(getattr(state, "accumulated_cost", 0.0)) + cost

        return float(step_score)

    # ---------------- scoring helpers ----------------

    @staticmethod
    def _clamp01(x: float) -> float:
        """将分数限制在[0,1]范围内"""
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return 0.0
        return max(0.0, min(1.0, float(x)))

    def _score_sem(self, track) -> float:
        """
        S_sem：语义匹配度
        - 由 MusicService + utils.scorer 预先写入 track.meta['sem_score'] ∈ [0,1]
        """
        meta = getattr(track, "meta", {}) or {}
        if "sem_score" not in meta:
            return 0.0  # 缺失时给0分，而不是保守分
        return self._clamp01(meta.get("sem_score", 0.0))

    def _score_sync(self, movement, track) -> float:
        """S_sync：节奏同步度"""
        meta = getattr(track, "meta", {}) or {}
        
        # 优先使用缓存分数
        if "sync_score" in meta:
            return self._clamp01(meta.get("sync_score", 0.0))

        # 缺失则现场计算（并缓存）
        filepath = meta.get("filepath", "")
        
        # 关键修改：支持新字段名 cut_points，兼容旧字段名 cut_times
        cut_times = getattr(movement, "cut_points", None)
        if not cut_times:
            # 兼容旧字段
            cut_times = getattr(movement, "cut_times", None)
        if not cut_times and hasattr(movement, "meta") and isinstance(movement.meta, dict):
            cut_times = movement.meta.get("cut_points") or movement.meta.get("cut_times") or []

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
        
        # 缓存结果
        meta["sync_score"] = float(score)
        meta["sync_debug"] = {
            "used_librosa": getattr(dbg, "used_librosa", None),
            "window_s": getattr(dbg, "window_s", None),
            "num_cuts": getattr(dbg, "num_cuts", None),
            "onset_norm_max": getattr(dbg, "onset_norm_max", None),
        }
        
        # 更新track meta
        if hasattr(track, 'meta'):
            track.meta = meta
        elif hasattr(track, '__dict__'):
            track.__dict__.setdefault('meta', {}).update(meta)
            
        return self._clamp01(score)

    def _score_flow(self, movement, track) -> float:
        """S_flow：能量流动匹配度"""
        meta = getattr(track, "meta", {}) or {}
        
        # 优先使用缓存分数
        if "flow_score" in meta:
            return self._clamp01(meta.get("flow_score", 0.0))

        filepath = meta.get("filepath", "")
        if not filepath:
            return 0.0

        # 关键修改：支持新字段名 visual_energy_curve，兼容旧字段名 visual_energy
        visual_energy = getattr(movement, "visual_energy_curve", None)
        if visual_energy is None:
            visual_energy = getattr(movement, "visual_energy", None)  # 兼容旧字段
        if visual_energy is None and hasattr(movement, "meta") and isinstance(movement.meta, dict):
            visual_energy = movement.meta.get("visual_energy_curve") or movement.meta.get("visual_energy")

        # 获取时间轴
        visual_times = getattr(movement, "visual_energy_times", None)
        if visual_times is None and hasattr(movement, "meta") and isinstance(movement.meta, dict):
            visual_times = movement.meta.get("visual_energy_times", None)

        if visual_energy is None:
            return 0.0

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
        
        # 缓存结果
        meta["flow_score"] = float(score)
        meta["flow_debug"] = {
            "used_librosa": getattr(dbg, "used_librosa", None),
            "resample_hz": getattr(dbg, "resample_hz", None),
            "n_points": getattr(dbg, "n_points", None),
            "pearson_corr": getattr(dbg, "pearson_corr", None),
        }
        
        # 更新track meta
        if hasattr(track, 'meta'):
            track.meta = meta
        elif hasattr(track, '__dict__'):
            track.__dict__.setdefault('meta', {}).update(meta)
            
        return self._clamp01(score)

    def _score_harm(self, state, track, movement_idx: int) -> float:
        """
        S_harm：和声连续性
        - 使用"纯五度圈距离"版本，确定性、可复用
        """
        prev_track = self._get_prev_track(state, movement_idx)
        if prev_track is None:
            return 1.0  # 第一首没有前一首，给满分

        cur_meta = getattr(track, "meta", {}) or {}
        prev_meta = getattr(prev_track, "meta", {}) or {}

        cur_key = cur_meta.get("key", None)
        prev_key = prev_meta.get("key", None)

        dist = harmonic_distance(prev_key, cur_key)  # 解析失败 -> 6
        return self._clamp01(harmonic_score_from_distance(dist))

    def _get_track_duration(self, track) -> float:
        """安全获取音轨时长"""
        if track is None:
            return 0.0
            
        # 尝试多种方式获取时长
        if hasattr(track, 'duration'):
            dur = getattr(track, 'duration')
            if dur is not None:
                return float(dur)
                
        if hasattr(track, 'meta'):
            meta = getattr(track, 'meta', {}) or {}
            if 'duration' in meta:
                return float(meta['duration'])
                
        # 从filepath读取（最后的手段）
        if hasattr(track, 'meta'):
            meta = getattr(track, 'meta', {}) or {}
            filepath = meta.get('filepath', '')
            if filepath:
                # 这里可以添加从文件读取时长的逻辑，但为了性能，最好在track创建时就算好
                pass
                
        return 0.0

    @staticmethod
    def _get_prev_track(state, movement_idx: int):
        """获取前一首分配的音轨"""
        current_idx = movement_idx
        prev_idx = current_idx - 1
        if prev_idx < 0:
            return None
            
        assigned = getattr(state, "assigned_tracks", {}) or {}
        return assigned.get(prev_idx, None)
    
    def _is_redundant_search(self, state, track_id: str, movement_idx: int) -> bool:
        hist = getattr(state, "action_history", [])[-3:]
        cnt = 0
        for h in hist:
            if h.get("movement_idx") == movement_idx and h.get("action") == "SEARCH":
                # 如果你把 track_id 也记录进 history，这里就能判“同结果”
                if h.get("track_id") == track_id:
                    cnt += 1
        return cnt >= 1

    def _estimate_source_start(self, state, movement_idx: int, track) -> float:
        """
        估算当前 movement 对应的音频 source_start（与 renderer 的逻辑一致）
        
        关键逻辑：
        1. 找到这首歌从哪个 movement 开始延续
        2. 累加从起点到当前 movement 之前的所有 movement 时长
        3. 加上 track 的 play_start
        """
        if track is None:
            return 0.0
            
        # 获取 track 的播放起始时间
        base = float(getattr(track, "play_start", 0.0) or 0.0)

        assigned = getattr(state, "assigned_tracks", {}) or {}
        track_id = getattr(track, "id", None)
        
        if track_id is None:
            return base

        # 回溯找到这首歌从哪个 movement 开始延续
        start_idx = movement_idx
        while start_idx > 0:
            prev_track = assigned.get(start_idx - 1)
            if prev_track and getattr(prev_track, "id", None) == track_id:
                start_idx -= 1
            else:
                break

        # 累加从 start_idx 到 movement_idx 之间的 movement 时长（不包含当前）
        elapsed = 0.0
        movements = getattr(state, "movements", [])
        
        for k in range(start_idx, movement_idx):
            if k < len(movements):
                m = movements[k]
                elapsed += float(getattr(m, "end_time", 0.0) - getattr(m, "start_time", 0.0))

        return base + elapsed

    # ---------------- failure tags ----------------

    def _detect_failure_tags(self, s_sem, s_harm, s_sync, s_flow, movement, track, 
                           oob_penalty_applied=False, movement_idx=0, state=None) -> Dict[str, bool]:
        """
        生成诊断标签，用于驱动 agent 决策
        
        关键修改：包含OOB标签和片段长度相关标签
        """
        tags: Dict[str, bool] = {}

        # 低分标签
        if s_sem < self.thresholds["sem"]:
            tags["SEM_LOW"] = True
        if s_harm < self.thresholds["harm"]:
            tags["HARM_BAD"] = True
        if s_sync < self.thresholds["sync"]:
            tags["SYNC_BAD"] = True
        if s_flow < self.thresholds["flow"]:
            tags["FLOW_BAD"] = True

        # 片段长度相关标签
        dur = float(getattr(movement, "end_time", 0.0)) - float(getattr(movement, "start_time", 0.0))
        if dur > 30.0:
            visual_variance = float(getattr(movement, "visual_variance", 0.0))
            if visual_variance > 0.8:
                tags["TOO_LONG_VAR"] = True
            tags["LONG_FRAGMENT"] = True
            
        if dur > 45.0:
            tags["TOO_LONG_FOR_ONE_TRACK"] = True
            
        if dur < 5.0:
            tags["TOO_SHORT_FRAGMENT"] = True

        # 缺失特征标签
        meta = getattr(track, "meta", {}) or {}
        if "sem_score" not in meta:
            tags["MISSING_SEM_SCORE"] = True
        if "sync_score" not in meta:
            tags["MISSING_SYNC_SCORE"] = True
        if "flow_score" not in meta:
            tags["MISSING_FLOW_SCORE"] = True
        if "key" not in meta:
            tags["MISSING_KEY"] = True

        # OOB相关标签
        if oob_penalty_applied:
            tags["TRACK_OOB"] = True
            
            # 计算具体信息
            track_dur = self._get_track_duration(track)
            if track_dur > 0:
                source_start = self._estimate_source_start(state, movement_idx, track)
                remaining = track_dur - source_start
                if remaining < 0:
                    tags["TRACK_OVERFLOW_SEC"] = True
                elif remaining < 5.0:
                    tags["TRACK_REMAIN_TOO_SHORT"] = True

        return tags