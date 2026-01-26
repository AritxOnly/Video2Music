from typing import Dict, Any, List
import numpy as np
from agent.state import AgentState, Track, Movement, ActionType

class Critic:
    def __init__(self):
        # 权重配置
        self.weights = {
            'sem': 1.0,
            'harm': 0.8,
            'sync': 0.6,
            'flow': 0.5
        }
        
        # 操作开销配置
        self.cost_table = {
            ActionType.CONTINUE: 0.0,
            ActionType.REQUERY: 0.1,       # 惩罚盲目试错
            ActionType.SEARCH: 0.1,        # 基础搜索开销
            ActionType.GENERATE_SUNO: 0.5, # 强约束，防止过度依赖生成
            ActionType.TRIM: 0.05,
            ActionType.SPLIT: 0.2,         # 结构调整成本较高
            ActionType.MERGE: 0.1,
            ActionType.SHIFT_ALIGN: 0.1
        }

        # Failure Tags 阈值
        self.thresholds = {
            'sem': 0.4,
            'harm': 0.3,
            'sync': 0.3,
            'flow': 0.2
        }

    def evaluate(self, state: AgentState, action_type: ActionType, movement: Movement, track: Track) -> float:
        """
        计算单步得分 Delta J，并更新状态的 Failure Tags
        J = w_sem*S_sem + w_harm*S_harm + w_sync*S_sync + w_flow*S_flow - Cost
        """
        # 1. 计算各项子分 (Sub-scores)
        # 注意：实际项目中这里需要调用 external modules (vsem, mret, etc.)
        # 这里为了跑通逻辑，假设 track.meta 和 movement 已经包含了预计算特征
        
        s_sem = self._score_sem(movement, track)
        s_harm = self._score_harm(state, track)
        s_sync = self._score_sync(movement, track)
        s_flow = self._score_flow(movement, track)
        
        # 2. 计算操作开销 (Operation Cost)
        cost = self.cost_table.get(action_type, 0.1)
        
        # 3. 加权求和
        step_score = (
            self.weights['sem'] * s_sem +
            self.weights['harm'] * s_harm +
            self.weights['sync'] * s_sync +
            self.weights['flow'] * s_flow - 
            cost
        )

        # 4. 生成诊断标签 (Failure Tags) 
        new_tags = self._detect_failure_tags(s_sem, s_harm, s_sync, s_flow, movement, track)
        state.failure_tags = new_tags
        
        # 更新状态中的累计 Cost
        state.accumulated_cost += cost
        
        return step_score

    def _score_sem(self, movement: Movement, track: Track) -> float:
        """语义匹配度: Embedding Cosine + Keyword Jaccard"""
        # 实际应调用: vsem.calculate_similarity(movement.visual_summary, track.meta['tags'])
        # 这里由外部检索器返回的 score 模拟
        return track.meta.get('sem_score', 0.5)

    def _score_harm(self, state: AgentState, track: Track) -> float:
        """和声连续性: 五度圈距离"""
        # 获取上一段音乐的 Key
        prev_idx = state.current_movement_index - 1
        if prev_idx < 0 or prev_idx not in state.assigned_tracks:
            return 1.0 # 第一段默认和谐
            
        prev_track = state.assigned_tracks[prev_idx]
        
        # 模拟计算逻辑
        # 基于五度圈理论
        current_key = track.meta.get('key', 'C')
        prev_key = prev_track.meta.get('key', 'C')
        
        # TODO: 接入 librosa/madmom 或简单的五度圈查找表
        # circle_dist = get_circle_of_fifths_distance(prev_key, current_key)
        # return 1.0 - (circle_dist / 6.0)
        
        return 0.8 # Placeholder

    def _score_sync(self, movement: Movement, track: Track) -> float:
        """节奏对齐度: Onset Strength at Cut Points"""
        # 这是一个计算密集型操作，实际应在 toolbox.analyze_sync 中完成
        return track.meta.get('sync_score', 0.5)

    def _score_flow(self, movement: Movement, track: Track) -> float:
        """能量对齐度: Pearson Correlation"""
        return track.meta.get('flow_score', 0.5)

    def _detect_failure_tags(self, s_sem, s_harm, s_sync, s_flow, movement, track) -> Dict[str, bool]:
        tags = {}
        
        # 基础低分 Tag
        if s_sem < self.thresholds['sem']: tags['SEM_LOW'] = True
        if s_harm < self.thresholds['harm']: tags['HARM_BAD'] = True
        if s_sync < self.thresholds['sync']: tags['SYNC_BAD'] = True
        if s_flow < self.thresholds['flow']: tags['FLOW_BAD'] = True
        
        # TOO_LONG_VAR
        # movement 太长且视觉变化大
        if movement.end_time - movement.start_time > 30.0:
            # 假设 visual_variance 已经预计算
            if getattr(movement, 'visual_variance', 0.0) > 0.8:
                tags['TOO_LONG_VAR'] = True
                
        # TOO_SHORT_FRAGMENT
        if movement.end_time - movement.start_time < 5.0:
            tags['TOO_SHORT_FRAGMENT'] = True
            
        return tags