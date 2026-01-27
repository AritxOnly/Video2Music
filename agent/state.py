from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum

from vsem.model import SegmentSemantics

# 定义动作类型
class ActionType(Enum):
    # Structural
    SPLIT = "SPLIT"
    MERGE = "MERGE"
    
    # Retrieval & Healing
    SEARCH = "SEARCH"
    REQUERY = "REQUERY"
    RELAX_CONSTRAINT = "RELAX_CONSTRAINT"
    
    # Editing
    CONTINUE = "CONTINUE"
    TRIM = "TRIM"
    SHIFT_ALIGN = "SHIFT_ALIGN"
    
    # Fallback
    GENERATE_SUNO = "GENERATE_SUNO"

@dataclass
class Track:
    id: str
    source: str  # 'library', 'suno', etc.
    duration: float
    meta: Dict[str, Any]  # bpm, key, tags, etc.
    # 实际用于播放的片段信息
    play_start: float = 0.0
    play_duration: float = 0.0

@dataclass
class Movement:
    """对应 Macro Layer 规划出的一个乐章 (Movement)"""
    id: str
    start_time: float
    end_time: float
    shots: List[SegmentSemantics] = field(default_factory=list)
    
    # 视觉特征缓存 (避免重复计算)
    visual_summary: str = ""
    visual_energy_curve: List[float] = field(default_factory=list)
    cut_points: List[float] = field(default_factory=list)

@dataclass
class AgentState:
    """搜索树中的一个节点 (State)"""
    # 结构信息
    movements: List[Movement]
    current_movement_index: int = 0
    
    # 已做出的决策 (History)
    # Mapping: movement_index -> assigned_track (Track Object)
    assigned_tracks: Dict[int, Track] = field(default_factory=dict)
    
    # 全局感知上下文
    global_semantics: Optional[Any] = None
    
    # 路径记录 (用于回溯 Debug)
    action_history: List[Dict[str, Any]] = field(default_factory=list) # [{'action': 'SEARCH', 'param': 'Sad', 'step_score': 0.8}]
    
    # 评分状态
    total_score: float = 0.0
    accumulated_cost: float = 0.0
    
    # Failure Tags (当前状态的“诊断书”)
    # e.g., {'SEM_LOW': True, 'SYNC_BAD': False}
    failure_tags: Dict[str, bool] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.current_movement_index >= len(self.movements)

    def clone(self):
        import copy
        return copy.deepcopy(self)