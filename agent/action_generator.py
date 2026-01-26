from typing import List, Dict
import random
from agent.state import AgentState, ActionType

class ActionGenerator:
    def __init__(self, llm_interface, toolbox):
        self.llm = llm_interface
        self.toolbox = toolbox
        
        # 初始权值映射 (Soft Priors)
        # Base probability weights
        self.base_weights = {
            ActionType.CONTINUE: 1.0,
            ActionType.SEARCH: 1.0,
            ActionType.REQUERY: 0.5,
            ActionType.TRIM: 0.5,
            ActionType.SPLIT: 0.2, # 结构调整慎用
            ActionType.MERGE: 0.2,
            ActionType.GENERATE_SUNO: 0.1 # Cost高，慎用
        }

    def get_action_priors(self, state: AgentState) -> Dict[ActionType, float]:
        """
        根据当前状态的 Failure Tags 动态调整动作权重
        Failure Tags -> Action Masks/Weights
        """
        weights = self.base_weights.copy()
        tags = state.failure_tags
        
        # 如果是初始状态 (没有 track)，只能 SEARCH 或 GENERATE
        if state.current_movement_index not in state.assigned_tracks:
            return {
                ActionType.SEARCH: 2.0,
                ActionType.GENERATE_SUNO: 0.5,
                ActionType.MERGE: 0.5 # 允许开局合并
            }

        # SEM_LOW: 倾向 REQUERY, SEARCH
        if tags.get('SEM_LOW'):
            weights[ActionType.REQUERY] *= 3.0
            weights[ActionType.SEARCH] *= 2.0
            weights[ActionType.CONTINUE] *= 0.1

        # FILTER_KILL: 倾向 RELAX_CONSTRAINT
        if tags.get('FILTER_KILL'):
            weights[ActionType.RELAX_CONSTRAINT] *= 5.0
            weights[ActionType.SEARCH] *= 1.5

        # HARM_BAD: 倾向 SELECT_NEW (或重新 Search)
        if tags.get('HARM_BAD'):
            weights[ActionType.SEARCH] *= 2.0
            weights[ActionType.CONTINUE] *= 0.0  # 绝对不能继续难听的

        # SYNC_BAD: 倾向 SHIFT_ALIGN, TRIM
        if tags.get('SYNC_BAD'):
            weights[ActionType.SHIFT_ALIGN] *= 4.0
            weights[ActionType.TRIM] *= 3.0
            
        # TOO_LONG_VAR: 优先 SPLIT
        if tags.get('TOO_LONG_VAR'):
            weights[ActionType.SPLIT] *= 10.0 # 强烈建议
            
        # TOO_SHORT_FRAGMENT: 优先 MERGE, CONTINUE
        if tags.get('TOO_SHORT_FRAGMENT'):
            weights[ActionType.MERGE] *= 5.0
            weights[ActionType.CONTINUE] *= 3.0

        return weights

    def propose(self, state: AgentState, k=3) -> List[Dict]:
        """
        1. 计算 Prior Weights
        2. 如果权重分布极度倾斜，直接生成规则动作
        3. 否则，Prompt LLM 生成具体参数 (如 Search Query)
        """
        priors = self.get_action_priors(state)
        
        # 简单策略：选择权重最高的 Top-N 类型，然后让 LLM 填充参数
        # 实际代码中，这里会调用 self.llm.get_action_params(action_type, context)
        # ...
        pass