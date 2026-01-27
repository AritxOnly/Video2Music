import json
import os
from pathlib import Path
import re
import random
from typing import List, Dict, Any, Tuple
from agent.state import AgentState, ActionType, Movement
from llm.interface import LLMInterface

class ActionGenerator:
    def __init__(self, llm_interface: LLMInterface, guidance_path=None, bridge=None):
        self.llm = llm_interface
        self.bridge = bridge
        
        # [Hard Prior 1] 基础权重配置
        self.base_weights = {
            ActionType.CONTINUE: 1.0,
            ActionType.SEARCH: 1.0,
            ActionType.REQUERY: 0.5,
            ActionType.TRIM: 0.5,
            ActionType.SPLIT: 0.2, # 结构调整慎用
            ActionType.MERGE: 0.2,
            ActionType.GENERATE_SUNO: 0.1, # Cost高，慎用
            ActionType.RELAX_CONSTRAINT: 0.1,
            ActionType.SHIFT_ALIGN: 0.3
        }

        # [Soft Prior] 症状 -> 目标 -> 动作模板 (注入到 Prompt 中)
        if guidance_path is None:
            # 默认在同级目录下寻找 guidance.json
            guidance_path = os.path.join(os.path.dirname(__file__), "guidance.json")
            
        try:
            with open(guidance_path, 'r', encoding='utf-8') as f:
                self.guidance_templates = json.load(f)
            print(f"[ActionGen] Loaded guidance templates from {Path(guidance_path).name}")
        except Exception as e:
            print(f"[ActionGen] Warning: Could not load guidance.json: {e}")
            self.guidance_templates = {}

    def get_action_priors(self, state: AgentState) -> Dict[ActionType, float]:
        """
        [Hard Logic] 计算每个动作的采样概率权重
        """
        weights = self.base_weights.copy()
        tags = state.failure_tags
        
        # 初始状态特殊处理
        if state.current_movement_index not in state.assigned_tracks and not tags:
            return {
                ActionType.SEARCH: 2.0,
                ActionType.GENERATE_SUNO: 0.5,
                ActionType.MERGE: 0.5 
            }

        # 根据 Failure Tags 调整权重 (Hard Constraints)
        if tags.get('SEM_LOW'):
            weights[ActionType.REQUERY] *= 3.0
            weights[ActionType.SEARCH] *= 2.0
            weights[ActionType.CONTINUE] *= 0.1

        if tags.get('FILTER_KILL'):
            weights[ActionType.RELAX_CONSTRAINT] *= 10.0 # 极大提升
            weights[ActionType.SEARCH] *= 1.5

        if tags.get('HARM_BAD'):
            weights[ActionType.SEARCH] *= 2.0
            weights[ActionType.CONTINUE] *= 0.0 

        if tags.get('SYNC_BAD'):
            weights[ActionType.SHIFT_ALIGN] *= 5.0
            weights[ActionType.TRIM] *= 3.0
            
        if tags.get('TOO_LONG_VAR'):
            weights[ActionType.SPLIT] *= 15.0 # 强烈建议切分
            
        if tags.get('TOO_SHORT_FRAGMENT'):
            weights[ActionType.MERGE] *= 10.0
            weights[ActionType.CONTINUE] *= 2.0
            weights[ActionType.SEARCH] *= 0.1

        return weights

    def propose(self, state: AgentState, k=3) -> List[Dict]:
        """
        融合 Hard Priors 和 Soft Priors (LLM) 进行决策
        """
        # 1. 计算 Hard Priors
        priors = self.get_action_priors(state)
        
        # 2. 筛选 Top-K 动作类型 (作为 LLM 的候选菜单)
        # 我们不把所有动作都扔给 LLM，只扔权重高的，减少幻觉
        sorted_actions = sorted(priors.items(), key=lambda x: x[1], reverse=True)
        top_actions_types = [a[0] for a in sorted_actions if a[1] > 0.3][:4] # 取前4个且权重>0.3的
        
        # 如果 SPLIT 权重极高 (>10)，强制只推荐 SPLIT，让 LLM 专注于找时间点
        if priors.get(ActionType.SPLIT, 0) > 8.0:
            top_actions_types = [ActionType.SPLIT]

        # 3. 构造 Prompt (注入 Soft Priors)
        system_prompt = self._construct_system_prompt()
        user_prompt = self._construct_user_prompt(state, top_actions_types)

        # 4. LLM 推理
        try:
            response = self.llm.chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.5 # 降低随机性，遵循先验
            )
            actions = self._parse_response(response)
        except Exception as e:
            print(f"[ActionGen] LLM Failed: {e}")
            actions = []

        # 5. 兜底机制：如果 LLM 没生成有效动作，直接用 Hard Prior Top-1
        if not actions:
            best_action = sorted_actions[0][0]
            actions = [self._fallback_rule_action(best_action, state)]

        return actions[:k]

    def _construct_system_prompt(self) -> str:
        return """You are the Director Agent. Your goal is to fix soundtrack issues based on specific failure tags.
You must output a JSON list of actions.
"""

    def _construct_user_prompt(self, state: AgentState, allowed_actions: List[ActionType]) -> str:
        idx = state.current_movement_index
        mov = state.movements[idx]
        
        # 基础上下文
        content = f"Movement: {mov.id} ({mov.start_time}-{mov.end_time}s)\n"
        content += f"Visual Summary: {mov.visual_summary}\n"
        
        # 注入感知层数据 (shots) 辅助参数填充
        if mov.shots:
            shots_str = ", ".join([f"{s.end_sec:.2f}s" for s in mov.shots])
            content += f"Shot Boundaries (Potential Split Points): [{shots_str}]\n"

        # 诊断信息 (Soft Priors Injection)
        tags = [k for k, v in state.failure_tags.items() if v]
        content += f"\nDetected Symptoms (Failure Tags): {tags}\n"
        
        if tags:
            content += "Expert Guidelines:\n"
            for t in tags:
                if t in self.guidance_templates:
                    guide = self.guidance_templates[t]
                    content += f"- [{t}]: {guide['symptom']} -> Goal: {guide['goal']}\n"
                    content += f"  Recommended Action: {guide['action']}\n"
                    content += f"  Avoid: {guide['anti_pattern']}\n"

        # 强制约束
        content += f"\nAllowed Action Types: {[a.name for a in allowed_actions]}\n"
        content += "Please generate 1-3 concrete actions strictly choosing from the Allowed Types. Fill specific parameters (e.g. timestamp for SPLIT, query for REQUERY)."
        
        return content

    def _fallback_rule_action(self, action_type: ActionType, state: AgentState) -> Dict:
        """当 LLM 挂掉时，根据 ActionType 生成默认参数"""
        mov = state.movements[state.current_movement_index]
        
        if action_type == ActionType.SPLIT:
            # 默认切中间
            return {"type": ActionType.SPLIT, "params": {"timestamp": (mov.start_time+mov.end_time)/2}}
        
        if action_type == ActionType.SEARCH or action_type == ActionType.REQUERY:
            return {"type": ActionType.SEARCH, "params": {"query": mov.visual_summary}}
            
        if action_type == ActionType.RELAX_CONSTRAINT:
            return {"type": ActionType.RELAX_CONSTRAINT, "params": {}}
            
        return {"type": ActionType.CONTINUE, "params": {}}

    def _parse_response(self, text: str) -> List[Dict]:
        # (同之前的解析逻辑，略)
        try:
            match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if match: text = match.group(1)
            data = json.loads(text)
            valid = []
            for item in data:
                if hasattr(ActionType, item['type']):
                    item['type'] = ActionType[item['type']]
                    valid.append(item)
            return valid
        except:
            return []