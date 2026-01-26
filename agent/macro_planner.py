from typing import List, Dict
from agent.state import Movement
import json

class MacroPlanner:
    def __init__(self, llm_interface):
        self.llm = llm_interface

    def plan(self, shots: List[Dict], total_duration: float) -> List[Movement]:
        """
        入口函数：结合 LLM 提议和规则基线，选择最佳的分段方案
        """
        candidates = []

        # 1. LLM Proposals 
        try:
            llm_plans = self._propose_by_llm(shots)
            candidates.extend(llm_plans)
        except Exception as e:
            print(f"LLM Macro Planning failed: {e}")

        # 2. Rule Baseline 
        # 兜底方案：简单地按时间均匀切分或基于视觉转场切分
        rule_plan = self._propose_by_rule(shots, total_duration)
        candidates.append(rule_plan)

        # 3. Critique & Selection 
        best_plan = max(candidates, key=lambda p: self._evaluate_structure(p, shots))
        
        # 4. 转换为 Movement 对象
        movements = []
        for i, segment in enumerate(best_plan):
            movements.append(Movement(
                id=f"mov_{i}",
                start_time=segment['start'],
                end_time=segment['end'],
                shots=segment.get('shots', []),
                visual_summary=segment.get('summary', "")
            ))
            
        return movements

    def _propose_by_llm(self, shots: List[Dict]) -> List[List[Dict]]:
        """调用 LLM 生成 3 种可能的划分方案"""
        # 这里构造 Prompt，传入 shots 的时间戳和描述
        # 预期 LLM 返回 JSON 格式的分段列表
        # return [[{'start': 0, 'end': 10, ...}, ...], ...]
        return [] # Placeholder

    def _propose_by_rule(self, shots: List[Dict], total_duration: float) -> List[Dict]:
        """基于 visual_energy 或固定时长切分"""
        # 简单实现：每 15 秒切一段
        segments = []
        cursor = 0.0
        while cursor < total_duration:
            end = min(cursor + 15.0, total_duration)
            segments.append({'start': cursor, 'end': end, 'summary': 'Auto segmented'})
            cursor = end
        return segments

    def _evaluate_structure(self, plan: List[Dict], shots: List[Dict]) -> float:
        """
        Macro Score 计算
        Score = w1*Consistency + w2*BoundaryContrast + Reg(Len)
        """
        score = 0.0
        # 1. 长度惩罚：太碎或太长都不好
        for seg in plan:
            duration = seg['end'] - seg['start']
            if duration < 5: score -= 1.0
            if duration > 40: score -= 1.0
            
        # 2. 边界对齐奖励 (假设 shot 边界是硬切点)
        # 如果 segment 的 end_time 恰好是某个 shot 的 end_time，加分
        return score