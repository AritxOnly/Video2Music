import json
import re
from typing import List, Dict, Any
from agent.state import Movement, SegmentSemantics
from llm.interface import LLMInterface

class MacroPlanner:
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def plan(self, raw_movements: List[Movement], min_duration: float = 15.0) -> List[Movement]:
        """
        入口函数：将碎片化的 raw_movements 聚合为结构化的 structural_movements
        """
        if not raw_movements:
            return []

        print(f"[MacroPlanner] Analyzing {len(raw_movements)} shots to form a structure...")

        # 1. 尝试使用 LLM 进行语义聚类
        try:
            grouped_indices = self._propose_by_llm(raw_movements)
            # 校验 LLM 输出是否覆盖了所有 shot
            if not self._validate_indices(grouped_indices, len(raw_movements)):
                print("[MacroPlanner] LLM structure invalid. Fallback to rule.")
                grouped_indices = self._propose_by_rule(raw_movements, min_duration)
        except Exception as e:
            print(f"[MacroPlanner] LLM Error: {e}. Fallback to rule.")
            grouped_indices = self._propose_by_rule(raw_movements, min_duration)

        # 2. 根据聚类结果重组 Movement 对象
        structured_movements = []
        for group in grouped_indices:
            # group 是一个 index 列表，例如 [0, 1, 2, 3]
            sub_movements = [raw_movements[i] for i in group]
            merged_mov = self._merge_movements(sub_movements)
            structured_movements.append(merged_mov)

        print(f"[MacroPlanner] Compressed {len(raw_movements)} shots -> {len(structured_movements)} movements.")
        return structured_movements

    def _merge_movements(self, movs: List[Movement]) -> Movement:
        """核心工具：将多个微小的 Movement 合并为一个大的"""
        if not movs: return None
        
        first = movs[0]
        last = movs[-1]
        
        # 1. 合并所有的 shots
        all_shots = []
        for m in movs:
            all_shots.extend(m.shots)
            
        # 2. 合并 Summary (简单拼接或提取主要情绪)
        # 取出现频率最高的 Mood 作为主 Mood
        all_moods = [s.mood for s in all_shots if s.mood]
        main_mood = max(set(all_moods), key=all_moods.count) if all_moods else "Unknown"
        
        # 提取 unique activities
        activities = set(s.activity for s in all_shots if s.activity)
        act_str = ", ".join(list(activities)[:3])
        
        # 构造新的 ID 和 Summary
        new_id = f"mov_merged_{first.id.split('_')[1]}_to_{last.id.split('_')[1]}"
        new_summary = f"Mood: {main_mood}. Main Actions: {act_str}. (Contains {len(movs)} shots)"

        return Movement(
            id=new_id,
            start_time=first.start_time,
            end_time=last.end_time,
            shots=all_shots,
            visual_summary=new_summary
        )

    def _propose_by_rule(self, movements: List[Movement], min_duration: float) -> List[List[int]]:
        """
        兜底规则：贪婪合并，直到每段长度达到 min_duration
        """
        groups = []
        current_group = []
        current_duration = 0.0
        
        for i, mov in enumerate(movements):
            dur = mov.end_time - mov.start_time
            
            # 如果当前 Mood 和上一段差异过大（可选），强制切分
            # 这里简单起见只看时长
            
            current_group.append(i)
            current_duration += dur
            
            # 如果积累够长了，封包
            # 但如果是最后一段，尽量不要剩下单独的一个短片，除非它本身够长
            if current_duration >= min_duration:
                groups.append(current_group)
                current_group = []
                current_duration = 0.0
        
        # 处理剩余的尾巴
        if current_group:
            # 如果尾巴太短，且前面有组，就合并到前面去
            if current_duration < 5.0 and groups:
                groups[-1].extend(current_group)
            else:
                groups.append(current_group)
                
        return groups

    def _propose_by_llm(self, movements: List[Movement]) -> List[List[int]]:
        """
        让 LLM 阅读分镜表，进行结构划分
        """
        # 1. 构造剧本大纲
        storyboard = ""
        for i, mov in enumerate(movements):
            storyboard += f"Shot {i} ({mov.end_time - mov.start_time:.1f}s): {mov.visual_summary}\n"

        prompt = f"""
You are a Video Director. Below is a sequence of video shots (VLM output).
Your task is to group these shots into 3-6 coherent "Movements" (Musical Sections) based on narrative flow and emotion.

Input Storyboard:
{storyboard}

Requirements:
1. Merge adjacent shots that share similar moods or activities.
2. A Movement should generally be 15s - 60s long.
3. Do not leave single short shots ( < 5s) isolated unless they are dramatic punctuation.
4. Output a JSON list of lists, where each inner list contains the Shot Indexes for that movement.

Example Output:
[[0, 1, 2], [3, 4, 5, 6], [7, 8]]
"""
        response = self.llm.chat_completion(
            system_prompt="You are an expert film editor.",
            user_prompt=prompt,
            temperature=0.3 # 需要比较理性的结构
        )
        
        return self._parse_json(response)

    def _parse_json(self, text: str) -> List[List[int]]:
        try:
            match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if match: text = match.group(1)
            return json.loads(text)
        except:
            return []

    def _validate_indices(self, groups: List[List[int]], total_shots: int) -> bool:
        """确保 LLM 没有漏掉镜头或幻觉出不存在的镜头"""
        if not groups: return False
        flat = [i for g in groups for i in g]
        # 检查是否连续且覆盖 0 到 N-1
        if len(flat) != total_shots: return False
        if sorted(flat) != list(range(total_shots)): return False
        return True