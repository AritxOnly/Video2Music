from dataclasses import asdict
from typing import Dict, Any, List, Optional
from agent.state import ActionType, AgentState, Track, Movement
from toolbox.interface import Toolbox

class AgentToolboxBridge:
    """
    连接 Agent (Objects) 和 Toolbox (JSON) 的桥梁。
    MicroPlanner -> Bridge -> Toolbox -> Real Implementation
    """
    def __init__(self, toolbox: Toolbox):
        self.toolbox = toolbox
        
        # 建立 Enum 到 String UID 的映射
        # 确保这些 key 和 toolbox.json 里的 uid 一致
        self.action_map = {
            ActionType.SEARCH: "retrieval.search",
            ActionType.REQUERY: "retrieval.requery",
            ActionType.RELAX_CONSTRAINT: "retrieval.relax_constraint",
            ActionType.TRIM: "edit.trim",
            ActionType.SPLIT: "struct.split",
            ActionType.MERGE: "struct.merge",
            ActionType.SHIFT_ALIGN: "edit.shift_align",
            ActionType.CONTINUE: "edit.continue",
            ActionType.GENERATE_SUNO: "gen.suno"
        }

    def execute(self, action_type: ActionType, params: Dict[str, Any], movement: Movement, state: AgentState) -> Any:
        """
        MicroPlanner 调用的标准接口
        """
        # 1. 获取 UID
        uid = self.action_map.get(action_type)
        if not uid:
            print(f"[Bridge] Error: No UID mapping for {action_type}")
            return None

        # 2. 构造参数 (Context Injection)
        tool_args = params.copy()
        tool_args.update({
            "movement_id": movement.id,
            "start_time": movement.start_time,
            "end_time": movement.end_time,
            "visual_summary": movement.visual_summary,
            "shots": [asdict(s) for s in movement.shots]
        })
        
        # 3. 注入上下文信息 (Previous Key) 用于 Search/Relax
        if state.current_movement_index > 0:
            prev_idx = state.current_movement_index - 1
            if prev_idx in state.assigned_tracks:
                tool_args["prev_track_key"] = state.assigned_tracks[prev_idx].meta.get('key')
                
        # 4. 注入下一个 Movement 信息 (用于 Merge)
        if action_type == ActionType.MERGE:
            next_idx = state.current_movement_index + 1
            if next_idx < len(state.movements):
                next_mov = state.movements[next_idx]
                tool_args["next_end_time"] = next_mov.end_time
                tool_args["next_summary"] = next_mov.visual_summary
                tool_args["next_shots"] = [asdict(s) for s in next_mov.shots]

        # 5. 执行 Toolbox
        print(f"  [Bridge] Calling {uid} with {tool_args}")
        result = self.toolbox.execute(uid, tool_args)

        if not result.ok:
            print(f"  [Bridge] Tool execution failed: {result.error}")
            return None

        # 6. 结果反序列化 (JSON -> Object)
        return self._parse_result(action_type, result.data, movement)

    def _parse_result(self, action_type: ActionType, data: Dict, context_movement: Movement) -> Any:
        """
        将 Toolbox 返回的纯字典转换为 Agent 需要的对象 (Track, List[Movement] 等)
        """
        # --- 针对 Search/Requery/Generate 返回 Track ---
        if action_type in [ActionType.SEARCH, ActionType.REQUERY, ActionType.GENERATE_SUNO]:
            if not data: return None
            # 假设工具返回的是 {"track_id": "...", "meta": {...}, "url": "..."}
            return Track(
                id=data.get("track_id", "unknown"),
                source=data.get("source", "library"),
                duration=data.get("duration", 0.0),
                meta=data.get("meta", {}),
                play_start=data.get("play_start", 0.0),
                play_duration=data.get("play_duration", 0.0)
            )

        # --- 针对 Split 返回 List[Movement] ---
        if action_type == ActionType.SPLIT:
            # 假设工具返回 {"movements": [{"start": 0, "end": 5}, ...]}
            raw_movs = data.get("movements", [])
            new_movements = []
            for i, rm in enumerate(raw_movs):
                new_movements.append(Movement(
                    id=f"{context_movement.id}_split_{i}",
                    start_time=rm['start'],
                    end_time=rm['end'],
                    shots=rm.get('shots', []), # 理想情况下 Split 工具应该重新分配 shots
                    visual_summary=rm.get('summary', context_movement.visual_summary)
                ))
            return new_movements

        # --- 针对 Merge 返回 Movement ---
        if action_type == ActionType.MERGE:
             # 假设工具返回 {"merged_movement": {...}}
             rm = data.get("merged_movement", {})
             return Movement(
                id=f"{context_movement.id}_merged",
                start_time=rm.get('start', context_movement.start_time),
                end_time=rm.get('end', context_movement.end_time),
                shots=rm.get('shots', []),
                visual_summary=rm.get('summary', "")
             )

        # --- 针对 Trim/Shift/Relax ---
        # 此时可能只需要返回 True 或者更新后的 meta，视 micro_planner 逻辑而定
        # 这里直接返回 data 供 planner 处理
        return data