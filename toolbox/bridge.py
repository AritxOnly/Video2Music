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
        
        # Mapping Enum to String UIDs in toolbox.json
        self.action_map = {
            ActionType.SEARCH: "retrieval.search",
            ActionType.REQUERY: "retrieval.requery",
            ActionType.RELAX_CONSTRAINT: "retrieval.relax_constraint",
            ActionType.TRIM: "edit.trim",
            ActionType.SPLIT: "struct.split",
            ActionType.MERGE: "struct.merge",
            ActionType.SHIFT_ALIGN: "edit.shift_align",
            # ActionType.CONTINUE is handled by planner logic, usually not via tool
            ActionType.CONTINUE: "edit.continue", 
            ActionType.GENERATE_SUNO: "gen.suno"
        }

    def execute(self, action_type: ActionType, data: Any, movement: Movement, state: AgentState) -> Any:
        """
        Standard interface called by MicroPlanner.
        """
        if not data:
            data = {}
        
        # 1. Get Tool UID
        uid = self.action_map.get(action_type)
        if not uid:
            print(f"[Bridge] Error: No UID mapping for {action_type}")
            return None

        # 2. Construct Arguments (Context Injection)
        tool_args = data.copy() if isinstance(data, dict) else {}
        
        # Inject current movement context
        tool_args.update({
            "movement_id": movement.id,
            "start_time": movement.start_time,
            "end_time": movement.end_time,
            "visual_summary": movement.visual_summary,
            "shots": [asdict(s) for s in movement.shots] if movement.shots else []
        })
        
        # 3. Inject Previous Key (for Search/Relax harmony constraints)
        if state.current_movement_index > 0:
            prev_idx = state.current_movement_index - 1
            prev_track = state.assigned_tracks.get(prev_idx)
            if prev_track:
                # Safe access to key in meta
                meta = prev_track.meta or {}
                tool_args["prev_track_key"] = meta.get('key')
                
        # 4. Inject Next Movement Info (for Merge)
        if action_type == ActionType.MERGE:
            next_idx = state.current_movement_index + 1
            if next_idx < len(state.movements):
                next_mov = state.movements[next_idx]
                tool_args["next_end_time"] = next_mov.end_time
                tool_args["next_summary"] = next_mov.visual_summary
                tool_args["next_shots"] = [asdict(s) for s in next_mov.shots] if next_mov.shots else []

        # 5. Execute via Toolbox
        # print(f"  [Bridge] Calling {uid}...") # Optional debug print
        result = self.toolbox.execute(uid, tool_args)

        if not result.ok:
            print(f"  [Bridge] Tool execution failed: {result.error}")
            return None

        # 6. Parse Result (Object Pass-through or JSON -> Object)
        return self._parse_result(action_type, result.data, movement)

    def _parse_result(self, action_type: ActionType, data: Any, context_movement: Movement) -> Any:
        """
        Converts Toolbox results into Agent objects.
        Crucially: If 'data' is already an object (Track/Movement), it returns it directly.
        """
        if not data:
            return None

        # === 1. Direct Pass-through (The Fix) ===
        # If the tool implementation already returned the correct object type, use it.
        if isinstance(data, Track):
            return data
        if isinstance(data, Movement):
            return data
        if isinstance(data, list):
            # Check if it's a list of Movements (for SPLIT)
            if data and isinstance(data[0], Movement):
                return data
            # Check if it's a list of Tracks (rare, but possible)
            if data and isinstance(data[0], Track):
                return data

        # === 2. Dictionary Parsing (Legacy / Remote Tools Support) ===
        if not isinstance(data, dict):
            # If it's not an object and not a dict, we can't parse it.
            return data

        # --- Parse Track ---
        if action_type in [ActionType.SEARCH, ActionType.REQUERY, ActionType.GENERATE_SUNO, ActionType.RELAX_CONSTRAINT]:
            return Track(
                id=str(data.get("track_id") or data.get("id") or "unknown"),
                source=str(data.get("source", "library")),
                duration=float(data.get("duration", 0.0)),
                meta=data.get("meta", {}),
                play_start=float(data.get("play_start", 0.0)),
                play_duration=float(data.get("play_duration", 0.0))
            )

        # --- Parse Split (List[Movement]) ---
        if action_type == ActionType.SPLIT:
            raw_movs = data.get("movements", [])
            new_movements = []
            for i, rm in enumerate(raw_movs):
                # Need to handle shots carefully here if they are dicts
                shots_data = rm.get('shots', [])
                # (Assuming simple conversion or that shots are already compatible)
                
                new_movements.append(Movement(
                    id=f"{context_movement.id}_split_{i}",
                    start_time=float(rm.get('start', 0.0)),
                    end_time=float(rm.get('end', 0.0)),
                    shots=[], # Placeholder, ideally parse shots_data back to objects if needed
                    visual_summary=rm.get('summary', context_movement.visual_summary)
                ))
            return new_movements

        # --- Parse Merge (Movement) ---
        if action_type == ActionType.MERGE:
             rm = data.get("merged_movement", {})
             if not rm: return None
             return Movement(
                id=f"{context_movement.id}_merged",
                start_time=float(rm.get('start', context_movement.start_time)),
                end_time=float(rm.get('end', context_movement.end_time)),
                shots=[], 
                visual_summary=rm.get('summary', "")
             )

        # --- Default: Return raw data ---
        return data