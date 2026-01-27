from typing import List
import copy
from agent.state import AgentState, ActionType, Track
from agent.action_generator import ActionGenerator
from agent.critic import Critic

class MicroPlanner:
    def __init__(self, action_gen: ActionGenerator, critic: Critic, toolbox):
        self.action_gen = action_gen
        self.critic = critic
        self.toolbox = toolbox
        self.beam_width = 3  # 

    def plan(self, initial_state: AgentState) -> AgentState:
        beams = [initial_state]
        
        # 增加步数限制，防止死循环
        max_steps = len(initial_state.movements) * 3 
        step_count = 0
        
        print(f"[MicroPlanner] Start planning for {len(initial_state.movements)} movements...")
        
        while step_count < max_steps:
            candidates = []
            all_beams_terminal = True
            
            for state in beams:
                if state.is_terminal:
                    candidates.append(state)
                    continue
                
                all_beams_terminal = False
                
                # 获取当前要处理的 Movement
                current_mov_idx = state.current_movement_index
                
                # [安全检查] 防止越界
                if current_mov_idx >= len(state.movements):
                    state.current_movement_index = len(state.movements) # 强制终结
                    candidates.append(state)
                    continue
                    
                current_mov = state.movements[current_mov_idx]
                
                # 1. 动作提议
                # 如果当前状态没有 failure tags，ActionGenerator 可能会返回 CONTINUE
                # 但如果是第一次进入该 Movement，必须强制 SEARCH/GENERATE
                if current_mov_idx not in state.assigned_tracks and not state.failure_tags:
                    # 强制冷启动逻辑，不依赖 LLM 随机性
                    proposed_actions = [{
                        "type": ActionType.SEARCH,
                        "params": {"query": current_mov.visual_summary}
                    }]
                else:
                    proposed_actions = self.action_gen.propose(state, k=3)
                
                if not proposed_actions:
                    print(f"[Planner] No actions proposed for step {step_count}. Skipping beam.")
                    continue

                for action_dict in proposed_actions:
                    action_type = action_dict['type']
                    params = action_dict.get('params', {})
                    
                    # 2. 执行工具
                    # 务必传入 state
                    execution_result = self.toolbox.execute(
                        action_type, params, current_mov, state
                    )
                    
                    # 3. 状态克隆与更新
                    new_state = state.clone()
                    
                    track = None
                    score = 0.0
                    
                    # === [CRITICAL FIX] 状态更新逻辑 ===
                    
                    # case A: 选曲成功 (SEARCH, REQUERY, GENERATE)
                    if action_type in [ActionType.SEARCH, ActionType.REQUERY, ActionType.GENERATE_SUNO]:
                        if execution_result: # 确保找到了结果
                            track = execution_result
                            new_state.assigned_tracks[current_mov_idx] = track
                            # 只有选到了歌，指针才前进！
                            new_state.current_movement_index += 1
                            # 清除之前的 Failure Tags，因为我们已经采取了行动
                            new_state.failure_tags = {} 
                        else:
                            # 搜索失败，保持原地不动，但 Critic 会打低分
                            pass

                    # case B: 编辑操作 (CONTINUE)
                    elif action_type == ActionType.CONTINUE:
                        # 延续上一首
                        if current_mov_idx > 0:
                            prev_track = new_state.assigned_tracks.get(current_mov_idx - 1)
                            if prev_track:
                                new_state.assigned_tracks[current_mov_idx] = prev_track
                                new_state.current_movement_index += 1
                                new_state.failure_tags = {}
                                track = prev_track

                    # case C: 结构调整 (SPLIT)
                    elif action_type == ActionType.SPLIT:
                        new_movs = execution_result # [m1, m2]
                        if new_movs:
                            new_state.movements[current_mov_idx] = new_movs[0]
                            new_state.movements.insert(current_mov_idx + 1, new_movs[1])
                            # 指针不前进，因为 m1 还需要配乐
                            new_state.failure_tags = {}

                    # case D: 结构调整 (MERGE)
                    elif action_type == ActionType.MERGE:
                        new_mov = execution_result
                        if new_mov and (current_mov_idx + 1 < len(new_state.movements)):
                            new_state.movements[current_mov_idx] = new_mov
                            del new_state.movements[current_mov_idx + 1]
                            # 指针不前进，因为合并后的 movement 需要重新配乐
                            new_state.failure_tags = {}

                    # 4. 评分
                    # 如果刚才没选到 track (比如 SPLIT 或 搜索失败)，造一个 dummy track 用于评分
                    eval_track = track if track else Track("temp", "none", 0, {})
                    
                    step_score = self.critic.evaluate(
                        new_state, action_type, current_mov, eval_track
                    )
                    
                    new_state.total_score += step_score
                    
                    # 记录历史
                    # [Fix] 存储 action_type.name 字符串，避免打印时报错
                    new_state.action_history.append({
                        'step': step_count,
                        'movement_idx': current_mov_idx,
                        'action': action_type.name if hasattr(action_type, 'name') else str(action_type), 
                        'params': params,
                        'score': step_score,
                        'tags': new_state.failure_tags.copy()
                    })
                    
                    candidates.append(new_state)
            
            if all_beams_terminal:
                break
            
            if not candidates:
                print("[Planner] All beams died (empty candidates). Stopping.")
                break

            # 5. 剪枝 (Selection)
            candidates.sort(key=lambda s: s.total_score, reverse=True)
            beams = candidates[:self.beam_width]
            
            # 打印调试信息
            best = beams[0]
            print(f"  >> Step {step_count}: Best Score={best.total_score:.2f} | Mov={best.current_movement_index}/{len(best.movements)} | Action={best.action_history[-1]['action']}")
            
            step_count += 1
            
        return beams[0]