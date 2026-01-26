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
        """
        执行 Beam Search，为每个 Movement 寻找最佳配乐
        """
        # 初始 Beam
        beams = [initial_state]

        # 循环直到所有 Movement 都处理完
        # 注意：由于 Split/Merge 操作，movements 的数量是动态的，
        # 所以不能简单用 for i in range(len(movements))
        
        max_steps = 20 # 防止无限循环
        step_count = 0
        
        while step_count < max_steps:
            candidates = []
            
            all_beams_terminal = True
            
            for state in beams:
                if state.is_terminal:
                    candidates.append(state)
                    continue
                
                all_beams_terminal = False
                
                # 当前要处理的 Movement
                current_mov_idx = state.current_movement_index
                current_mov = state.movements[current_mov_idx]
                
                # 1. Action Proposal 
                # 根据 Failure Tags 提议动作 (e.g., SEARCH, SPLIT...)
                proposed_actions = self.action_gen.propose(state, k=3)
                
                for action_dict in proposed_actions:
                    action_type = action_dict['type']
                    params = action_dict.get('params', {})
                    
                    # 2. Execution (Tool Use)
                    # 调用 toolbox 执行具体操作
                    # 比如 toolbox.search_music(query) -> Track
                    execution_result = self.toolbox.execute(action_type, params, current_mov)
                    
                    if not execution_result:
                        continue
                        
                    # 3. State Update & Scoring
                    new_state = state.clone()
                    
                    # 根据 Action 类型更新状态
                    track = None
                    if action_type in [ActionType.SEARCH, ActionType.GENERATE_SUNO]:
                        track = execution_result # 假设返回的是 Track 对象
                        new_state.assigned_tracks[current_mov_idx] = track
                        # 只有成功选曲后，指针才前进
                        new_state.current_movement_index += 1
                        
                    elif action_type == ActionType.SPLIT:
                        # 结构改变，movements 列表变长，指针不变（需重新为拆分后的第一段配乐）
                        new_movements = execution_result # Expecting [m1, m2]
                        new_state.movements[current_mov_idx] = new_movements[0]
                        new_state.movements.insert(current_mov_idx + 1, new_movements[1])
                        
                    elif action_type == ActionType.MERGE:
                        # 结构改变，movements 列表变短
                        new_mov = execution_result
                        new_state.movements[current_mov_idx] = new_mov
                        del new_state.movements[current_mov_idx + 1]
                    
                    # 计算得分
                    step_score = self.critic.evaluate(
                        new_state, action_type, current_mov, track if track else Track("temp", "none", 0, {})
                    )
                    
                    new_state.total_score += step_score
                    new_state.action_history.append({
                        'step': step_count,
                        'action': action_type.value,
                        'params': params,
                        'score': step_score,
                        'tags': new_state.failure_tags.copy()
                    })
                    
                    candidates.append(new_state)
            
            if all_beams_terminal:
                break
                
            # 4. Selection (Pruning)
            # 按总分排序，保留 Top K
            candidates.sort(key=lambda s: s.total_score, reverse=True)
            beams = candidates[:self.beam_width]
            step_count += 1
            
        return beams[0] # 返回得分最高的最终状态