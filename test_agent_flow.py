import unittest
import sys
import os
from typing import List, Dict, Any

sys.path.append(os.getcwd())

from agent.state import AgentState, Movement, Track, ActionType
from agent.critic import Critic
from agent.macro_planner import MacroPlanner
from agent.micro_planner import MicroPlanner

# ==========================================
# 1. Mocks (模拟外部依赖)
# ==========================================

class MockLLM:
    """模拟 LLM，用于 MacroPlanner 提议"""
    def propose_structures(self, shots, n=3):
        # 模拟返回一个简单的结构建议
        return [[
            {'start': 0.0, 'end': 10.0, 'summary': 'Intro'},
            {'start': 10.0, 'end': 20.0, 'summary': 'Climax'}
        ]]

class MockToolbox:
    """模拟 Toolbox，用于执行具体动作"""
    def execute(self, action_type: ActionType, params: Dict, movement: Movement):
        print(f"  [Toolbox] Executing {action_type.name} with params {params}")
        
        # 模拟 SEARCH: 返回一首假歌
        if action_type == ActionType.SEARCH:
            return Track(
                id=f"track_{params.get('query', 'unknown')}",
                source="mock_lib",
                duration=30.0,
                meta={
                    'bpm': 120, 
                    'key': 'C Major',
                    'sem_score': 0.8, # 模拟高语义匹配
                    'harm_score': 0.9,
                    'sync_score': 0.7,
                    'flow_score': 0.6
                }
            )
        
        # 模拟 SPLIT: 返回两个新的 Movement
        if action_type == ActionType.SPLIT:
            mid_point = (movement.start_time + movement.end_time) / 2
            return [
                Movement("split_1", movement.start_time, mid_point, []),
                Movement("split_2", mid_point, movement.end_time, [])
            ]
            
        return None

class MockActionGenerator:
    """模拟 ActionGenerator，保证测试路径确定"""
    def __init__(self):
        pass
        
    def propose(self, state: AgentState, k=3) -> List[Dict]:
        """
        简单的确定性策略：
        1. 如果当前 Movement 还没配乐 -> SEARCH
        2. 如果已经配乐 -> CONTINUE (为了结束循环)
        """
        current_idx = state.current_movement_index
        
        # 如果当前位置没有 track，尝试搜索
        if current_idx not in state.assigned_tracks:
            return [{
                'type': ActionType.SEARCH,
                'params': {'query': 'Happy Rock'}
            }]
        
        # 这里模拟 Beam Search 结束或者进入下一段逻辑
        # 在真实 MicroPlanner 中，配好乐后 state.current_movement_index 会 +1
        # 所以理论上只要状态没终结，Mock 就会被调用。
        # 如果走到这里，说明逻辑有误或需要特殊处理，这里简单返回 SEARCH 以便测试继续
        return [{
            'type': ActionType.SEARCH, 
            'params': {'query': 'Next Song'}
        }]

# ==========================================
# 2. Test Cases
# ==========================================

class TestAgentSystem(unittest.TestCase):
    
    def setUp(self):
        # 准备一些假的 Shots 数据
        self.dummy_shots = [
            {'start': 0, 'end': 5, 'desc': 'man walking'},
            {'start': 5, 'end': 10, 'desc': 'man running'},
            {'start': 10, 'end': 20, 'desc': 'car chase'}
        ]
        self.total_duration = 20.0
        
        # 初始化各个组件
        self.mock_llm = MockLLM()
        self.mock_toolbox = MockToolbox()
        self.critic = Critic()
        self.macro_planner = MacroPlanner(self.mock_llm)
        # 注意：这里我们注入 MockActionGenerator
        self.micro_planner = MicroPlanner(MockActionGenerator(), self.critic, self.mock_toolbox)

    def test_1_macro_planning(self):
        """测试宏观规划能否正确切分乐章"""
        print("\n=== Test 1: Macro Planning ===")
        movements = self.macro_planner.plan(self.dummy_shots, self.total_duration)
        
        print(f"Generated {len(movements)} movements.")
        for m in movements:
            print(f" - Movement {m.id}: {m.start_time}s -> {m.end_time}s")
            
        self.assertTrue(len(movements) > 0)
        self.assertIsInstance(movements[0], Movement)

    def test_2_critic_evaluation(self):
        """测试评分系统"""
        print("\n=== Test 2: Critic Score ===")
        state = AgentState(movements=[], current_movement_index=0)
        mov = Movement("test_m", 0, 10, [])
        track = Track("t1", "lib", 30, {'sem_score': 0.1}) # 故意给低分
        
        score = self.critic.evaluate(state, ActionType.SEARCH, mov, track)
        print(f"Score: {score}")
        print(f"Failure Tags: {state.failure_tags}")
        
        # 验证 Failure Tags 是否被触发 SEM_LOW
        self.assertIn('SEM_LOW', state.failure_tags)
        self.assertLess(score, 2.0)

    def test_3_micro_flow(self):
        """测试 MicroPlanner 的 Beam Search 流程"""
        print("\n=== Test 3: Micro Planner Loop ===")
        
        # 1. 先跑 Macro 得到 movements
        movements = self.macro_planner.plan(self.dummy_shots, self.total_duration)
        initial_state = AgentState(movements=movements)
        
        # 2. 跑 Micro Plan
        best_state = self.micro_planner.plan(initial_state)
        
        print(f"\nResult State:")
        print(f"Total Score: {best_state.total_score}")
        print(f"Assigned Tracks: {len(best_state.assigned_tracks)}")
        
        # 打印决策历史
        print("Action History:")
        for step in best_state.action_history:
            print(f" -> Step {step['step']}: {step['action']} | Score: {step['score']:.2f}")

        # 验证
        # 应该为所有 movements 都分配了 track
        self.assertEqual(len(best_state.assigned_tracks), len(movements))
        self.assertTrue(best_state.is_terminal)

if __name__ == '__main__':
    unittest.main()