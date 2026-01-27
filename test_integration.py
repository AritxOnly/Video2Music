import unittest
import os
import sys
from pathlib import Path

# 路径设置
sys.path.append(os.getcwd())

from agent.state import AgentState, Movement, ActionType
from agent.critic import Critic
from agent.micro_planner import MicroPlanner
from agent.action_generator import ActionGenerator

# 引入 Toolbox 组件
from toolbox.interface import Toolbox
from toolbox.bridge import AgentToolboxBridge
from toolbox.impl import register_real_tools

class TestIntegration(unittest.TestCase):
    def setUp(self):
        # 1. 初始化原始 Toolbox
        # 确保 toolbox.json 存在于 toolbox 目录下
        json_path = Path("toolbox/toolbox.json")
        if not json_path.exists():
            # 临时创建一个
            import json
            with open(json_path, 'w') as f:
                # 把上面的 JSON 内容写进去，这里为了简化省略
                pass 
                
        self.raw_toolbox = Toolbox(json_path)
        
        real_tracks_path = "/Users/aritxonly/Codes/Agent/Video2Music/mgen/tracks.auto.json"
        ctx = {"tracks_json_path": real_tracks_path}
        
        # 2. 注册实现 (Implementations)
        register_real_tools(self.raw_toolbox, context=ctx)
        
        # 3. 初始化 Bridge
        self.bridge = AgentToolboxBridge(self.raw_toolbox)
        
        # 4. 初始化 Agent 组件
        # 注意：这里我们给 ActionGenerator 传入 bridge，而不是 raw_toolbox
        # 虽然 ActionGenerator 里目前只是用来 generate priors，但如果要让它感知工具，也应该用 bridge
        self.llm_mock = None # 暂时不需要 LLM
        self.action_gen = ActionGenerator(self.llm_mock, self.bridge)
        self.critic = Critic()
        
        # 5. 初始化 Planner，注入 Bridge 作为 toolbox
        # MicroPlanner 调用的 .execute() 方法完全匹配 Bridge 的签名
        self.planner = MicroPlanner(self.action_gen, self.critic, self.bridge)

    def test_end_to_end_search(self):
        """测试 Agent 通过 Bridge 调用 Toolbox 完成搜索"""
        print("\n=== Integration Test: Agent -> Bridge -> Toolbox ===")
        
        # 构造初始状态
        mov1 = Movement("m1", 0, 10, [], visual_summary="A happy dog running")
        state = AgentState(movements=[mov1])
        
        # 手动注入一个 Mock 的 Proposal (绕过 ActionGenerator 的 LLM)
        # 强行让 Planner 执行 SEARCH
        
        # 我们这里不跑 planner.plan() 循环，而是直接测试 execute 环节
        # 模拟 micro_planner 内部的调用：
        
        action_type = ActionType.SEARCH
        params = {"query": "Happy Dog Music"}
        
        print(f"Agent requesting: {action_type} with {params}")
        
        # 执行！
        result_track = self.bridge.execute(action_type, params, mov1)
        
        print(f"Result Track: {result_track}")
        
        self.assertIsNotNone(result_track)
        # self.assertEqual(result_track.meta['sem_score'], 0.85) # 验证拿到了 implementations.py 里的假数据
        print(f'Sem score {result_track.meta['sem_score']}')
        print("Success! Agent successfully talked to Toolbox implementation.")

if __name__ == "__main__":
    unittest.main()