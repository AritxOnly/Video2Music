import os
import sys
import logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.getcwd())

from agent.state import AgentState, Movement, Track, SegmentSemantics
from agent.critic import Critic
from agent.micro_planner import MicroPlanner
from agent.action_generator import ActionGenerator
from toolbox.interface import Toolbox
from toolbox.bridge import AgentToolboxBridge
from toolbox.impl import register_real_tools
from llm import get_llm

TRACKS_PATH = "/Users/aritxonly/Codes/Agent/Video2Music/mgen/tracks.auto.json"
TOOLBOX_CONFIG = "toolbox/toolbox.json"

def main():
    print("====== Video2Music Agent: Real World Test ======")
    
    # 1. 初始化基础设施 (Toolbox & MGen)
    print("\n[1] Initializing Toolbox...")
    if not os.path.exists(TOOLBOX_CONFIG):
        # 临时创建配置，防止报错
        import json
        with open(TOOLBOX_CONFIG, 'w') as f:
            json.dump({"tools": []}, f) # 内容不重要，implementations 会覆盖

    raw_toolbox = Toolbox(TOOLBOX_CONFIG)
    # 注入真实音乐库
    register_real_tools(raw_toolbox, context={'tracks_json_path': TRACKS_PATH})
    bridge = AgentToolboxBridge(raw_toolbox)

    # 2. 初始化大脑 (LLM & ActionGen)
    print("\n[2] Initializing Agent Brain (DeepSeek)...")
    try:
        # 使用 DeepSeek-V3 (deepseek-chat)
        llm = get_llm(name='deepseek-v3')
        
        # 加载 ActionGenerator (自动读取 guidance.json)
        action_gen = ActionGenerator(llm_interface=llm)
    except Exception as e:
        print(f"Failed to init LLM: {e}")
        return

    # 3. 初始化评价与规划 (Critic & Planner)
    critic = Critic()
    planner = MicroPlanner(action_gen, critic, bridge)
    planner.beam_width = 2 # 真实测试保持小一点，方便看日志

    # 4. 构造测试场景
    print("\n[3] Constructing Scenario...")
    # 场景：一段 15秒 的视频，画面是“赛博朋克追逐战”，非常紧张。
    # 挑战：我们故意不在初始状态给它配乐，看它能不能自己搜到“Cyberpunk/Action”类的音乐。
    
    shots = [
        SegmentSemantics(0, 5, ["neon", "city"], "mysterious", "walking"),
        SegmentSemantics(5, 15, ["chase", "gun"], "intense", "running")
    ]
    
    mov1 = Movement(
        id="mov_cyberpunk_01",
        start_time=0.0,
        end_time=15.0,
        shots=shots,
        visual_summary="A futuristic cyberpunk city. A detective walks in rain, then suddenly starts chasing a suspect. High tension."
    )
    
    state = AgentState(movements=[mov1])
    
    # 5. 启动 Agent！
    print("\n>>> STARTING AGENT EXECUTION <<<\n")
    final_state = planner.plan(state)
    
    # 6. 输出结果
    print("\n" + "="*40)
    print("       MISSION ACCOMPLISHED")
    print("="*40)
    
    print(f"Final Score: {final_state.total_score:.2f}")
    
    best_track = final_state.assigned_tracks.get(0)
    if best_track:
        print(f"Selected Track: {best_track.meta.get('title')} (ID: {best_track.id})")
        print(f"Source: {best_track.source}")
        
        print(">>> best_track.meta keys:", sorted(best_track.meta.keys()))
        print(">>> best_track.meta:", best_track.meta)
        
        def fmt_score(x):
            return "NA" if x is None else f"{x:.2f}"
        
        sem = best_track.meta.get('sem_score')
        harm = best_track.meta.get('harm_score')
        
        print(f"Scores -> Sem:{fmt_score(sem)} | Harm:{fmt_score(harm)}")
    else:
        print("Failed to select a track.")

    print("\nAction History:")
    for step in final_state.action_history:
        act = step['action']
        params = step['params']
        # 尝试打印 LLM 给出的 reason，如果有的话
        # 注意：ActionGen 解析时可能没把 reason 存进 params，视具体实现而定
        print(f" -> Step {step['step']}: {act} | {params}")
        if step.get('tags'):
            print(f"    Diagnosis: {step['tags']}")

if __name__ == "__main__":
    main()