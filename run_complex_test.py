import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# [Fix] 屏蔽 Tokenizers 并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 添加项目根目录到路径
sys.path.append(os.getcwd())

from agent.state import AgentState, Movement, SegmentSemantics
from agent.critic import Critic
from agent.micro_planner import MicroPlanner
from agent.action_generator import ActionGenerator
from toolbox.interface import Toolbox
from toolbox.bridge import AgentToolboxBridge
from toolbox.impl import register_real_tools
from llm import get_llm

# === 配置 ===
TRACKS_PATH = "/Users/aritxonly/Codes/Agent/Video2Music/mgen/tracks.auto.json"
CACHE_FILE = "z___outputs/result_v2.mp4.cache.json" # 你的真实数据文件
TOOLBOX_CONFIG = "toolbox/toolbox.json"

def load_movements_from_cache(json_path: str):
    """
    将 VLM 产生的 cache.json (TimelineEvent List) 转换为 Agent 的 Movement List
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {json_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"[Loader] Loaded {len(data)} events from {json_path}")
    
    movements = []
    for i, event in enumerate(data):
        # 提取字段
        start = event.get('start_sec', 0.0)
        end = event.get('end_sec', 0.0)
        label = event.get('label', '') # Mood
        summary = event.get('text_summary', '')
        # 如果 summary 为空，用 label 和 extra 拼凑一下
        if not summary:
            extra = event.get('extra', {})
            activity = extra.get('activity', '')
            tags = extra.get('tags', [])
            tags_str = ", ".join(tags) if isinstance(tags, list) else str(tags)
            summary = f"Mood: {label}. Activity: {activity}. Visual tags: {tags_str}"
            
        # 构造 Movement
        # 注意：这里我们把每一个 VLM Event 当作一个独立的 Movement
        # 也可以在 Agent 内部通过 Merge 工具合并它们
        mov = Movement(
            id=f"mov_{i}_{label.replace(' ', '_')}",
            start_time=start,
            end_time=end,
            visual_summary=summary,
            # 构造一个伪 Shot 列表 (假设该 Event 对应一段连续画面)
            shots=[SegmentSemantics(start, end, [], label, "")]
        )
        movements.append(mov)
        
    return movements

def main():
    print("====== Video2Music Agent: Complex Context Test ======")
    
    # 1. 初始化工具箱
    if not os.path.exists(TOOLBOX_CONFIG):
        with open(TOOLBOX_CONFIG, 'w') as f:
            json.dump({"tools": []}, f)

    raw_toolbox = Toolbox(TOOLBOX_CONFIG)
    register_real_tools(raw_toolbox, context={'tracks_json_path': TRACKS_PATH})
    bridge = AgentToolboxBridge(raw_toolbox)

    # 2. 初始化 Agent
    try:     
        llm = get_llm(name="deepseek-v3")
        action_gen = ActionGenerator(llm_interface=llm)
    except Exception as e:
        print(f"Failed to init LLM: {e}")
        return

    critic = Critic()
    planner = MicroPlanner(action_gen, critic, bridge)
    # 增加 Beam Width 以应对复杂场景
    planner.beam_width = 2 

    # 3. 加载真实数据
    try:
        movements = load_movements_from_cache(CACHE_FILE)
        if not movements:
            print("No movements loaded. Exiting.")
            return
            
        # 构造初始状态
        # 这是一个包含多个乐章的复杂状态
        state = AgentState(movements=movements)
        
        # 注入全局 Video Semantics (Mock 一下，实际应从 cache 读取)
        # state.global_semantics = ... 
        
    except Exception as e:
        print(f"Error loading cache: {e}")
        # 如果文件不存在，生成 mock 数据兜底，方便你直接测试代码逻辑
        print(">> Fallback: Generating Mock Complex Data (3 Segments)")
        movements = [
            Movement("mov_0", 0, 10, [], "Peaceful nature, sunny day, birds flying."),
            Movement("mov_1", 10, 20, [], "Sudden storm, dark clouds, heavy rain, intense atmosphere."),
            Movement("mov_2", 20, 30, [], "Rain stops, rainbow appears, calm again.")
        ]
        state = AgentState(movements=movements)

    # 4. 运行 Agent
    print(f"\n>>> AGENT STARTING | Processing {len(state.movements)} Movements <<<\n")
    
    final_state = planner.plan(state)
    
    # 5. 结果分析
    print("\n" + "="*50)
    print("             FINAL PLAN REPORT")
    print("="*50)
    print(f"Total Score: {final_state.total_score:.2f}")
    
    for idx, mov in enumerate(final_state.movements):
        track = final_state.assigned_tracks.get(idx)
        print(f"\n[Movement {idx}] {mov.visual_summary[:50]}... ({mov.start_time}-{mov.end_time}s)")
        if track:
            print(f"  Music: {track.meta.get('title')} (Key: {track.meta.get('key')})")
            print(f"  Scores: Sem={track.meta.get('sem_score'):.2f} | Harm={track.meta.get('harm_score'):.2f}")
        else:
            print("  Music: [NO TRACK SELECTED]")

    print("\n" + "-"*50)
    print("Action History Trace:")
    for step in final_state.action_history:
        print(f"Step {step['step']}: {step['action']} | {step['params']}")
        if step.get('tags'):
            print(f"   Diagnosis: {step['tags']}")

if __name__ == "__main__":
    main()