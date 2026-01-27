import pytest

# 按你的实际模块路径改这两行：
from agent.critic import Critic
from agent.state import ActionType


# ------------------------
# 最小可用 stub（避免引入真实 state/track/movement 复杂依赖）
# ------------------------

class DummyTrack:
    def __init__(self, meta=None, play_start=0.0):
        self.meta = meta or {}
        self.play_start = play_start


class DummyMovement:
    def __init__(
        self,
        start_time=0.0,
        end_time=10.0,
        cut_times=None,
        visual_energy=None,
        visual_variance=0.0,
        meta=None,
    ):
        self.start_time = start_time
        self.end_time = end_time
        self.cut_times = cut_times if cut_times is not None else []
        self.visual_energy = visual_energy
        self.visual_variance = visual_variance
        self.meta = meta or {}


class DummyState:
    def __init__(self, current_movement_index=0, assigned_tracks=None):
        self.current_movement_index = current_movement_index
        self.assigned_tracks = assigned_tracks or {}
        self.failure_tags = {}
        self.accumulated_cost = 0.0


# ------------------------
# fixtures
# ------------------------

@pytest.fixture
def critic():
    return Critic()


@pytest.fixture
def base_state():
    # 第二段 movement（index=1），因此 prev_idx=0 存在上一段 track
    prev = DummyTrack(meta={"key": "C"})
    st = DummyState(current_movement_index=1, assigned_tracks={0: prev})
    return st


# ------------------------
# 核心测试：evaluate 计算、写回缓存、failure tags、cost
# ------------------------

def test_evaluate_updates_cost_and_tags_and_score(monkeypatch, critic, base_state):
    """
    目标：
    - sem_score 从 meta 读取
    - harm 从 prev_key -> cur_key 计算
    - sync/flow 缺失时会调用 compute_* 并写回 meta
    - accumulated_cost 增加 cost
    - failure_tags 低分/缺失标签符合预期（我们这里让 sync/flow 算出来不低）
    """

    # mock: sync/flow 计算函数固定返回
    # 注意：按你 critic.py 的 import 路径 patch（critic.py 里是 from utils.sync import compute_sync_score）
    def fake_sync(*args, **kwargs):
        class D:  # SyncDebug
            used_librosa = False
            window_s = 0.5
            num_cuts = 2
            per_cut_maxima = [0.9, 0.8]
            onset_norm_max = 1.0
        return 0.85, D()

    def fake_flow(*args, **kwargs):
        class D:  # FlowDebug
            used_librosa = False
            resample_hz = 1.0
            n_points = 10
            pearson_corr = 0.6
        # score = 0.8（只是示例；这里直接返回最终 score）
        return 0.80, D()

    # 关键：patch 到 critic 模块里用到的名字
    import agent.critic as critic_module
    monkeypatch.setattr(critic_module, "compute_sync_score", fake_sync)
    monkeypatch.setattr(critic_module, "compute_flow_score", fake_flow)

    # 当前 track：提供 sem_score 和 key、filepath（哪怕 mock 不用也给齐）
    tr = DummyTrack(
        meta={
            "sem_score": 0.70,
            "key": "G",
            "filepath": "dummy.mp3",
            # sync_score/flow_score 不提供 -> 会触发计算
        },
        play_start=0.0,
    )

    mv = DummyMovement(
        start_time=0.0,
        end_time=10.0,
        cut_times=[2.0, 6.0],
        visual_energy=[0.1, 0.2, 0.3, 0.4],
        visual_variance=0.1,
    )

    # action: SEARCH cost=0.1
    score = critic.evaluate(base_state, ActionType.SEARCH, mv, tr)

    # 1) cost 更新
    assert pytest.approx(base_state.accumulated_cost, 1e-6) == 0.1

    # 2) sync/flow 写回缓存
    assert "sync_score" in tr.meta
    assert "flow_score" in tr.meta
    assert pytest.approx(tr.meta["sync_score"], 1e-6) == 0.85
    assert pytest.approx(tr.meta["flow_score"], 1e-6) == 0.80

    # 3) failure_tags：sem/harm/sync/flow 都不应触发低分
    tags = base_state.failure_tags
    assert "SEM_LOW" not in tags
    assert "HARM_BAD" not in tags
    assert "SYNC_BAD" not in tags
    assert "FLOW_BAD" not in tags

    # 4) score 合理：不要求精确到 harm（因为 harm 取决于你 harmonics 实现）
    # 但至少应大于 0（因为 sem/harm/sync/flow 都较高，只有 cost=0.1）
    assert score > 0.0


def test_sync_flow_cached_second_call_does_not_recompute(monkeypatch, critic, base_state):
    """
    目标：
    - 第一次 evaluate 触发 compute_* 并写回 meta
    - 第二次 evaluate 直接读取 meta，不再调用 compute_*
    """

    calls = {"sync": 0, "flow": 0}

    def fake_sync(*args, **kwargs):
        calls["sync"] += 1
        class D:
            used_librosa = False
            window_s = 0.5
            num_cuts = 1
            per_cut_maxima = [1.0]
            onset_norm_max = 1.0
        return 0.9, D()

    def fake_flow(*args, **kwargs):
        calls["flow"] += 1
        class D:
            used_librosa = False
            resample_hz = 1.0
            n_points = 5
            pearson_corr = 0.0
        return 0.5, D()

    import agent.critic as critic_module
    monkeypatch.setattr(critic_module, "compute_sync_score", fake_sync)
    monkeypatch.setattr(critic_module, "compute_flow_score", fake_flow)

    tr = DummyTrack(meta={"sem_score": 0.6, "key": "C", "filepath": "dummy.mp3"}, play_start=0.0)
    mv = DummyMovement(
        start_time=0.0,
        end_time=10.0,
        cut_times=[3.0],
        visual_energy=[0.1, 0.2, 0.3],
    )

    # 第一次：会算
    critic.evaluate(base_state, ActionType.CONTINUE, mv, tr)
    assert calls["sync"] == 1
    assert calls["flow"] == 1

    # 第二次：走缓存，不再算
    critic.evaluate(base_state, ActionType.CONTINUE, mv, tr)
    assert calls["sync"] == 1
    assert calls["flow"] == 1


def test_harm_score_degrades_when_keys_far(monkeypatch, critic):
    """
    目标：
    - 验证 harm 的基本趋势：同调 > 远关系
    - 这里不用 mock harmonics，直接靠你的 harmonic_distance 实现
    """

    prev = DummyTrack(meta={"key": "C"})
    state = DummyState(current_movement_index=1, assigned_tracks={0: prev})

    # 当前 track 同调
    tr_same = DummyTrack(meta={"sem_score": 0.6, "key": "C", "filepath": "x.mp3"})
    mv = DummyMovement(start_time=0.0, end_time=10.0, cut_times=[], visual_energy=[0.1, 0.2])

    # 为了不触发真实 sync/flow，这里 mock 让它们固定返回 0（但不影响 harm 对比）
    import agent.critic as critic_module
    def fake_sync(*args, **kwargs):
        class D: pass
        return 0.0, D()
    def fake_flow(*args, **kwargs):
        class D: pass
        return 0.0, D()
    monkeypatch.setattr(critic_module, "compute_sync_score", fake_sync)
    monkeypatch.setattr(critic_module, "compute_flow_score", fake_flow)

    s1 = critic.evaluate(state, ActionType.CONTINUE, mv, tr_same)

    # 远关系：例如 C -> F#（五度圈距离较大）
    state2 = DummyState(current_movement_index=1, assigned_tracks={0: prev})
    tr_far = DummyTrack(meta={"sem_score": 0.6, "key": "F#", "filepath": "x.mp3"})
    s2 = critic.evaluate(state2, ActionType.CONTINUE, mv, tr_far)

    # 同调的总分应该 >= 远关系（因为 sem 相同，sync/flow 相同）
    assert s1 >= s2