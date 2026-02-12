from typing import List, Dict, Any, Optional, Tuple
import copy
from collections import Counter

from agent.state import AgentState, ActionType, Track, Movement
from agent.action_generator import ActionGenerator
from agent.critic import Critic


class MicroPlanner:
    def __init__(self, action_gen: ActionGenerator, critic: Critic, toolbox):
        self.action_gen = action_gen
        self.critic = critic
        self.toolbox = toolbox
        self.beam_width = 3

        self.max_steps_factor = 5
        self.safety_sec = 0.5
        self.rebuild_cursors_on_edit = True

    # ---------- helpers ----------
    @staticmethod
    def _mov_dur(m: Movement) -> float:
        return float(m.end_time - m.start_time)

    @staticmethod
    def _track_dur(t: Track) -> float:
        d = float(getattr(t, "duration", 0.0) or 0.0)
        if d > 0:
            return d
        meta = getattr(t, "meta", {}) or {}
        return float(meta.get("duration", 0.0) or 0.0)

    @staticmethod
    def _to_track(obj: Any) -> Optional[Track]:
        if obj is None:
            return None
        if isinstance(obj, Track):
            return obj
        if isinstance(obj, dict):
            return Track(
                id=str(obj.get("track_id") or obj.get("id") or ""),
                source=obj.get("source", "library"),
                duration=float(obj.get("duration", 0.0) or 0.0),
                meta=obj.get("meta", {}) or {},
                play_start=float(obj.get("play_start", 0.0) or 0.0),
                play_duration=float(obj.get("play_duration", 0.0) or 0.0),
            )
        return None

    def _normalize_tool_result(self, res: Any) -> List[Track]:
        if res is None:
            return []
        if isinstance(res, list):
            out = []
            for x in res:
                t = self._to_track(x)
                if t and t.id:
                    out.append(t)
            return out
        t = self._to_track(res)
        return [t] if (t and t.id) else []

    def _infer_prev_key(self, st: AgentState, mov_idx: int) -> Optional[str]:
        if mov_idx <= 0:
            return None
        prev = self._to_track(st.assigned_tracks.get(mov_idx - 1))
        if not prev:
            return None
        meta = prev.meta or {}
        return meta.get("key")

    # ---------- cursor rebuild ----------
    def _rebuild_track_cursors(self, st: AgentState) -> None:
        cursors: Dict[str, float] = {}
        for i, mov in enumerate(st.movements):
            if i not in st.assigned_tracks:
                continue
            tr = self._to_track(st.assigned_tracks.get(i))
            if not tr or not tr.id:
                continue

            # prefer per-movement track.play_start; fallback to state.source_starts
            src = None
            if getattr(tr, "play_start", None) is not None:
                src = float(tr.play_start)
            elif hasattr(st, "source_starts") and i in st.source_starts:
                src = float(st.source_starts[i])

            if src is None:
                continue

            end = src + self._mov_dur(mov)
            cursors[tr.id] = max(cursors.get(tr.id, 0.0), end)

        st.track_cursors = cursors

    # ---------- hard constraints ----------
    def _hard_prune_continue(self, st: AgentState, mov_idx: int) -> Tuple[bool, str]:
        if mov_idx <= 0:
            return True, "CONTINUE_ON_FIRST"
        prev = self._to_track(st.assigned_tracks.get(mov_idx - 1))
        if not prev:
            return True, "NO_PREV_TRACK"

        # src_start for current mov if continuing
        prev_src = None
        prev_assigned = self._to_track(st.assigned_tracks.get(mov_idx - 1))
        if prev_assigned and getattr(prev_assigned, "play_start", None) is not None:
            prev_src = float(prev_assigned.play_start)
        elif mov_idx - 1 in st.source_starts:
            prev_src = float(st.source_starts[mov_idx - 1])
        if prev_src is None:
            return True, "NO_PREV_SRC"

        src_start = prev_src + self._mov_dur(st.movements[mov_idx - 1])
        need_end = src_start + self._mov_dur(st.movements[mov_idx])

        total = self._track_dur(prev)
        if total <= 0:
            return True, "NO_TRACK_DUR"
        if need_end > (total - self.safety_sec):
            return True, f"OOB_CONTINUE need_end={need_end:.2f} total={total:.2f}"
        return False, "OK"

    def _hard_prune_assign(self, st: AgentState, mov_idx: int, mov: Movement, tr: Track) -> Tuple[bool, str, float]:
        """
        For SEARCH/REQUERY/GENERATE: compute the src_start we'd use, and prune if OOB.
        Returns (prune?, reason, src_start)
        """
        total = self._track_dur(tr)
        if total <= 0:
            return True, "NO_TRACK_DUR", 0.0

        base = float(getattr(tr, "play_start", 0.0) or 0.0)
        src_start = float(st.track_cursors.get(tr.id, base))

        need_end = src_start + self._mov_dur(mov)
        if need_end > (total - self.safety_sec):
            return True, f"OOB_ASSIGN need_end={need_end:.2f} total={total:.2f}", src_start

        return False, "OK", src_start

    # ---------- apply transitions ----------
    def _apply_track_for_movement(self, st: AgentState, mov_idx: int, mov: Movement, tr: Track, src_start: float) -> Track:
        # IMPORTANT: make a per-movement copy to avoid overwriting play_start across movements
        seg = copy.deepcopy(tr)
        seg.play_start = float(src_start)
        seg.play_duration = float(self._mov_dur(mov))

        st.assigned_tracks[mov_idx] = seg
        st.source_starts[mov_idx] = float(src_start)
        st.track_cursors[seg.id] = float(src_start + self._mov_dur(mov))

        st.current_movement_index += 1
        st.failure_tags = {}
        return seg

    def _apply_continue(self, st: AgentState, base: AgentState, mov_idx: int) -> Optional[Track]:
        prev_seg = self._to_track(base.assigned_tracks.get(mov_idx - 1))
        if not prev_seg:
            return None
        prev_src = float(prev_seg.play_start) if getattr(prev_seg, "play_start", None) is not None else None
        if prev_src is None and (mov_idx - 1 in base.source_starts):
            prev_src = float(base.source_starts[mov_idx - 1])
        if prev_src is None:
            return None

        src_start = prev_src + self._mov_dur(base.movements[mov_idx - 1])
        # copy prev track to new segment
        tr = copy.deepcopy(prev_seg)
        tr.play_start = float(src_start)
        tr.play_duration = float(self._mov_dur(base.movements[mov_idx]))

        st.assigned_tracks[mov_idx] = tr
        st.source_starts[mov_idx] = float(src_start)
        st.track_cursors[tr.id] = float(src_start + self._mov_dur(base.movements[mov_idx]))

        st.current_movement_index += 1
        st.failure_tags = {}
        return tr

    def _score_and_record(self, st: AgentState, action_type: ActionType, mov_idx: int,
                          mov: Movement, tr: Optional[Track], step: int, params: Dict[str, Any]) -> None:
        eval_tr = tr if tr else Track("temp", "none", 0.0, {})
        step_score = self.critic.evaluate(st, action_type, mov, eval_tr, mov_idx)
        st.total_score += float(step_score)
        st.action_history.append({
            "step": step,
            "movement_idx": mov_idx,
            "action": action_type.name,
            "params": params,
            "score": float(step_score),
            "tags": st.failure_tags.copy(),
            "track_id": getattr(tr, "id", None) if tr else None,
            "source_start": float(getattr(tr, "play_start", 0.0)) if tr else None,
        })

    # ---------- main ----------
    def plan(self, initial_state: AgentState) -> AgentState:
        beams = [initial_state]
        max_steps = max(10, len(initial_state.movements) * self.max_steps_factor)
        step = 0

        print(f"[MicroPlanner] Start planning for {len(initial_state.movements)} movements...")

        while step < max_steps:
            candidates: List[AgentState] = []
            all_terminal = True

            pruned = Counter()
            expanded = Counter()
            branch = []

            for st0 in beams:
                if st0.is_terminal:
                    candidates.append(st0)
                    continue
                all_terminal = False

                mov_idx = st0.current_movement_index
                if mov_idx >= len(st0.movements):
                    st_term = st0.clone()
                    st_term.current_movement_index = len(st_term.movements)
                    candidates.append(st_term)
                    continue

                mov = st0.movements[mov_idx]

                # propose
                if mov_idx not in st0.assigned_tracks and not st0.failure_tags:
                    proposed = [
                        {"type": ActionType.SEARCH, "params": {"query": mov.visual_summary}},
                        {"type": ActionType.REQUERY, "params": {"query": mov.visual_summary}},
                        {"type": ActionType.GENERATE_SUNO, "params": {"prompt": mov.visual_summary, "duration": self._mov_dur(mov)}},
                        {"type": ActionType.CONTINUE, "params": {}},  # allow, will be pruned if impossible
                    ]
                else:
                    proposed = self.action_gen.propose(st0, k=3) or []

                # ensure at least one retrieval action exists
                tset = {a["type"] for a in proposed}
                if ActionType.SEARCH not in tset:
                    proposed.append({"type": ActionType.SEARCH, "params": {"query": mov.visual_summary}})
                if ActionType.REQUERY not in tset:
                    proposed.append({"type": ActionType.REQUERY, "params": {"query": mov.visual_summary}})

                local = 0

                for a in proposed:
                    at: ActionType = a["type"]
                    params: Dict[str, Any] = a.get("params", {}) or {}

                    if at == ActionType.CONTINUE:
                        prune, reason = self._hard_prune_continue(st0, mov_idx)
                        if prune:
                            pruned["CONTINUE"] += 1
                            pruned[reason] += 1
                            continue

                        st = st0.clone()
                        seg = self._apply_continue(st, st0, mov_idx)
                        if not seg:
                            pruned["CONTINUE_APPLY_FAIL"] += 1
                            continue

                        self._score_and_record(st, ActionType.CONTINUE, mov_idx, mov, seg, step, params)
                        candidates.append(st)
                        local += 1
                        expanded["CONTINUE"] += 1
                        continue

                    # execute tool
                    exec_res = self.toolbox.execute(at, params, mov, st0)
                    tracks = self._normalize_tool_result(exec_res)

                    if at in (ActionType.SEARCH, ActionType.REQUERY, ActionType.GENERATE_SUNO):
                        if not tracks:
                            pruned[f"{at.name}_NO_RESULT"] += 1
                            continue

                        # IMPORTANT: prune OOB candidates here
                        for tr in tracks:
                            p, reason, src_start = self._hard_prune_assign(st0, mov_idx, mov, tr)
                            if p:
                                pruned[at.name] += 1
                                pruned[reason] += 1
                                continue

                            st = st0.clone()
                            seg = self._apply_track_for_movement(st, mov_idx, mov, tr, src_start)
                            self._score_and_record(st, at, mov_idx, mov, seg, step, params)
                            candidates.append(st)
                            local += 1
                            expanded[at.name] += 1

                    elif at == ActionType.SPLIT:
                        if not exec_res:
                            pruned["SPLIT_NO_RESULT"] += 1
                            continue
                        st = st0.clone()
                        st.movements[mov_idx] = exec_res[0]
                        st.movements.insert(mov_idx + 1, exec_res[1])
                        st.failure_tags = {}
                        if self.rebuild_cursors_on_edit:
                            self._rebuild_track_cursors(st)
                        self._score_and_record(st, ActionType.SPLIT, mov_idx, st.movements[mov_idx], None, step, params)
                        candidates.append(st)
                        local += 1
                        expanded["SPLIT"] += 1

                    elif at == ActionType.MERGE:
                        if not exec_res or mov_idx + 1 >= len(st0.movements):
                            pruned["MERGE_NO_RESULT"] += 1
                            continue
                        st = st0.clone()
                        st.movements[mov_idx] = exec_res
                        del st.movements[mov_idx + 1]
                        st.failure_tags = {}
                        if self.rebuild_cursors_on_edit:
                            self._rebuild_track_cursors(st)
                        self._score_and_record(st, ActionType.MERGE, mov_idx, st.movements[mov_idx], None, step, params)
                        candidates.append(st)
                        local += 1
                        expanded["MERGE"] += 1

                branch.append(local)

            if all_terminal:
                break
            if not candidates:
                print("[MicroPlanner] All beams died (empty candidates). Stopping.")
                break

            # step stats
            dist = Counter()
            for s in candidates:
                if s.action_history:
                    dist[s.action_history[-1]["action"]] += 1

            print(
                f"[MicroPlanner] Step {step} stats: "
                f"candidates={len(candidates)} branch_per_beam={branch} "
                f"dist={dict(dist)} expanded={dict(expanded)} pruned={dict(pruned)}"
            )

            # select
            candidates.sort(key=lambda s: s.total_score, reverse=True)
            beams = candidates[: self.beam_width]

            best = beams[0]
            last = best.action_history[-1] if best.action_history else {}
            print(
                f"  >> Step {step}: Best Score={best.total_score:.2f} "
                f"| Mov={best.current_movement_index}/{len(best.movements)} "
                f"| Action={last.get('action')} "
                f"| Track={last.get('track_id')} "
                f"| Src={last.get('source_start')}"
            )

            step += 1

        return beams[0]