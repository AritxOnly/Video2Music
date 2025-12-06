# TODO – Video2Music API Baseline / vsem+mgen

> 目标：先把 API 基线版稳稳跑通，再逐步替换控制信号（时间轴 + 情绪）为我们自己的逻辑。

---

## 0. 当前基线状态确认

- [ ] 使用真实 VLM（qwen）+ `generate_sample.py` 生成并固化一个 `vlm/test/sample.json`（长样例视频 OK）。
- [ ] 在 `vlm/factory.get_vlm()` 中接好 `"sample"` backend。
- [ ] 用 `backend=sample` 跑通整条 pipeline：
  - [ ] `run_video2music(...)` 成功执行；
  - [ ] 输出视频 `output_with_bgm.mp4` 可正常播放，有 BGM 覆盖。

---

## 1. vsem：数据结构与基本构建

### 1.1 扩展 SegmentSemantics 结构

- [ ] 在 `vsem/model.py` 中为 `SegmentSemantics` 增加情绪轴字段：
  - [ ] `valence: float  # [-1, 1] 或 [0, 1]，情绪正负/愉悦度`
  - [ ] `arousal: float  # [0, 1]，激烈程度/能量`
- [ ] 明确字段含义：
  - [ ] `tags`: 保留为「内容标签」（场景/物体/事件），不负责情绪；
  - [ ] `mood/activity/text_summary`: 文本描述；
  - [ ] `valence/arousal`: 后续专门作为配乐控制信号。

### 1.2 build_from_vlm 基础逻辑

- [ ] 在 `build_from_vlm(struct_res, tag_res)` 中：
  - [ ] 优先从 `structure_result.timeline` 构建 `SegmentSemantics`；
  - [ ] 如果 timeline 为空，再 fallback 解析 `raw_text` 里的 JSON；
  - [ ] 从 `tagging_result.raw_text` 按行拆分得到 `global_tags`（全局内容词袋）；
  - [ ] 在构造 `SegmentSemantics` 时，调用一个简单规则函数：
    - [ ] `estimate_emotion(mood, description) -> (valence, arousal)`  
          用中文情绪词（紧张/宁静/温暖/危险/悲伤等）给 valence/arousal 初始值。

---

## 2. vsem：时间轴修正（短期 heuristics）

### 2.1 时间线清洗/合并

- [ ] 新增一个后处理函数（可放 `vsem/builder.py` 或独立模块）：
  - [ ] `refine_segments_with_heuristics(segments, video_duration) -> List[SegmentSemantics]`
- [ ] 在这个函数中实现：
  - [ ] 丢弃长度过短的段（如 `< 2s`）；
  - [ ] 合并相邻且情绪/活动相似的段（`mood` 相同，`activity` 前几个字相近）；
  - [ ] 修正边界：`start_sec >= 0`，`end_sec <= video_duration`。
- [ ] 在 pipeline 中接入：
  - [ ] `semantics = build_from_vlm(struct_res, tag_res)`
  - [ ] `semantics.segments = refine_segments_with_heuristics(semantics.segments, video_duration)`

### 2.2 日后替换路线预留

- [ ] 在 TODO 中记下未来替换点（不必现在实现）：
  - [ ] 用 ffmpeg/OpenCV 做 shot detection（场景切换检测）得到更准的剪辑点；
  - [ ] 将 `SegmentSemantics` 的 start/end snap 到 shot 边界；
  - [ ] 将来可进一步 snap 到音乐 beat（当 mgen/music 端有节拍信息后）。

---

## 3. mgen：从 vsem 信号到 BGM 能量

### 3.1 SimpleRuleArranger 适配 SegmentSemantics

- [ ] 保持 `MusicArrangerInterface.arrange(timeline, library, options)` 接口不变；
- [ ] 在 `simple_arranger.py` 中：
  - [ ] 允许 `timeline` 传入 `SegmentSemantics` 列表（已经部分实现，要确保完整）；
  - [ ] 取时间时优先用 `start_sec/end_sec`；
  - [ ] 取文本时兼容 `label/description` 和 `mood/activity/text_summary`。

### 3.2 能量映射改为用 arousal

- [ ] 修改 `_infer_energy_from_label()` 的使用逻辑：
  - [ ] 如果事件对象有 `arousal` 字段，则 `target_energy = ev.arousal`；
  - [ ] 否则再 fallback 用 label+desc 文本规则估 energy。
- [ ] 保留原 `_infer_energy_from_label()` 仅作为兼容/兜底；
- [ ] 使用 `target_energy` 在指定 style 的 track 列表中选曲（已实现 `_pick_best_track`，用 energy 差值最小）。

---

## 4. render：ffmpeg 渲染检查

- [ ] 确认 `render_with_bgm(...)` 行为符合预期：
  - [ ] 所有 `SegmentMusicPlan` 的 start/end/fade_in/fade_out/gain_db 正常作用；
  - [ ] 多段 BGM 正常使用 `adelay + amix` 对齐时间轴；
  - [ ] 输出视频长度与原视频差不多（`-shortest` 生效）。
- [ ] 若未来要保留原声：
  - [ ] 在 `filter_complex` 中再加一步 `[0:a][bgm_mix]amix=inputs=2` 的 TODO 标记。

---

## 5. vsem：情绪 head（中长期）

> 这部分先在 TODO 里占坑，后续真正动手时再细化。

- [ ] 设计 `EmotionHead` 接口（放在 `vsem/emotion_head.py`）：
  - [ ] `class EmotionHead: def __call__(self, seg: SegmentSemantics) -> SegmentSemantics: ...`
- [ ] 确定输入特征：
  - [ ] 文本特征：`mood/activity/text_summary`；
  - [ ] VLM embedding（如果后端能提供，存进 `SegmentSemantics.extra["vlm_embedding"]`）。
- [ ] 确定输出目标：
  - [ ] 更准确的 `valence/arousal`；
  - [ ] 可选额外控制：`tension`, `brightness`, `scene_type` 等。
- [ ] 在 pipeline 中预留调用点：
  - [ ] 在 `build_from_vlm` 之后、`SimpleRuleArranger` 之前，对每个 segment 过一遍 `emotion_head(seg)`。

---

## 6. tags 用途：为 mret / 检索做准备

- [ ] 约定：`VideoSemantics.global_tags` + `SegmentSemantics.tags` 主要给 mret 用：
  - [ ] 用于将来音乐检索库（曲库的文本描述/标签）做 embedding+检索；
  - [ ] 不直接作为情绪控制信号。
- [ ] 后续在 mret 中：
  - [ ] `keyword_retriever` 使用 tags 做关键词检索；
  - [ ] `embedding_retriever` 使用 VLM embedding / 文本 embedding 做语义检索；
  - [ ] 再把检索结果交回 mgen 参与编排。

---