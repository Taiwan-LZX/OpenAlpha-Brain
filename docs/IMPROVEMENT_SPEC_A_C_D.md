# 三智能体 + BRAIN 适配 + 模板约束范式转移 改进规格书

> 创建日期：2026-05-30 | 最后更新：2026-05-31 (v3 范式转移)
> 基于论文：AlphaAgent (KDD 2025), QuantaAlpha (2026), Alpha-GPT (WQ IQC Top-10), AlphaBench (ICLR 2026), 101 Formulaic Alphas (Kakushadze 2016), Finding Alphas (Tulchinsky)
> 核心原则：LLM 不猜测分数，BRAIN 真实回测是唯一 ground truth；结构（模板）是生存底线，字段（数据）才是探索边界

---

## 实施顺序：A → C → D → E → **F (新增)**

---

## Phase F：模板约束范式转移 (v3)

> 背景调研：WQ BRAIN 平台的 101 Formulaic Alphas 全部由不到 30 个基础算子按固定三段论模板嵌套而成。LLM 自由发明算子嵌套 = 高维随机游走 = 大量垃圾代码。范式转移：强模板约束 + 字段空间探索。

### 核心原则
- "结构（模板）是生存的底线，字段（数据）才是探索的边界"
- Agent 不发明公式，只选模板 + 填字段 + 微调参数
- 15 模板 × 40 字段族 = 600 arm MAB 探索空间
- 三层字段代理图谱 (FieldProxyMap): 语义类别 → 字段族 → 适用模板

### F1: FieldProxyMap 字段代理图谱

| 参数 | 值 |
|------|-----|
| 标注层数 | 3 层 |
| L1 语义类别 | price/volume/fundamental/sentiment/alternative/macro/microstructure/analyst |
| L2 字段族 | ~40 个 (如: supply_chain_sentiment, analyst_expectation, short_term_liquidity, cold_fundamental) |
| L3 适用模板 | 每个字段族的适用模板 ID 列表 |
| 数据来源 | BRAIN API `/data-fields` 端点 |
| LLM 标注策略 | 按 dataset 分批，每批 50-100 字段，输出 JSON 标注 |
| 存储方案 | JSON 主存储 + FAISS 向量索引 (语义搜索) |
| 新文件 | `knowledge/field_proxy_map.py` |

**实现要点：**
- `FieldProxyMap` 类: 封装字段标注加载、查询、推荐
- `_pull_fields_from_brain()`: 从 BRAIN API 拉取全部字段 metadata
- `_annotate_batch()`: LLM 分批标注字段到三层结构
- `_build_faiss_index()`: 构建 FAISS 向量索引用于语义搜索
- `recommend_fields_for_template(template_id, direction)`: 模板匹配推荐
- `get_field_families()`: 返回所有字段族列表用于 MAB arm 空间

### F2: AlphaLogicLibrary 模板库增强 (15 模板)

| 方向 | 短尺度 (5-10d) | 中尺度 (20-60d) | 长尺度 (100-200d) |
|------|---------------|-----------------|-------------------|
| **Momentum/Reversal** | 短期反转 | 中期动量延续 | 长期反转 |
| **Lead-Lag** | 价量先行滞后 | 跨字段延迟 | 行业轮动 |
| **Mean Reversion** | Z-score 偏离回归 | 布林带回归 | 估值回归 |
| **Volatility** | 波动率异常 | 波动率聚类 | 低波溢价 |
| **Value** | — | — | PE/BP 回归 |
| **Liquidity** | 流动性冲击 | 换手率衰减 | — |

每个模板 2-4 个 FASTEXPR 变体，含 `{placeholder}` 占位符。

**实现要点：**
- 更新 `data/market_logics.json` 扩展到 15 个 MarketLogic
- 每个 MarketLogic 的 `factor_templates` 使用 `{price_field}`, `{volume_field}`, `{fundamental_field}`, `{short_lb}`, `{medium_lb}`, `{long_lb}`, `{corr_lb}`, `{zscore_lb}` 等占位符
- `AlphaLogicLibrary.get_templates_for_direction(direction)` 返回该方向的所有模板
- 模板按 `evidence_count` 排序，高证据模板优先

### F3: MAB Arm 空间重构 (600 arm)

| 参数 | 值 |
|------|-----|
| Arm 粒度 | 模板 × 字段族 (15 × 40 = 600) |
| UCB 公式 | `scoring = avg_sharpe + c * sqrt(ln(N) / n)` |
| c 值策略 | 动态: 初期=0.7, 稳定期=0.5, 成熟期=0.3 |
| 探索优先级 | 未探索字段族 > 冷门字段族 > 低 sharpe 字段族 |
| FeatureMap 集成 | explore 模式 → 优先未探索字段族；exploit 模式 → 优先高 sharpe 字段族 |
| Arm 统计 | 每个 arm 追踪: visits, total_sharpe, pass_count, fail_count, avg_fitness |
| 文件 | `learning/mab.py`, `core/loop_engine.py` |

**实现要点：**
- `MAB._arm_key(template_id, field_family)`: 生成 arm 标识
- `MAB.select()`: 返回 `{template_id, field_family, direction, ucb_score}`
- `MAB.record_arm_result(arm_key, sharpe, passed)`: 更新 arm 统计
- 首次选择时优先未探索的 arm (UCB 给 inf)
- 集成 `FeatureMap.get_explore_exploit_schedule()` 动态调整探索权重

### F4: FactorAgent 模板强制

| 参数 | 值 |
|------|-----|
| 默认路径 | 模板选择 → 字段填充 → 参数微调 |
| 模板选择 | 从 MAB arm 获取 template_id + field_family |
| 字段填充 | FieldProxyMap 推荐字段族内具体字段 → LLM 填充 |
| 参数微调 | 调整 window, decay, neutralization 参数 |
| 自由生成触发 | 仅当模板路径连续 3 次 BRAIN 失败 |
| 模板提炼 | 自由生成成功 → 提取算子结构 → 新增模板到库 |

**实现要点：**
- `FactorAgent._select_template(arm)`: 从 MAB arm 获取模板
- `FactorAgent._fill_template_with_fields(template, field_family)`: 字段填充
- `FactorAgent._refine_parameters(expression)`: 参数微调 (window, decay)
- `FactorAgent._extract_template(expression)`: 提取算子结构为新模板
- 更新 `_FACTOR_SYSTEM_PROMPT` 和 `_TEMPLATE_REFINEMENT_PROMPT`

### F5: IdeaAgent 字段驱动命题

| 参数 | 值 |
|------|-----|
| 旧职责 | 自由生成金融假设 (编故事) |
| 新职责 | 字段驱动命题: 匹配字段族到模板 |
| 输入 | 方向 + 模板 + FieldProxyMap |
| 输出 | 具体字段组合推荐 + 命题描述 |
| 字段探索历史 | 维护已尝试的字段组合，避免重复 |
| 关联推理 | 基于字段代理图谱的语义关联推荐相关字段族 |

**实现要点：**
- `IdeaAgent.generate_hypothesis()`: 改为接收 MAB arm 作为输入
- 从 FieldProxyMap 查询字段族的语义描述和关联字段
- 生成"字段组合命题": "假设 {field_family_A} 字段对 {field_family_B} 有 Lead-Lag 效应"
- 更新 `_IDEA_SYSTEM_PROMPT` 为字段驱动模式

---

### 改动文件清单 (Phase F)

| 文件 | F1 | F2 | F3 | F4 | F5 |
|------|:---:|:---:|:---:|:---:|:---:|
| `knowledge/field_proxy_map.py` | **新建** | - | - | - | - |
| `data/field_proxy_map.json` | **新建** | - | - | - | - |
| `generation/alpha_logics.py` | - | 修改 | - | - | - |
| `data/market_logics.json` | - | 修改 | - | - | - |
| `learning/mab.py` | - | - | 修改 | - | - |
| `core/loop_engine.py` | - | - | 修改 | - | - |
| `agents/multi_agent.py` | - | - | - | 修改 | 修改 |
| `generation/prompts.py` | - | - | - | 修改 | 修改 |

### 实施状态

| Phase | 模块 | 状态 | 说明 |
|-------|------|:----:|------|
| F1 | FieldProxyMap | ⬜ 待实施 | 从 BRAIN API 拉取字段 + LLM 三层标注 |
| F2 | 模板库增强 | ⬜ 待实施 | 扩展到 15 模板 × 6 方向 |
| F3 | MAB Arm 重构 | ⬜ 待实施 | 改为 模板×字段族 (600 arm) |
| F4 | FactorAgent 模板强制 | ⬜ 待实施 | Prompt 改造 + 模板提炼机制 |
| F5 | IdeaAgent 字段驱动 | ⬜ 待实施 | 从编故事转为字段匹配器 |

---

## Phase A：AlphaAgent 三重正则化

### 涉及文件
- `agents/multi_agent.py` — EvalAgent 新增方法 + MultiAgentOrchestrator 调整
- `core/loop_engine.py` — 传入成功alpha池引用
- `services/brain_submitter.py` — 重复变异触发

### A1: AST原创性去重（两阶段过滤 + 重复变异）

| 参数 | 值 |
|------|-----|
| Stage 1 | Embedding 余弦相似度（向量化表达式，快速初筛） |
| Stage 1 阈值 | 余弦相似度 > 0.85 |
| Stage 2 (Top-K) | Top-5 相似候选进入 LLM 语义等价精筛 |
| 触发时机 | 入池时（BRAIN提交成功后） |
| 池存储格式 | 表达式原文 + embedding 向量对 |
| 重复 > 0.7 | 不拒绝，触发变异生成变体（替换算子/窗口绕过重复阈值） |

**实现要点：**
- `EvalAgent._semantic_dedup_check(expression)` — 两阶段去重
- `EvalAgent._archive_success(expression, result)` — 成功后入池（生成embedding + 存储）
- `BrainSubmitter._on_duplication(result)` — BRAIN返回重复>0.7时触发生成变体
- 成功alpha池在 `loop_engine.py` 中维护，作为 EvalAgent 构造参数传入

### A2: 假设-因子 LLM 对齐评分

| 参数 | 值 |
|------|-----|
| 评分方式 | LLM 评估 FactorAgent 表达式 vs IdeaAgent 假设的一致性 |
| 低分阈值 | 对齐分 < 0.6 |
| 低分行为 | IdeaAgent 重新产生更具体的假设描述 |
| 最大重试 | 最多 2 次重产（共 3 次尝试） |
| 最终选择 | 取最高对齐分的版本 |

**实现要点：**
- `EvalAgent._llm_align_check(hypothesis, expression)` — LLM 对齐评分
- `MultiAgentOrchestrator.run_iteration()` 中增加对齐重试循环
- 重试时 IdeaAgent 接收之前失败的诊断信息作为 prompt 上下文

### A3: 复杂度多维限制

| 参数 | 值 |
|------|-----|
| AST 深度阈值 | > 5 |
| 节点数阈值 | > 20 |
| 超限行为 (阶段1) | brain_checks 标记 OVERFITTING_RISK=WARN，不拒绝 |
| 超限行为 (阶段2) | LLM 自动简化表达式（去除冗余嵌套、合并同质算子） |
| 简化触发时机 | 评估阶段即时简化 + BRAIN 失败后简化变体重提交 |

**实现要点：**
- `EvalAgent._check_complexity(expression)` — AST 深度+节点数计算
- `EvalAgent._llm_simplify(expression)` — LLM 驱动的表达式简化
- 复杂度结果写入 brain_checks 的 OVERFITTING_RISK 字段

---

## Phase C：EvalAgent 批判-修订闭环

### 核心原则
- LLM 不猜测分数（AlphaBench ICLR 2026 证明这是 LLM 最弱能力）
- BRAIN 平台真实回测结果是唯一 ground truth
- EvalAgent 职责分离：
  - **提交前**：结构/逻辑检查（算子合理性、假设一致性、语法正确性）
  - **BRAIN 失败后**：解读真实数据 + 结构归因 → FactorAgent 修订

### 配置

| 参数 | 值 |
|------|-----|
| 轮数 | 固定 3 轮 |
| 每轮流程 | EvalAgent 发修改指令 → FactorAgent 定向修订 |
| 批判依据 | BRAIN 回测数据优先 + LLM 结构分析为辅 |
| 最终版本 | 只取第 3 轮修订结果（吸收所有批判后） |
| 嵌入位置 | EvalAgent.evaluate() 后内嵌子循环 |
| EvalAgent 职责1 (提交前) | 结构/逻辑检查：算子组合合理性、假设一致性、语法正确 |
| EvalAgent 职责2 (失败后) | BRAIN 数据分析 + 结构归因 → 生成具体修改指令 |

**实现要点：**
- `MultiAgentOrchestrator._critique_revise_loop(expression, brain_result)` — 3轮批判修订
- EvalAgent 提交前只做结构检查，不输出质量分（避免 LLM 臆测分数）
- BRAIN 失败后 EvalAgent 解读真实反馈，生成修改指令
- FactorAgent 接收修改指令定向修订
- 3 轮后输出最终版本

---

## Phase D：BRAIN 平台特性深度适配

### D1: LLM 动态修复建议（失败模式分类 + 经验蒸馏）

| 参数 | 值 |
|------|-----|
| 修复策略 | LLM 动态生成（非硬编码映射表） |
| LLM 输入上下文 | 完整上下文：失败alpha表达式 + BRAIN所有返回指标 + 同方向成功alpha表达式 + 当前市场状态 |
| 失败模式 | SELF_CORRELATION / LOW_SHARPE / HIGH_TURNOVER / OVERFITTING / 等 |
| 蒸馏目标 | 失败经验注入后续 prompt |

**实现要点：**
- `BrainSubmitter._llm_analyze_failure(result, context)` — LLM 分析失败根因
- 失败分析结果存入经验蒸馏器，注入后续 FactorAgent/EvalAgent prompt
- 维护失败模式→成功修复的历史映射表（用于冷启动时提供参考）

### D2: UCB 动态 Slot 排序

| 参数 | 值 |
|------|-----|
| 算法 | UCB (Upper Confidence Bound) |
| 公式 | `scoring = avg_sharpe + c * sqrt(ln(N) / n)` |
| c 值策略 | 动态 c 值：初期=0.7, 稳定期=0.5, 成熟期=0.3 |
| 阶段划分 | 按总提交数：< 50次=初期, 50-200次=稳定期, > 200次=成熟期 |
| 作用位置 | AlphaCachePool 出队排序 |

**实现要点：**
- `AlphaCachePool` 新增 `_ucb_score()` 方法
- 跟踪每个方向（direction）的提交次数 N 和平均 sharpe
- `next_to_submit()` 改为按 UCB 分数排序出队
- c 值根据当前总提交数动态调整

### D3: Alpha Decay 滑动窗口检测

| 参数 | 值 |
|------|-----|
| 窗口大小 N | 5 次提交 |
| 判定条件 | sharpe 连续下降 + 最新值 < 历史均值的 50% |
| 检测对象 | 按 direction（方向）分组追踪 |
| 衰减后行为 | MAB 降低该方向权重 + IdeaAgent prompt 提示避开衰减方向 |
| 阶段划分 | 同 D2：< 50=初期, 50-200=稳定期, > 200=成熟期 |
| 远期升级 | Phase 4 升级到 ADWIN 多维漂移检测 |

**实现要点：**
- `BrainSubmitter._track_decay(direction, sharpe)` — 记录每次提交的 sharpe
- `BrainSubmitter._check_decay(direction)` — 滑动窗口检测
- 检测到衰减 → 标记方向 DECAY → MAB 权重下调
- 衰减信息注入 IdeaAgent prompt

---

## Phase E：演进/变异算法升级

> 参考论文：MAP-Elites (Nature 2015), NSGA-II (Deb 2002), UCB1 (Auer 2002)
> 核心原则：锦标赛替代顺序选择、UCB1替代随机变异、MAP-Elites替代单一fitness排序

### 涉及文件
- `evolution/crossover_mutation.py` — E1 锦标赛+E2自适应变异+E4方向感知fitness
- `evolution/quality_diversity.py` — E3 MAP-Elites质量多样性

### E1: 锦标赛选择 + 精英保留

| 参数 | 值 |
|------|-----|
| 选择算法 | k-way Tournament Selection |
| k 值 | 3 |
| 精英数量 | Top-2 |
| 精英行为 | 直接复制到下一代变异候选列表 |
| 父代配对 | 两轮独立锦标赛（避免相同父代） |
| 变异父代 | 每次从排序种群中独立锦标赛选取 |

**实现要点：**
- `CrossoverMutationEngine._tournament_select(population, k)` — 锦标赛选择
- `generate_variants()` 开头按 fitness 降序排序 → Top-2 精英直接入队
- 交叉父代两轮独立锦标赛（parent2 排除 parent1）
- 变异父代从排序种群中锦标赛选取（3轮）

### E2: 双维度自适应变异

| 维度 | 机制 | 参数 |
|------|------|------|
| 算子级 | UCB1 动态排序 | c=1.41, 追踪 visits/reward |
| 种群级 | CV多样性监控 | 窗口=10, 低多样性阈值=0.15 |
| Boost触发 | CV < 0.15 | variant_limits ×1.5~2.0 |
| 算子统计 | record_operator_result() | 外部调用记录成功/失败 |

**实现要点：**
- `GradientMutation._operator_stats` — 4种算子各自追踪 visits/reward
- `GradientMutation._ucb1_select_operators()` — UCB1 分数排序，inf优先探索
- `GradientMutation.record_population_fitness()` — 外部每次提交后喂入fitness值
- `GradientMutation.mutate()` — UCB1排序算子 → 按diversity boost限额执行
- 低多样性日志：记录 CV 值和 boost 后的各算子限额

### E3: MAP-Elites 质量多样性

| 参数 | 值 |
|------|-----|
| 行为空间 | direction(6) × time_horizon(3) × mechanism(4) = 72 cells |
| 每cell精英数 | Top 3 |
| 准入规则 | fitness > 该cell最差精英的fitness（或cell未满3个） |
| 淘汰规则 | 新精英准入时自动移除最差旧精英 |
| 衰减响应 | **两级响应** |
| L3_DIR_LIGHT (0.40) | 标记observing + 暂停准入 + 保留精英 |
| L4_DIR_HEAVY (0.20) | 标记blacklisted + 清空所有精英 + 永久拒绝 |
| 自适应调度 | `get_explore_exploit_schedule()` 返回策略 |
| 探索权重 | 覆盖率 < 30% → 0.6; 30%-60% → 线性递减; >60% → 1-coverage |
| 精英密度修正 | elite_density × 0.1 作为 exploit bonus |

**实现要点：**
- `FeatureCell.elites: list[dict]` — 每个cell存储top-k个精英entry
- `FeatureCell.decay_state` — "active" / "observing" / "blacklisted"
- `FeatureCell.admission_paused` — L3时暂停准入
- `FeatureCell._update_best_from_elites()` — 自动同步 best_expr/best_fitness
- `FeatureMap.add_candidate()` — 完整MAP-Elites准入/淘汰逻辑
- `FeatureMap.mark_cell_decay()` — DecayDetector回调接口
- `FeatureMap.get_explore_exploit_schedule()` — 自适应探索/利用策略
- `FeatureMap.sample_elite()` — 全网格fitness比例采样
- `FeatureMap.get_explore_targets()` — 返回空cell用于主动探索
- `_load()` / `_save()` 支持新字段序列化

### E4: 方向感知动态加权fitness + 轻量Pareto

| 参数 | 值 |
|------|-----|
| 基准权重 | 6方向预定义（Sharpe/Turnover/Complexity三目标） |
| momentum | S:0.60, T:0.20, C:0.20 |
| mean_reversion | S:0.55, T:0.30, C:0.15 |
| volatility | S:0.50, T:0.25, C:0.25 |
| statistical | S:0.55, T:0.20, C:0.25 |
| volume | S:0.50, T:0.30, C:0.20 |
| interaction | S:0.45, T:0.25, C:0.30 |
| 自适应步长 | ±0.05 |
| 自适应批次 | 每20个alpha触发一次 |
| 自适应规则 | 连续3次同类拒绝 → 对应权重+0.05, Sharpe-0.05 |
| 权重边界 | [0.10, 0.70] |
| Pareto触发 | fitness差距 < 10% 时 |
| Pareto三维 | Sharpe ↗ / Turnover ↘ / Complexity ↘ |

**实现要点：**
- `CrossoverMutationEngine._direction_weights` — 运行时方向权重（初始=基准）
- `CrossoverMutationEngine.compute_fitness()` — 加权融合 fitness
- `CrossoverMutationEngine.record_alpha_outcome()` — 记录成败，每20触发自适应
- `CrossoverMutationEngine._adapt_weights()` — 权重调整 + 归一化
- `CrossoverMutationEngine._pareto_dominates()` — Pareto支配判断
- `CrossoverMutationEngine._pareto_tie_break_sort()` — fitness排序后tie-break
- `generate_variants()` 新增 `direction` 参数用于子代方向追踪

---

## 改动文件清单

| 文件 | Phase A | Phase C | Phase D | Phase E |
|------|:---:|:---:|:---:|:---:|
| `agents/multi_agent.py` | A1 A2 A3 | C | - | - |
| `core/loop_engine.py` | A1 | C | D2 D3 | - |
| `services/brain_submitter.py` | A1 | - | D1 D2 D3 | - |
| `core/pipeline.py` | - | - | D2 | - |
| `core/loop_state.py` | - | - | D3 | - |
| `generation/prompts.py` | - | - | D3 | - |
| `validation/decay_detector.py` | - | - | D3 (新建) | - |
| `evolution/crossover_mutation.py` | - | - | - | E1 E2 E4 |
| `evolution/quality_diversity.py` | - | - | - | E3 |

---

## 实施状态

| Phase | 模块 | 状态 | 说明 |
|-------|------|:----:|------|
| A1 | AST原创性去重 | ✅ | 两阶段过滤 + _semantic_dedup_check + _archive_success |
| A2 | 假设-因子对齐 | ✅ | LLM 对齐评分 + 低分触发重新生成 |
| A3 | 复杂度限制 | ✅ | AST深度+节点数计算 + OVERFITTING_RISK标记 |
| C | 批判-修订闭环 | ✅ | 3轮 EvalAgent批判→FactorAgent修订 + critique_revise_alpha |
| D1 | LLM动态修复建议 | ✅ | llm_diagnose_failure 接入 + 失败经验记录 |
| D2 | UCB动态Slot排序 | ✅ | 三阶段动态c值 + UCB调度 |
| D3 | Alpha Decay检测 | ✅ | EWMA多指标 + 四级分级响应 |
| E1 | 锦标赛选择 + 精英保留 | ✅ | k=3锦标赛父代选择 + Top-2精英直接保留 |
| E2 | 双维度自适应变异 | ✅ | 算子级UCB1追踪 + 种群级CV多样性触发boost |
| E3 | MAP-Elites质量多样性 | ✅ | Top3精英/准入淘汰规则/L3观察L4黑名单/自适应调度 |
| E4 | 方向感知动态加权fitness | ✅ | 6方向基准权重 + 0.05步长自适应 + Pareto tie-break(10%) |

---

## 设计决策记录

1. **LLM 不猜测分数** — 基于 AlphaBench (ICLR 2026) 发现：LLM zero-shot 因子质量评估是最弱能力
2. **重复 ≠ 拒绝** — BRAIN 平台自带重复检测，重复 > 0.7 触发变异而非拒绝
3. **对齐低分 → 根源修正** — 不是让 FactorAgent 重试，而是让 IdeaAgent 重产假设
4. **复杂度超限不拒绝** — WARN 标记 + LLM 简化，不硬拒绝高质量复杂 alpha
5. **UCB 探索系数动态化** — 初期多探索，成熟期多利用，自适应调整
6. **批判闭环在 BRAIN 失败后** — 基于真实数据批判，不做臆测
7. **所有阈值/映射都由 LLM 动态决策** — 硬编码映射表仅作为 fallback
8. **锦标赛替代顺序选择** — 顺序选择导致弱个体反复配对，锦标赛基于fitness竞争保证优胜劣汰
9. **UCB1替代随机变异** — 4种变异算子不再等概率调用，UCB1追踪历史成功率动态排序，自动淘汰低效算子
10. **MAP-Elites网格替代单一fitness排序** — 3D行为描述符网格(72 cells)保证跨策略类型多样性，避免momentum方向霸权
11. **Top-3精英而非Top-1** — 同类型保留多个高质量alpha，避免过度限制多样性；准入规则必须先beat最差精英
12. **两级衰减响应而非一刀切** — L3观察中(保留精英等待恢复) → L4黑名单(清空拒绝)，避免过早丢弃可能恢复的方向
13. **方向感知权重 + 极简自适应** — 拒绝NSGA-II/超体积等学术方案，采用方向固定基准+0.05步长自适应，业务可解释、工程可调试
14. **Pareto仅做tie-break** — fitness差距<10%时才触发Pareto支配判断，避免推翻方向感知加权的主排序逻辑