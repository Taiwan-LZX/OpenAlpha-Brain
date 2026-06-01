# Changelog

All notable changes to OpenAlpha-Brain will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.9.0] - 2026-05-31

### 论文对齐重构 — 消灭 Anti-Patterns + 补齐 SOTA 能力

基于 AlphaBench / CodeEvolve / CogAlpha / R&D-Agent-Quant / Reflexion 等论文的架构对齐重构，
消除 v0.8.0 中残留的传统 GP Anti-Patterns，补齐 SOTA 研究已验证的关键能力。

#### P0 致命 Anti-Patterns 修复（4 项）

##### 1. SemanticCrossover 重写 — LLM 语义交叉（LMX/CodeEvolve 范式）
**文件**: [crossover_mutation.py](file:///d:\zmini\qqq\OpenAlpha-Brain\src\openalpha_brain\evolution\crossover_mutation.py)

- **删除 ~130 行传统 GP 随机子树交换代码**
- **新增 LLM 语义交叉流程**: `parse_blocks` → LLM 分析 Block A 金融语义 → AST 校验 → `reassemble`(锁 B/C)
- **3 级 Fallback**: LLM 失败 → 确定性回退(`rank+delta/corr`) → 适应度优先(高 sharpe 父代)
- **接口变更为 `async def crossover()`**

##### 2. CMAEmitter → DecayParameterTuner — 连续参数空间严格约束
**文件**: [quality_diversity.py](file:///d:\zmini\qqq\OpenAlpha-Brain\src\openalpha_brain\evolution\quality_diversity.py)

- CMA-ES search_space 从离散行为空间**严格限制为 Block C 的 3 个连续参数**:
  `{decay_window: (5,30), decay_weight: (0.5,1.0), rank_threshold: (0,1)}`
- **删除 `discretize()` 方法**
- EmitterOrchestrator 新增 **stagnation 检测**（连续 5 轮无正向反馈 → 指数衰减权重）

##### 3. ASTValidator 硬校验系统（新建）
**文件**: [ast_validator.py](file:///d:\zmini\qqq\OpenAlpha-Brain\src\openalpha_brain\validation\ast_validator.py)

- **66 算子白名单**（从 `brain_operators.json` 提取）
- **8 项 AST 级校验规则**:
  - 语法合法性 / 算子白名单 / 中性化段检测 / 衰减段检测
  - 嵌套深度 ≤ 4 / 括号平衡 / 三段式结构识别 / 参数类型检查
- 集成到 **GenerationGates**（硬拦截门控）和 **ThreeBlockTemplate.validate_assembly()**

##### 4. ThreeBlockTemplate AST 升级
**文件**: [alpha_logics.py](file:///d:\zmini\qqq\OpenAlpha-Brain\src\openalpha_brain\generation\alpha_logics.py)

- `validate_assembly()` 从关键词 `in` 检查升级为 **ASTValidator 结构验证**
- `instantiate_template()` 新增 assemble 后 **auto-fix 流程**

#### P1 严重缺失补齐（5 项）

##### 5. Experience Replay 经验回放系统
**文件**: [rag_engine.py](file:///d:\zmini\qqq\OpenAlpha-Brain\src\openalpha_brain\knowledge\rag_engine.py)

- **ExperienceCard 数据类**: 输入特征 + 修复动作 + 结果 + 元数据（含 `success_rate` 自适应学习）
- **ExperienceReplayManager**:
  - 4 维加权检索: `failure_type` 0.35 / `structure` 0.25 / `metrics` 0.20 / `field` 0.20
  - JSON 持久化
- **FactorContext 数据类**支持 RAG 定向检索
- 集成到 `brain_submitter._brain_improvement_loop`（mutator 前查询 → 提交后记录）

##### 6. Non-stationary MAB（滑动窗口 + 冷启动先验）
**文件**: [mab.py](file:///d:\zmini\qqq\OpenAlpha-Brain\src\openalpha_brain\learning\mab.py)

- **SlidingWindowUCB**: `deque(maxlen=20)` 滑动窗口 + 指数衰减加权均值(`decay_factor ** hours_ago`) + 先验冷启动
- **MABPriorInitializer**: 三源加权先验——模板经典度(0.40, 基于 101 Alphas) + 字段族数据质量(0.30) + 兼容性(0.30)
- **ComputeAllocator**: Thompson Sampling 做 Generate vs Improve 算力分配
- TemplateFamilyBandit 内层 ThompsonBandit → SlidingWindowUCB

##### 7. Reflexion 反思闭环 + ProxyEvaluator 本地预验证
**文件**: [brain_submitter.py](file:///d:\zmini\qqq\OpenAlpha-Brain\src\openalpha_brain\services\brain_submitter.py)

- **ProxyEvaluator 5 维度本地预验证**:
  - 语法存活率(30%) / 结构合规性(25%) / 字段合理性(20%) / 参数合理性(15%) / 历史相似度(10%)
  - 三级决策: ≥0.75 直接提 / 0.55-0.75 边界提 / <0.55 拒绝
- **ReflexionEngine**: `max_reflections=2` 轮 Action→Observation→Reflection→Updated Memory 循环
- `_brain_improvement_loop` 新流: classify → mutator → **Reflexion** → **ProxyEval 门控** → submit

##### 8. RAG 定向检索
**文件**: [rag_engine.py](file:///d:\zmini\qqq\OpenAlpha-Brain\src\openalpha_brain\knowledge\rag_engine.py)

- `retrieve()` 新增 `factor_context: FactorContext` 参数
- 定向模式自动融合经验回放结果

##### 9. MAB 冷启动先验
- 含在第 6 项中（MABPriorInitializer）

#### QA 修复

- 修复 `crossover()` 缺少 `async` 声明的 bug
- 修复 5 个包的 `__init__.py` 导出（validation / knowledge / learning / evolution / services）

#### 测试
- **90 项测试全通过**
- **10/10 包级 import 验证通过**

---

## [0.8.0] - 2026-05-31

### 设计宣言对齐 — WQ BRAIN 平台四原则合规修复

#### 背景与动机
基于对 WQ BRAIN 平台本质的深度认知（黑箱选秀、3 Slot 限制、66 算子+7000 字段约束），
确立了项目核心设计哲学（design_manifesto.md），定义了四条不可违反的工程铁律。

#### 原则 2 修复：三段式模板刚性锁死（C+ → A）
**文件**: [alpha_logics.py](file:///d:\zmini\qqq\OpenAlpha-Brain\src\openalpha_brain\generation\alpha_logics.py)

- **新增 `ThreeBlockTemplate` 数据结构**: `block_a`(信号/可探索) + `block_b`(中性/🔒锁死) + `block_c`(衰减/🔒锁死)
- **21 个模板全部重构为三段式格式**:
  - momentum/value/quality/size/liquidity/volatility (14个): B=`group_neutralize`, C=`ts_decay_linear`
  - lead-lag (3个): B=`group_neutralize(group_mean(...))`, C=`ts_decay_linear`
  - mean-reversion (3个): B=`group_zscore`, C=`ts_decay_linear`
- **`instantiate_template()` 重构**: 强制走 assemble→validate 流程，验证失败自动注入默认 B/C 段
- **安全保证**: LLM 无法删除或绕过 group_neutralize / ts_decay_linear

#### 原则 4 修复：AST 锁段变异改进闭环（D → A）
**文件**: [brain_submitter.py](file:///d:\zmini\qqq\OpenAlpha-Brain\src\openalpha_brain\services\brain_submitter.py)

- **新增 `FailureType` 枚举 (7 类)**: HIGH_TURNOVER / LOW_SHARPE / OVERFIT / DECAY_FAST / CORRELATION_HIGH / SYNTAX_ERROR / UNKNOWN
- **新增 `FailureClassifier`**: 纯规则引擎失败分类器（不依赖 LLM），5 条阈值规则覆盖主要场景，置信度 0.7-0.9
- **新增 `SegmentLockedMutator`**: 段锁变异器
  - `parse_blocks()`: 正则解析表达式为 A/B/C 三段
  - `mutate_block_a()`: 仅变异信号段（按 failure_type 分派策略）
  - `adjust_block_c()`: **非 LLM** 参数微调（换手率超标→窗口翻倍）
  - `reassemble()`: 安全重组保证 B/C 段完整性
  - `improve()`: 主入口 classify→parse→dispatch→reassemble→validate
- **新增 `validate_block_integrity()`**: 三段完整性验证器（neutralize 存在? decay 存在? 括号匹配? 嵌套深度?）
- **重构 `_brain_improvement_loop()`**: 从"LLM 自由重写"改为"FailureClassifier → SegmentLockedMutator 分派"

#### 四原则合规评级变化
| 原则 | 修复前 | 修复后 |
|------|--------|--------|
| P1: FieldProxyMap 约束数据空间 | A- | A- (不变) |
| P2: 三段式模板锁死 | **C+** | **A** ✅ |
| P3: MAB 调度约束探索 | B+ | B+ (不变) |
| P4: AST 锁段变异 | **D** | **A** ✅ |

### 测试
- P2 组装测试: `ts_decay_linear(group_neutralize(ts_rank(close,5)-ts_rank(close,20),sector),10)` ✅
- P2 完整性校验: neutralize=True, decay=True ✅
- P4 分类测试: low_sharpe/high_turnover/overfit 全部正确分类 ✅
- P4 解析测试: 三段解析 structure_valid=True ✅
- P4 完整性验证: valid=True, has_B=True, has_C=True ✅
- **13/13 模块全量回归测试通过**

## [0.7.0] - 2026-05-31

### MAP-Elites Layer 2 升级 — 工业级质量多样性内核

#### GridArchive（numpy 结构化 Archive）
- **GridArchive 类**: 基于 numpy ndarray 的 MAP-Elites Archive，替代原 dict[str, FeatureCell] 存储
  - `_occupied: bool[6,3,4]` + `_best_fitness: float32[dims]` + `_solution_count: int32[dims]`
  - `add()` 单解插入 + `batch_add()` 向量化批量插入
  - `sample_elite(k, method)` 支持 uniform / fitness_proportional / novelty 三种采样模式
  - `as_array()` 导出结构化 numpy 字典用于分析
  - QD Metrics 属性：`qd_score`(Σfitness) / `coverage`(占用率) / `normalized_qd_score`∈[0,1] / `max_fitness`
  - `get_empty_cells()` + `get_frontier_cells()` 用于探索导向和 IsoLine 计算
  - `save_state()` / `load_state()` 持久化（.npz + JSON 双轨）

#### Emitter 系统（参考 pyribs 设计）
- **BaseEmitter**: 抽象基类，定义 `ask(archive, n)→list[EmitterOutput]` / `tell(archive, behaviors, fitnesses, metadatas)` 接口
- **ExploreEmitter**: 探索型 Emitter，边界 cell 优先选择（boundary_preference=0.7），大跨度变异 hint
- **ExploitEmitter**: 利用型 Emitter，softmax(temperature) 加权采样高 fitness cell，邻域微调变异
- **CMAEvolutionStrategy**: 简化 CMA-ES 适配器，mean/σ/C 协方差矩阵演化 + discretize 连续→离散映射
- **CMAEmitter**: CMA-ES 包装为 Emitter 接口，混合方向引导与局部利用
- **IsoLineDirectionCalculator**: 梯度场分析计算改进方向向量，稀疏 archive(<3 cells) 自动降级为加权质心方向
- **EmitterOrchestrator**: 多 Emitter 编排器，UCB1 自适应权重调整（reward/count + exploration bonus）

#### FeatureMap 向后兼容升级
- 内部委托到 `self._archive: GridArchive`，所有公开方法签名不变
- 新增属性：`.archive` (GridArchive 实例) / `.qd_score` / `.coverage` / `.normalized_qd_score`
- 新增方法：`.batch_add_candidates()` / `.get_frontier_targets()`
- 行为空间枚举映射：`behavior_to_index(direction, time_horizon, mechanism) → tuple[int,int,int]`

### 测试
- GridArchive 全功能测试通过（add/batch_add/sample_elite/QD metrics/empty/frontier/persistence）
- CMA-ES 生命周期测试通过（ask→discretize→tell 多代演化）
- 4 种 Emitter 独立测试 + Orchestrator 编排测试通过
- FeatureMap 向后兼容测试通过（StrategyFeatures 接口不变）
- 全量模块导入回归测试保持通过

## [0.6.0] - 2026-05-31

### 算法质量重构 — 工业最佳实践对标与防御性加固

#### P0 修复（严重阻塞级缺陷）
- **SemanticCrossover (D→B+)**: 修复 L440-442 静默跳过与父代相同子代的 bug，新增详细 WARNING 日志记录匹配父代、交叉点、表达式对比信息
- **TrajectoryMutationV2 (F/D→B)**: 实现 4 级 Fallback 策略（参考 OpenEvolve 框架）：Level1 LLM变异 → Level2 AST表达式变异 → Level3 参数微调 → Level4 原始轨迹保底，**绝不返回空列表**
- **GradientMutation (C→B+)**: 修复 diversity=0 时行为未定义问题（DIVERSITY_EPSILON=1e-6 下界保护），9 个硬编码 magic numbers 参数化为类常量 + _validate_params() 校验
- **GenerationGates (C+→A-)**: H↔E 对齐从单一 R² 分数升级为多维度验证（R²:semantic 7:3 加权 + 方向一致性60%惩罚 + 字段匹配率20%惩罚），边界值(0.6-0.75)自动 WARNING；Holistic 检查新增四维检测（复杂度/过拟合风险/衰减指标/稳定性）

#### P1 修复（中等级别缺陷）
- **StrategyClassifier (D→B)**: find_complementary() 从无序 append 升级为多维度 relevance_score 降序排列（方向基础分0.7+Sharpe加成/机制基础分0.5+协同效应/时间跨度基础分0.45+多样化加成），TOP-K=5 截断，低分(<0.5)预警
- **HypothesisAligner (D→B)**: fallback scoring 从 3 布尔条件升级为 4 信号融合（字段重叠0.30/操作符相似0.25/结构复杂度0.20/方向关键词0.25），16 种模板对齐分数校准（双轨：原始分+校准分）

#### 防御性日志（22 处跨模块加固）
- **loop_engine.py** (5处): 代码块提取异常/模板获取失败/市场状态注入失败/技能库注入失败/策略去重失败 — 统一 `[DEFENSIVE_LOG]` 标记
- **brain_submitter.py** (2处): 多样性注入 feature_map 获取失败 / circuit breaker 异常变量捕获
- **validator.py** (4处): complexity/originality 检查跳过警告 + 通过/失败区分日志（clean vs partial-gates-skipped）
- **rag_engine.py** (11处): budget耗尽丢弃/rerank降级/空结果检测(ops/fields/full)/权重持久化失败/知识库空存储(成功案例/失败修复)

#### 开源参考研究
- 分析 pyribs (CMA-ME MAP-Elites) 用于 FeatureMap 未来升级路径
- 分析 CodeEvolve / OpenEvolve (LLM驱动代码进化) 的语义交叉和反馈驱动策略
- 分析 pybandits (PlaytikaOSS 生产级 Thompson Sampling) 确认现有 MAB 实现(A-)仅需少量加固

### 测试
- 全量 11 核心模块导入回归测试通过（SemanticCrossover, GradientMutation, TrajectoryMutationV2, GenerationGates, StrategyClassifier, HypothesisAligner, HierarchicalMAB, loop_engine, loop_state, BrainSubmitter, LLMClient）

## [0.5.0] - 2026-05-31

### 闭环修复 - 全模块数据回流打通

#### P0 修复（闭环完全断开）
- 修复 _brain_improvement_loop critique-revise 产出未写回 alpha 对象，自学习改进闭环现已完整
- 修复自动降级路径 initial_result=None 导致立即返回，BRAIN 饱和时逃生通道现已生效
- 修复 TrajectoryCrossover 3 个产出变量写入后零读取，轨迹交叉洞察现已注入 LLM 上下文和方向选择
- 新增 ExperienceDistiller._store_pattern() 方法，衰变经验存储不再静默失败
- 接入 DecayDetector.is_blacklisted() 到 Scheduler 方向选择，黑名单方向现已被过滤

#### P1 修复（闭环效率提升）
- 激活 ComplexityController P90 自适应阈值，成功 alpha 复杂度现被记录和自适应
- GenerationGates 使用 apply_with_retry() 自动重试，门控失败不再直接丢弃
- HypothesisAligner.build_alignment_feedback() 反馈注入 _brain_feedback_buffer
- 移除 PASS 路径重复的 experience/evidence distillation（6路→4路并行）
- FAIL 路径调用 run_post_brain_processing 补上策略分类/字段补充/工具冲突检测
- 激活 FeatureMap 5 个死代码方法（get_explore_targets/sample_elite/advance_generation 等）
- StrategyClassifier 互补建议 reward 提升 7.5 倍（0.02→0.15），top profiles 注入生成 prompt
- 移除 generator.py 中 add_success(sharpe=0.0) 的成功库污染

#### 质检修复
- 修复 scheduler decay_detector 属性名不匹配（添加 property/setter）
- 修复 loop_state.py add_success 方法不存在（改为 add_case）
- 修复 brain_submitter simulation_payload 空 dict 处理

### 测试
- 195 passed, 9 skipped

## [0.1.0] - 2026-05-30

### Added — Initial Open-Source Release

- Unified `openalpha` CLI as the single entry point (`openalpha run`, `openalpha status`)
- `src`-layout package restructuring (`src/openalpha_brain/`)
- Modular architecture: `cli`, `core`, `generation`, `validation`, `evolution`, `services`, `knowledge`, `learning`, `agents`, `data`, `utils`
- Autonomous alpha generation loop with LLM-driven closed-loop mutation
- Multi-provider LLM support (Groq, Gemini, OpenAI, Anthropic, LM Studio)
- WorldQuant BRAIN REST API integration (submit, poll, gate-check)
- Local syntax validation and IQC metric estimation
- POMDP memory architecture for anti-crowding exploration
- RAG engine with vector indexing for alpha knowledge retrieval
- Multi-agent coordination system
- Quality-diversity evolution and trajectory mutation strategies
- Experience distiller for learning from past sessions
- Multi-armed bandit (MAB) for model/strategy selection
- Web dashboard with real-time session monitoring (FastAPI + uvicorn)
- Docker deployment (Dockerfile + docker-compose.yml)
- MIT license