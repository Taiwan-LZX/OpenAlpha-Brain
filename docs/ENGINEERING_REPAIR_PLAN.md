# OpenAlpha-Brain 工程修复文档

> 生成日期：2026-05-30 | 最后更新：2026-05-31 (v3 范式转移)
> 总计发现问题：80+ | P0: 8 | P1: ~40 | P2: ~30

---

## 模块修复路线图

```
Phase 1: 异步基础设施 ──▶ Phase 2: LLM集成深化 ──▶ Phase 3: 金融算法补全
     (P0修复)                  (P1硬编码→LLM)            (论文级算法)
```

---

## 一、异步基础设施 (P0)

### 1.1 LLM 全局互斥锁 — 最大性能瓶颈
- **文件**: `services/llm_client.py`, `core/loop_engine.py`
- **问题**: 所有 LLM 调用通过单一 mutex 串行化，即使是不同 session 的调用也被阻塞
- **影响**: 多 worker 场景下 LLM 调用完全串行，pipeline 并发形同虚设
- **修复**: 改为 per-session 锁或连接池模式

### 1.2 三层 Semaphore 叠加 — BRAIN 提交并发降到 1
- **文件**: `core/async_pipeline.py`, `core/pipeline.py`, `services/brain_submitter.py`
- **问题**: Semaphore(3) + BRAINSubmitter._semaphore + httpx 连接限制多层叠加
- **影响**: 理论 3 并发 → 实际 1 并发
- **修复**: 统一并发控制，去除冗余 Semaphore 层

### 1.3 AlphaCachePool 条件竞争
- **文件**: `core/loop_engine.py` (run_loop_pipeline)
- **问题**: `next_to_submit()` 不是原子操作，多 worker 同时调用导致 slot 超配
- **修复**: 加 asyncio.Lock 保护

### 1.4 Poll 循环缺少 401 处理
- **文件**: `services/brain_submitter.py`, `services/brain_client.py`
- **问题**: Cookie 过期后 401 不重登录，整批 alpha 失败
- **修复**: 401 触发 re-login + retry

### 1.5 Polling Loop → Event-Driven
- **文件**: `core/async_pipeline.py` (_orchestration_loop)
- **问题**: `await asyncio.sleep(0.5)` 轮询浪费 CPU
- **修复**: 用 `asyncio.Event` 实现事件驱动调度

### 1.6 Improvement Worker 轮询
- **文件**: `core/async_pipeline.py` (ImprovementWorkerPool)
- **问题**: Worker 空闲时 `sleep(1.5)` 等待
- **修复**: 用 `asyncio.Condition` 通知

### 1.7 EvalAgent 占位符
- **文件**: `agents/multi_agent.py`
- **问题**: EvalAgent 定义但从未实际向 BRAIN 提交 alpha
- **修复**: 接入真实 BRAIN 提交流程

### 1.8 cross_mutate 死代码
- **文件**: `core/pipeline.py`
- **问题**: 模块级函数定义但从未被调用
- **修复**: 删除或接入调用链

---

## 二、LLM 集成深化 (P1 — 硬编码→LLM)

### 2.1 硬编码阈值 LLM 化
| 位置 | 当前 | 修复 |
|------|------|------|
| async_pipeline.py `_classify_priority` | `sharpe > 1.0/1.5/0.5` | LLM 根据 market state + failure type 动态判断 |
| loop_engine.py 层级奖励 | `< 0.3` 跳过 BRAIN | LLM go/no-go 决策 |
| loop_engine.py mutation sharpe 估算 | `* 1.15, max 4.0` | LLM 预测改善幅度 |
| async_pipeline.py `_source_priority_bonus` | 固定 -0.5/-0.3/0.0 | LLM 根据质量动态 |
| loop_engine.py coverage 警告 | `85%` / `90%` | 基于 FeatureMap 容量 |
| loop_engine.py 去重阈值 | `0.8` / `0.9` / `0.95` | 配置化 + 自适应 |

### 2.2 文本模板动态化
- **文件**: `core/async_pipeline.py` `_build_improvement_prompt`
- **问题**: 固定模板，无策略上下文、失败模式、历史成功案例
- **修复**: 接入 `build_dynamic_context` + logic_templates + success_lib + experience_distiller

### 2.3 ReflectionEngine 正则解析 → LLM 解析
- **文件**: `learning/reflection_engine.py`
- **问题**: 用正则匹配 BRAIN 结果文本，脆弱且不完整
- **修复**: LLM 结构化解析 BRAIN 失败原因

### 2.4 template_first 跳过 LLM
- **文件**: `generation/alpha_generator.py`
- **问题**: `template_first` 模式完全不调用 LLM，直接返回模板
- **修复**: 移除或改为 LLM 填充模板

---

## 三、异步 Pipeline 架构升级 (P1)

### 3.1 Poll 退避策略
- **文件**: `core/async_pipeline.py`
- **问题**: 固定 10s 轮询间隔，无指数退避
- **修复**: 指数退避 2s→4s→8s→...→cap 30s

### 3.2 Worker 数量配置化
- **文件**: `core/loop_engine.py` `run_loop_pipeline`
- **问题**: 硬编码 `range(3)` worker
- **修复**: 从 `settings.PIPELINE_MAX_SLOTS` 读取

### 3.3 AlphaCachePool O(n²) 遍历
- **文件**: `core/loop_engine.py` `_brain_submitter_worker`
- **问题**: 多次 O(n) 遍历 `state.passed_alphas`
- **修复**: 使用 dict 索引

### 3.4 BRAIN API 重试逻辑
- **文件**: `core/loop_engine.py`, `services/brain_submitter.py`
- **问题**: 网络抖动无自动重试
- **修复**: 指数退避重试 (最多 3 次)

### 3.5 events.py emit coroutine 检测 bug
- **文件**: `core/events.py`
- **问题**: `emit` 用 `iscoroutine()` 而非 `iscoroutinefunction()`
- **修复**: 统一使用 `emit_async`

---

## 四、Multi-Agent 系统升级 (P1)

### 4.1 真实协作 → 当前是顺序管道
- **文件**: `agents/multi_agent.py`
- **问题**: 5 个 agent 顺序调用，非真正辩论/协作
- **修复**: 
  - 并行生成多种方案
  - LLM-based consensus/voting
  - Debate loop (agent A critique → agent B revise → converge)

### 4.2 AdaptiveAgent 从未调用 LLM
- **文件**: `agents/adaptive_agent.py`
- **问题**: AdaptiveAgentFactory 创建专家 agent 但从未调用 LLM
- **修复**: 接入 LLM 专家路由

### 4.3 AgentReview 形式化
- **文件**: `agents/agent_review.py`
- **问题**: 审查流程机械，无真正批判性分析
- **修复**: LLM 驱动的批判性审查

---

## 五、金融领域算法补全 (P1)

### 5.1 MAB 算法升级 (当前是加权随机)
- **文件**: `learning/mab.py`
- **问题**: 
  - 不是真正的 Multi-Armed Bandit (Thompson Sampling / UCB)
  - 简单的加权随机选择
  - 缺少 focus_area 优先级（虽有参数但不生效）
- **论文参考**: 
  - "Thompson Sampling for Contextual Bandits" (Agrawal & Goyal, 2013)
  - "A Survey on Multi-Armed Bandit for Online Decision Making" (2023)
- **修复**: 实现 Thompson Sampling + contextual features + focus_area 优先级

### 5.2 Signal Arbiter (当前是简单平均)
- **文件**: `validation/signal_arbiter.py`
- **问题**: 不是真正的信号仲裁，只是加权平均
- **论文参考**: "Online Portfolio Selection: A Survey" (Li & Hoi, 2014)
- **修复**: 实现 Bayesian signal combination + 动态权重调整

### 5.3 Overfit Detector (统计不可靠)
- **文件**: `validation/overfit_detector.py`
- **问题**: 过拟合检测方法缺乏统计严格性
- **论文参考**: 
  - "Pseudo-Mathematics and Financial Charlatanism" (Bailey et al., 2014)
  - "The Deflated Sharpe Ratio" (Lopez de Prado & Bailey, 2014)
- **修复**: Deflated Sharpe Ratio + Probabilistic Sharpe Ratio

### 5.4 缺失因子模型
- **问题**: 整个系统缺少因子模型概念，不了解 alpha 的因子暴露
- **论文参考**: "Characteristics Are Covariances" (Kelly et al., 2019)
- **修复**: Factor exposure estimation + orthogonalization

### 5.5 缺失协方差估计
- **问题**: 无协方差/相关性建模
- **论文参考**: "Risk and Asset Allocation" (Meucci, 2005)
- **修复**: Shrinkage estimator (Ledoit-Wolf)

### 5.6 HypothesisAligner 模板不足
- **文件**: `evolution/hypothesis_aligner.py`
- **问题**: 模板数量少，覆盖度低
- **修复**: 扩展模板库到 30+ 覆盖主要量化策略类型

### 5.7 ExperienceDistiller 只存不学
- **文件**: `learning/experience_distiller.py`
- **问题**: 存储经验但不实际 distill（蒸馏），无反馈闭环
- **修复**: LLM 驱动的经验提炼 → 注入 prompt

---

## 六、Drift 检测与自适应 (P1)

### 6.1 概念漂移检测缺失
- **问题**: 策略失效无自动检测
- **论文参考**: "Learning under Concept Drift: A Review" (Lu et al., 2019)
- **修复**: 滑动窗口性能监控 + ADWIN 算法

### 6.2 市场状态切换检测
- **文件**: `utils/market_state.py`
- **问题**: MarketStateInferencer 只做简单分类
- **修复**: HMM regime detection + LLM 解读

### 6.3 Platform API 变更检测
- **问题**: BRAIN API 变更无自动发现
- **修复**: Schema diff + 自动告警

### 6.4 策略去重阈值自适应
- **问题**: 三个固定阈值 (0.8/0.9/0.95)
- **修复**: 基于 F1 反馈的自适应阈值

---

## 七、知识系统 (P2)

### 7.1 RAG 检索质量
- **文件**: `knowledge/rag_engine.py`
- **问题**: 向量检索可能退化为关键词匹配
- **修复**: Hybrid search (dense + sparse) + reranking

### 7.2 Vector Store 覆盖度
- **文件**: `data/vec_store/`
- **问题**: 预构建向量可能不完整
- **修复**: 增量索引 + auto-rebuild on schema change

### 7.3 Global Knowledge 时效性
- **文件**: `knowledge/global_knowledge.py`
- **问题**: 知识库无过期机制
- **修复**: TTL + 自动刷新

---

## 八、代码质量 (P2)

### 8.1 config.py 配置项过多 (120+)
- **问题**: 大量 `*_ENABLED` 开关，默认值无来源标注
- **修复**: 分组 + 文档化 + 标注论文来源

### 8.2 `dir()` 检查 anti-pattern
- **文件**: `core/loop_engine.py`
- **问题**: `if '_pipeline' in dir()` 检查变量存在性
- **修复**: 顶部显式 `_pipeline: Any = None`

### 8.3 `import heapq` 重复导入
- **文件**: `core/async_pipeline.py`
- **修复**: 移到文件顶部

### 8.4 events.py `enabled` 非线程安全
- **文件**: `core/events.py`
- **修复**: 加 threading.Lock

---

## 修复决策树

```
异步基础设施(P0) → 先修
    │
    ├─ LLM互斥锁 ──→ per-session锁
    ├─ Semaphore叠加 ──→ 统一并发
    ├─ AlphaCachePool race ──→ asyncio.Lock
    ├─ 401处理 ──→ re-login
    ├─ 轮询→事件 ──→ asyncio.Event
    └─ EvalAgent ──→ 接入BRAIN

LLM集成深化(P1) → 二修
    │
    ├─ 硬编码阈值LLM化 (9处)
    ├─ 文本模板动态化
    ├─ Reflection正则→LLM
    └─ template_first移除

金融算法补全(P1) → 三修
    │
    ├─ MAB Thompson Sampling
    ├─ Signal Arbiter Bayesian
    ├─ Overfit Detector DSR
    ├─ 因子模型 + 协方差
    ├─ Experience Distiller 闭环
    └─ HypothesisAligner 扩展

Drift检测(P1) → 四修
    │
    ├─ 概念漂移 ADWIN
    ├─ 市场状态 HMM
    └─ API变更检测

知识系统(P2) → 五修
代码质量(P2) → 六修

模板约束范式转移(v3) → 新规划
    │
    ├─ FieldProxyMap 新建 (P0)
    ├─ 模板库扩展15个 (P0)
    ├─ MAB Arm重构600arm (P1)
    ├─ FactorAgent模板强制 (P1)
    └─ IdeaAgent字段驱动 (P1)
```

---

## 九、v3 模板约束范式转移 (NEW)

### 9.1 FieldProxyMap 字段代理图谱 (NEW)
- **文件**: `knowledge/field_proxy_map.py` (新建)
- **问题**: 当前系统无字段语义理解，IdeaAgent 自由生成假设但不理解字段之间的关系
- **影响**: LLM 生成的表达式使用随机字段，缺乏有向探索
- **论文参考**: Finding Alphas (Tulchinsky), 101 Formulaic Alphas (Kakushadze 2016)
- **修复**: 
  - 从 BRAIN API 拉取全部 7000+ 字段 metadata
  - LLM 三层标注: 语义类别 → 字段族 → 适用模板
  - JSON 主存储 + FAISS 向量索引
  - 提供 `recommend_fields_for_template()` 接口

### 9.2 AlphaLogicLibrary 模板扩展 (NEW)
- **文件**: `data/market_logics.json`, `generation/alpha_logics.py`
- **问题**: 当前仅 10 个 MarketLogic，覆盖不足
- **影响**: 模板约束覆盖率低，LLM 仍有大量自由生成空间
- **修复**: 
  - 扩展到 15 个模板，覆盖 6 方向 × 3 时间尺度
  - 每个模板 2-4 个 {placeholder} FASTEXPR 变体
  - 新增方向: lead_lag, liquidity 完整覆盖

### 9.3 MAB 探索空间重构 (NEW)
- **文件**: `learning/mab.py`, `core/loop_engine.py`
- **问题**: 当前 MAB arm 是"方向"级 (momentum, value 等)，探索粒度太粗
- **影响**: 无法区分同一方向下不同字段族的性能差异
- **修复**: 
  - Arm 改为 模板 × 字段族 (15 × 40 = 600 arm)
  - UCB 调度 + FieldProxyMap 推荐未探索字段族
  - 集成 FeatureMap explore/exploit 调度

### 9.4 FactorAgent 模板强制 (NEW)
- **文件**: `agents/multi_agent.py`
- **问题**: 当前 FactorAgent 可选择自由生成，模板只是"建议"
- **影响**: LLM 产生大量语法正确但金融意义荒谬的表达式
- **修复**: 
  - 默认路径强制模板: 选模板 → 填字段 → 微调参数
  - 自由生成仅作为连续 3 次失败后的降级路径
  - 成功自由生成后自动提炼为新模板

### 9.5 IdeaAgent 字段驱动命题 (NEW)
- **文件**: `agents/multi_agent.py`
- **问题**: 当前 IdeaAgent 自由生成金融假设（编故事），不基于实际字段
- **影响**: 假设与字段脱节，FactorAgent 难以翻译为具体表达式
- **修复**: 
  - 改为字段驱动: 从 MAB arm 获取模板 + FieldProxyMap 获取字段族
  - 生成"字段组合命题": 具体到字段族级别的假设
  - 维护字段探索历史，避免重复组合