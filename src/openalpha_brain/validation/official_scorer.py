"""
WQ BRAIN 官方评分系统适配器

核心能力：
  1. 综合评分计算（Sharpe + Fitness + Turnover + Drawdown + Checks）
  2. 标准化评级（A+ 到 F）
  3. 多维度归因分析
  4. 改进建议生成

来源：
  - official_scoring.py: OfficialScoringSystem 核心逻辑
  - alpha_checks.py: AlphaCheckRegistry 检查项权重
  - 实践经验：WQ 平台评分标准

使用场景：
  - 在 _on_wq_completion() 收到 WQ 结果后调用
  - 提供标准化的评分报告用于决策

Usage::
    scorer = OfficialScoringAdapter()
    metrics = {
        "sharpe": 1.75,
        "fitness": 1.25,
        "turnover": 0.25,
        "returns": 0.05,
        "drawdown": 0.15,
        "checks": [
            {"name": "sharpe_positive", "value": 1.75, "limit": 1.25, "result": True},
            {"name": "turnover_platform", "value": 0.25, "limit": 0.70, "result": True},
        ]
    }
    report = scorer.compute_score(metrics)
    print(f"总分: {report.overall_score:.1f} ({report.grade})")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# ── 数据结构 ──────────────────────────────────────────────────────────────


@dataclass
class ScoreReport:
    """完整评分报告"""

    overall_score: float  # 0-100
    grade: str  # A+ to F
    breakdown: dict  # {sharpe_score, fitness_score, ...}
    passed: bool  # overall >= threshold (默认 60)
    details: str = ""
    improvement_hints: list[str] = field(default_factory=list)

    # 🆕 增强字段 (v1.0.0+)
    advanced_metrics: AdvancedMetrics | None = None  # 高级指标
    multi_layer_result: dict | None = None  # 多层评估结果
    factor_profile: str = ""  # 因子画像分类

    # 🆕 策略 D 字段 (v2.0.0+)
    icir_metrics: ICIRMetrics | None = None  # ICIR 推断指标
    multi_faceted_reward: MultiFacetedReward | None = None  # 多面奖励

    def to_dict(self) -> dict:
        base = {
            "overall_score": round(self.overall_score, 2),
            "grade": self.grade,
            "passed": self.passed,
            "breakdown": {k: round(v, 2) if isinstance(v, float) else v for k, v in self.breakdown.items()},
            "details": self.details,
            "improvement_hints": self.improvement_hints,
        }

        if self.advanced_metrics:
            base["advanced_metrics"] = self.advanced_metrics.to_dict()
        if self.multi_layer_result:
            base["multi_layer_result"] = self.multi_layer_result
        if self.factor_profile:
            base["factor_profile"] = self.factor_profile

        # 🆕 策略 D 新增字段
        if self.icir_metrics:
            base["icir_metrics"] = self.icir_metrics.to_dict()
        if self.multi_faceted_reward:
            base["multi_faceted_reward"] = self.multi_faceted_reward.to_dict()

        return base

    def __str__(self) -> str:
        lines = [
            f"Score Report: {self.overall_score:.1f}/100 ({self.grade})",
            f"Status: {'PASS' if self.passed else 'FAIL'}",
            "",
            "Breakdown:",
        ]
        for key, value in self.breakdown.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.1f}")
            else:
                lines.append(f"  {key}: {value}")

        if self.improvement_hints:
            lines.append("")
            lines.append("Improvement Hints:")
            for hint in self.improvement_hints[:5]:
                lines.append(f"  → {hint}")

        # 🆕 显示增强信息
        if self.factor_profile:
            lines.append("")
            lines.append(f"Factor Profile: {self.factor_profile}")

        if self.advanced_metrics and self.advanced_metrics.risk_level != "MEDIUM":
            lines.append(f"Risk Level: {self.advanced_metrics.risk_level}")

        return "\n".join(lines)


@dataclass
class CheckItem:
    """单个检查项"""

    name: str
    value: Any
    limit: Any
    result: bool
    severity: str = "ERROR"  # ERROR | WARNING | INFO
    weight: float = 1.0  # 权重（用于扣分计算）

    @property
    def passed(self) -> bool:
        return self.result


@dataclass
class AdvancedMetrics:
    """高级评估指标 (来自 AlphaBench metrics.py)

    当 WQ 不直接提供这些指标时，
    可从 Sharpe/Fitness/Turnover 推断或标记为 unavailable
    """

    # IC 系列 (信息系数)
    ic: float | None = None  # Information Coefficient (Pearson)
    rank_ic: float | None = None  # Rank IC (Spearman)
    ir: float | None = None  # Information Ratio (mean_IC / std_IC)
    icir: float | None = None  # ICIR (mean_IC * sqrt(n) / std_IC)
    rank_icir: float | None = None  # Rank ICIR

    # 收益分解
    quantile_returns: dict | None = None  # {Q1: -2%, Q2: -0.5%, ..., Q5: 3%, spread: 5%}

    # 稳定性指标
    win_rate: float | None = None  # 月度胜率 (>0 为正收益月份占比)
    stability: float | None = None  # IC 时间序列稳定性 (1 - CV_IC)

    # 综合诊断
    overall_diagnosis: str = ""  # 诊断结论 (如 "HIGH_SHARPE_LOW_FITNESS")
    risk_level: str = "MEDIUM"  # LOW / MEDIUM / HIGH / CRITICAL
    improvement_potential: float = 0.0  # 改进潜力 (0-1, 越高越值得投入资源改进)

    def to_dict(self) -> dict:
        result = {
            "ic": self.ic,
            "rank_ic": self.rank_ic,
            "ir": self.ir,
            "icir": self.icir,
            "rank_icir": self.rank_icir,
            "quantile_returns": self.quantile_returns,
            "win_rate": self.win_rate,
            "stability": self.stability,
            "overall_diagnosis": self.overall_diagnosis,
            "risk_level": self.risk_level,
            "improvement_potential": round(self.improvement_potential, 3),
        }
        return {k: v for k, v in result.items() if v is not None and v != "" and v != "MEDIUM"}


@dataclass
class ICIRMetrics:
    """ICIR 推断指标 (策略 D 核心组件)

    从 WQ 返回的 Sharpe/Fitness/Turnover 反推 ICIR 指标。
    基于 AlphaBench 和 RD-Agent 的评估框架。

    Attributes:
        ic: Information Coefficient (推断值)
        rank_ic: Rank IC (Spearman 相关性)
        ir: Information Ratio = mean(IC)/std(IC)
        icir: IC * IR 综合指标 (核心质量指标)
        predicted_fitness: 基于 ICIR 的 fitness 预测值
        confidence: 推断置信度 (0-1)
    """

    ic: float = 0.0
    rank_ic: float = 0.0
    ir: float = 0.0
    icir: float = 0.0
    predicted_fitness: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ic": round(self.ic, 6),
            "rank_ic": round(self.rank_ic, 6),
            "ir": round(self.ir, 4),
            "icir": round(self.icir, 4),
            "predicted_fitness": round(self.predicted_fitness, 4),
            "confidence": round(self.confidence, 3),
        }

    @property
    def is_high_icir(self) -> bool:
        """判断是否为高 ICIR 因子 (阈值 > 1.0)"""
        return self.icir > 1.0

    @property
    def is_low_fitness_high_icir(self) -> bool:
        """判断是否为高 ICIR 低 Fitness 类型"""
        return self.is_high_icir and self.predicted_fitness < 1.0


@dataclass
class MultiFacetedReward:
    """多面奖励函数 (参考 Alpha SAGE arXiv 2509.25055)

    从多个维度综合评估因子质量，不仅依赖单一 Sharpe 指标。

    权重配置:
      - signal_quality: 0.30 (Sharpe-based)
      - stability: 0.25 (Fitness-based)
      - efficiency: 0.20 (Return-per-turnover)
      - uniqueness: 0.15 (Decorrelation bonus)
      - simplicity: 0.10 (Complexity penalty)

    Attributes:
        signal_quality: 信号质量分量 (基于 Sharpe)
        stability: 稳定性分量 (基于 Fitness)
        efficiency: 效率分量 (|R|/TO)
        uniqueness: 独特性分量 (去相关奖励)
        simplicity: 简单性惩罚 (复杂度惩罚)
        total_reward: 加权总分 (0-1)
    """

    signal_quality: float = 0.0
    stability: float = 0.0
    efficiency: float = 0.0
    uniqueness: float = 0.0
    simplicity: float = 0.0
    total_reward: float = 0.0

    # 默认权重
    WEIGHTS = {
        "signal_quality": 0.30,
        "stability": 0.25,
        "efficiency": 0.20,
        "uniqueness": 0.15,
        "simplicity": 0.10,
    }

    def to_dict(self) -> dict:
        return {
            "signal_quality": round(self.signal_quality, 4),
            "stability": round(self.stability, 4),
            "efficiency": round(self.efficiency, 4),
            "uniqueness": round(self.uniqueness, 4),
            "simplicity": round(self.simplicity, 4),
            "total_reward": round(self.total_reward, 4),
        }

    @property
    def is_efficient_alpha(self) -> bool:
        """判断是否为高效因子 (efficiency > 0.3)"""
        return self.efficiency > 0.3

    @property
    def dominant_dimension(self) -> str:
        """识别主导维度"""
        dimensions = {
            "signal_quality": self.signal_quality,
            "stability": self.stability,
            "efficiency": self.efficiency,
            "uniqueness": self.uniqueness,
        }
        return max(dimensions, key=dimensions.get)


# ── 评分配置常量（来自 official_scoring.py 和 alpha_checks.py）───────────

# BRAIN 官方阈值（来自 alpha_checks.py）
BRAIN_THRESHOLDS = {
    "min_sharpe": 1.25,  # Delay >= 1 的最小 Sharpe
    "min_sharpe_delay0": 2.0,  # Delay = 0 的最小 Sharpe
    "min_fitness": 1.0,  # Delay >= 1 的最小 Fitness
    "min_fitness_delay0": 1.3,  # Delay = 0 的最小 Fitness
    "min_turnover": 0.01,  # 最小换手率 (1%)
    "platform_max_turnover": 0.70,  # 平台最大换手率 (70%)
    "target_max_turnover": 0.30,  # 目标最大换手率 (30%)
    "max_self_correlation": 0.70,  # 最大自相关
    "max_weight_concentration": 0.10,  # 最大权重集中度
    "max_drawdown": 0.25,  # 最大回撤
    "min_margin_bps": 4.0,  # 最小利润率 (bps)
}

# 评分维度权重（总计 100 分）
SCORE_WEIGHTS = {
    "sharpe_score": 40,  # Sharpe 分数 (0-40)
    "fitness_score": 30,  # Fitness 分数 (0-30)
    "turnover_score": 15,  # 换手率分数 (0-15)
    "drawdown_score": 10,  # 回撤分数 (0-10)
    "checks_penalty": 5,  # 检查项扣分 (0-5)
}

# 增强版权重 (当有高级指标数据时使用)
ENHANCED_SCORE_WEIGHTS = {
    "sharpe_score": 30,  # 降低 (原40→30)，因为有了更细的 IC 维度
    "fitness_score": 20,  # 降低 (原30→20)
    "turnover_score": 12,  # 微调 (原15→12)
    "drawdown_score": 8,  # 微调 (原10→8)
    "checks_penalty": 5,  # 保持
    # 🆕 新增维度 (当数据可用时激活)
    "ic_score": 10,  # IC 绝对值评分
    "icir_score": 8,  # ICIR 稳定性评分
    "stability_score": 7,  # 时间序列稳定性
}

# 评级标准
GRADE_THRESHOLDS = [
    (90, "A+"),
    (80, "A"),
    (70, "A-"),
    (60, "B+"),
    (50, "B"),
    (40, "B-"),
    (30, "C"),
    (20, "D"),
    (0, "F"),
]

# 默认通过阈值
DEFAULT_PASS_THRESHOLD = 60.0


class OfficialScoringAdapter:
    """WQ BRAIN 官方评分系统适配器

    计算逻辑：
      1. Sharpe score (0-40): sharpe * 20, capped at 40
         - sharpe >= 2.0 → 40 分 (满分)
         - sharpe >= 1.25 → 25 分 (及格线)
         - sharpe < 1.25 → 线性扣分

      2. Fitness score (0-30): fitness * 15, capped at 30
         - fitness >= 2.0 → 30 分 (满分)
         - fitness >= 1.0 → 15 分 (及格线)

      3. Turnover score (0-15): 理想范围 5%-50%
         - 5% <= turnover <= 30% → 15 分 (最优)
         - 30% < turnover <= 50% → 10 分
         - turnover < 5% 或 > 50% → 线性扣分

      4. Drawdown score (0-10): 越低越好
         - drawdown <= 0.10 → 10 分
         - drawdown <= 0.20 → 7 分
         - drawdown <= 0.25 → 5 分
         - drawdown > 0.25 → 线性扣分

      5. Checks penalty (0-5): 每个失败检查 -1 分
         - ERROR 级别失败: -1 分
         - WARNING 级别失败: -0.5 分

    Usage::
        scorer = OfficialScoringAdapter()
        report = scorer.compute_score(metrics)
        print(report)
    """

    def __init__(
        self,
        pass_threshold: float = DEFAULT_PASS_THRESHOLD,
        custom_weights: dict[str, int] | None = None,
    ) -> None:
        """
        Args:
            pass_threshold: 通过阈值（默认 60 分）
            custom_weights: 自定义评分权重（可选）
        """
        self._threshold = pass_threshold
        self._weights = custom_weights or SCORE_WEIGHTS

    def compute_score(self, metrics: dict) -> ScoreReport:
        """
        计算综合评分

        Args:
            metrics: {
                sharpe: float,           # Sharpe ratio
                fitness: float,          # Fitness ratio
                turnover: float,         # Turnover (0-1)
                returns: float,          # Returns
                drawdown: float,         # Max drawdown
                checks: list[dict],      # [{name, value, limit, result}]
                delay: int = 1,          # Delay (optional)
            }

        Returns:
            ScoreReport with overall_score and breakdown
        """
        # 提取指标
        sharpe = float(metrics.get("sharpe", 0))
        fitness = float(metrics.get("fitness", 0))
        turnover = float(metrics.get("turnover", 0))
        drawdown = abs(float(metrics.get("drawdown", 0)))
        checks = metrics.get("checks", [])
        delay = int(metrics.get("delay", 1))

        # 计算各维度分数
        breakdown = {}

        # 1. Sharpe 分数 (0-40)
        breakdown["sharpe_score"] = self._calc_sharpe_score(sharpe, delay)

        # 2. Fitness 分数 (0-30)
        breakdown["fitness_score"] = self._calc_fitness_score(fitness, delay)

        # 3. Turnover 分数 (0-15)
        breakdown["turnover_score"] = self._calc_turnover_score(turnover)

        # 4. Drawdown 分数 (0-10)
        breakdown["drawdown_score"] = self._calc_drawdown_score(drawdown)

        # 5. Checks 扣分 (0-5)
        breakdown["checks_penalty"] = self._calc_checks_penalty(checks)

        # 计算总分
        total = sum(breakdown.values())
        overall_score = max(0.0, min(100.0, total))

        # 评级
        grade = self.grade(overall_score)

        # 是否通过
        passed = overall_score >= self._threshold

        # 生成改进建议
        hints = self._generate_improvement_hints(metrics, breakdown)

        # 详细说明
        details = self._build_details(breakdown, metrics)

        # 🆕 增强评估 (v1.0.0+)
        advanced = self.infer_advanced_metrics(metrics)
        multi_layer = self.multi_layer_evaluate(metrics)
        profile = self.classify_factor_profile(metrics)

        # 🆕 策略 D: 计算 ICIR 和多面奖励
        icir = self.infer_icir_metrics(metrics)
        mfr = self.compute_multi_faceted_reward(
            metrics,
            expr=metrics.get("expression", ""),
        )

        return ScoreReport(
            overall_score=overall_score,
            grade=grade,
            breakdown=breakdown,
            passed=passed,
            details=details,
            improvement_hints=hints,
            advanced_metrics=advanced,  # 🆕 v1.0
            multi_layer_result=multi_layer,  # 🆕 v1.0
            factor_profile=profile,  # 🆕 v1.0
            icir_metrics=icir,  # 🆕 v2.0 策略 D
            multi_faceted_reward=mfr,  # 🆕 v2.0 策略 D
        )

    def grade(self, overall_score: float) -> str:
        """评级: A+/A/A-/B+/B/B-/C/D/F"""
        for threshold, letter in GRADE_THRESHOLDS:
            if overall_score >= threshold:
                return letter
        return "F"

    # ══════════════════════════════════════════════════════════════════════
    # 各维度评分方法
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _calc_sharpe_score(sharpe: float, delay: int = 1) -> float:
        """计算 Sharpe 分数 (0-40)

        来自 official_scoring.py empirical_score 逻辑
        """
        min_sharpe = BRAIN_THRESHOLDS["min_sharpe_delay0"] if delay == 0 else BRAIN_THRESHOLDS["min_sharpe"]

        if sharpe <= 0:
            return 0.0
        elif sharpe >= 2.0:
            return 40.0
        elif sharpe >= min_sharpe:
            # 线性映射: min_sharpe → 20分, 2.0 → 40分
            ratio = (sharpe - min_sharpe) / (2.0 - min_sharpe)
            return 20.0 + ratio * 20.0
        else:
            # 低于最低要求，线性扣分
            ratio = sharpe / min_sharpe
            return ratio * 20.0

    @staticmethod
    def _calc_fitness_score(fitness: float, delay: int = 1) -> float:
        """计算 Fitness 分数 (0-30)

        来自 alpha_checks.py fitness_minimum 检查逻辑
        """
        min_fitness = BRAIN_THRESHOLDS["min_fitness_delay0"] if delay == 0 else BRAIN_THRESHOLDS["min_fitness"]

        if fitness <= 0:
            return 0.0
        elif fitness >= 2.0:
            return 30.0
        elif fitness >= min_fitness:
            # 线性映射: min_fitness → 15分, 2.0 → 30分
            ratio = (fitness - min_fitness) / (2.0 - min_fitness)
            return 15.0 + ratio * 15.0
        else:
            ratio = fitness / min_fitness
            return ratio * 15.0

    @staticmethod
    def _calc_turnover_score(turnover: float) -> float:
        """计算换手率分数 (0-15)

        来自 alpha_checks.py turnover_platform/turnover_quality 检查逻辑
        """
        if turnover <= 0:
            return 0.0

        min_t = BRAIN_THRESHOLDS["min_turnover"]
        target_t = BRAIN_THRESHOLDS["target_max_turnover"]
        max_t = BRAIN_THRESHOLDS["platform_max_turnover"]

        if min_t <= turnover <= target_t:
            # 理想范围: 5%-30%
            return 15.0
        elif target_t < turnover <= 0.50:
            # 可接受范围: 30%-50%
            ratio = (turnover - target_t) / (0.50 - target_t)
            return 15.0 - ratio * 5.0
        elif turnover > 0.50 and turnover <= max_t:
            # 接近上限: 50%-70%
            ratio = (turnover - 0.50) / (max_t - 0.50)
            return 10.0 - ratio * 7.0
        elif turnover > max_t:
            # 超过平台限制
            return max(0.0, 3.0 - (turnover - max_t) * 10)
        else:
            # 低于最小值: < 1%
            ratio = turnover / min_t
            return ratio * 10.0

    @staticmethod
    def _calc_drawdown_score(drawdown: float) -> float:
        """计算回撤分数 (0-10)，越低越好

        来自 alpha_checks.py drawdown_limit 检查逻辑
        """
        if drawdown <= 0.05:
            return 10.0
        elif drawdown <= 0.10:
            return 9.0
        elif drawdown <= 0.15:
            return 7.0
        elif drawdown <= 0.20:
            return 5.0
        elif drawdown <= 0.25:
            return 3.0
        elif drawdown <= 0.35:
            ratio = (drawdown - 0.25) / 0.10
            return 3.0 - ratio * 2.0
        else:
            return max(0.0, 1.0 - (drawdown - 0.35) * 5)

    @classmethod
    def _calc_checks_penalty(cls, checks: list[dict]) -> float:
        """计算检查项扣分 (0-5)

        来自 alpha_checks.py AlphaCheckRegistry.evaluate() 逻辑
        """
        penalty = 0.0
        max_penalty = 5.0

        for check in checks:
            if not check.get("result", True):
                severity = check.get("severity", "ERROR").upper()
                if severity == "ERROR":
                    penalty += 1.0
                elif severity == "WARNING":
                    penalty += 0.5
                # INFO 级别不扣分

        return -min(penalty, max_penalty)  # 返回负值（扣分）

    # ══════════════════════════════════════════════════════════════════════
    # 辅助方法
    # ══════════════════════════════════════════════════════════════════════

    def _generate_improvement_hints(self, metrics: dict, breakdown: dict) -> list[str]:
        """生成改进建议（来自 official_scoring.py _generate_improvement_hints）"""
        hints = []

        sharpe = float(metrics.get("sharpe", 0))
        fitness = float(metrics.get("fitness", 0))
        turnover = float(metrics.get("turnover", 0))
        drawdown = abs(float(metrics.get("drawdown", 0)))
        checks = metrics.get("checks", [])

        # Sharpe 建议
        if breakdown.get("sharpe_score", 0) < 25:
            if sharpe < 1.25:
                hints.append(
                    f"Sharpe ({sharpe:.2f}) 低于 BRAIN 最低要求 (1.25)。建议: 缩短衰减窗口、更换 universe、添加风险控制"
                )
            elif sharpe < 1.75:
                hints.append(
                    f"Sharpe ({sharpe:.2f}) 未达理想水平。考虑: 使用 ts_decay_linear 替代 ts_mean、优化信号组合"
                )

        # Fitness 建议
        if breakdown.get("fitness_score", 0) < 18:
            if fitness < 1.0:
                hints.append(f"Fitness ({fitness:.2f}) 低于阈值。建议: 降低换手率以提高 returns/turnover 比值")
            else:
                hints.append(f"Fitness ({fitness:.2f}) 有提升空间。尝试: 增加窗口长度、减少频繁触发信号")

        # Turnover 建议
        if breakdown.get("turnover_score", 0) < 12:
            if turnover > 0.50:
                hints.append(
                    f"换手率 ({turnover:.1%}) 过高，接近 BRAIN 上限 (70%)。"
                    "建议: 使用更长窗口 (ts_mean, ts_decay_linear)、降低信号频率"
                )
            elif turnover < 0.05:
                hints.append(f"换手率 ({turnover:.1%}) 过低。可能信号过于平滑或触发条件过严")

        # Drawdown 建议
        if breakdown.get("drawdown_score", 0) < 6:
            hints.append(
                f"回撤 ({drawdown:.1%}) 较大。建议: 添加 group_neutralize(subindustry)、使用 winsorize、增加止损机制"
            )

        # 检查项失败建议
        failed_checks = [c for c in checks if not c.get("result", True)]
        for check in failed_checks[:3]:
            name = check.get("name", "")
            value = check.get("value")
            limit = check.get("limit")

            if name == "sharpe_positive":
                hints.append(f"Sharpe 检查未通过: {value} < {limit}")
            elif name == "turnover_platform":
                hints.append(f"换手率超出平台限制: {value} > {limit}")
            elif name == "self_correlation":
                hints.append(f"自相关过高: {value} >= {limit}，建议更改特征组合")
            elif name == "weight_concentration":
                hints.append(f"权重集中度过高: {value} > {limit}")

        return hints[:8]  # 最多返回 8 条建议

    @staticmethod
    def _build_details(breakdown: dict, metrics: dict) -> str:
        """构建详细说明"""
        parts = []

        sharpe = metrics.get("sharpe", 0)
        fitness = metrics.get("fitness", 0)
        turnover = metrics.get("turnover", 0)
        drawdown = abs(metrics.get("drawdown", 0))
        checks = metrics.get("checks", [])

        parts.append(f"Sharpe={sharpe:.3f} → {breakdown.get('sharpe_score', 0):.1f}/40")
        parts.append(f"Fitness={fitness:.3f} → {breakdown.get('fitness_score', 0):.1f}/30")
        parts.append(f"Turnover={turnover:.1%} → {breakdown.get('turnover_score', 0):.1f}/15")
        parts.append(f"Drawdown={drawdown:.1%} → {breakdown.get('drawdown_score', 0):.1f}/10")

        failed_count = sum(1 for c in checks if not c.get("result", True))
        total_count = len(checks)
        parts.append(
            f"Checks={total_count - failed_count}/{total_count} passed → {breakdown.get('checks_penalty', 0):.1f}/5"
        )

        return "; ".join(parts)

    # ══════════════════════════════════════════════════════════════════════
    # 🆕 策略 D: ICIR 推断与多面奖励函数 (v2.0.0+)
    # ══════════════════════════════════════════════════════════════════════

    def infer_icir_metrics(self, metrics: dict) -> ICIRMetrics:
        """从 WQ 返回的 Sharpe/Fitness/Turnover 反推 ICIR 指标

        基于经验公式和 AlphaBench 的评估框架：
          - IC ≈ sharpe * sqrt(turnover) * 0.1 (近似)
          - IR ≈ sharpe / sqrt(252) * sqrt(turnover_factor)
          - ICIR = IC * IR * sqrt(N) 其中 N 是回测天数

        Args:
            metrics: WQ 返回的基本指标字典 {
                sharpe, fitness, turnover, returns, drawdown
            }

        Returns:
            ICIRMetrics 数据类实例
        """
        sharpe = float(metrics.get("sharpe", 0))
        fitness = float(metrics.get("fitness", 0))
        turnover = max(float(metrics.get("turnover", 0.01)), 0.01)  # 避免除零
        float(metrics.get("returns", 0))

        result = ICIRMetrics()

        if sharpe <= 0:
            return result

        # ── IC 推断 ──
        # 经验公式: IC ≈ sharpe * sqrt(turnover) * 0.1
        # 考虑换手率对 IC 的影响：高换手通常意味着更频繁的交易信号
        turnover_factor = math.sqrt(min(turnover, 1.0))
        estimated_ic = sharpe * turnover_factor * 0.1
        result.ic = round(estimated_ic, 6)

        # Rank IC 通常比 IC 高 5-15% (更稳健)
        result.rank_ic = round(estimated_ic * 1.1, 6)

        # ── IR 推断 ──
        # IR = mean(IC) / std(IC), 通常在 0.5-2.0
        # 从 Fitness 推断: Fitness 高意味着 IR 较高
        if fitness > 0:
            # IR 与 Fitness 正相关，但有上限
            base_ir = min(fitness * 0.8, 2.5)
            # Sharpe 高时 IR 也倾向于高
            ir_from_sharpe = min(sharpe / 1.5, 2.0)
            # 加权平均
            estimated_ir = base_ir * 0.6 + ir_from_sharpe * 0.4
            result.ir = round(estimated_ir, 4)

        # ── ICIR 计算 ──
        # ICIR = IC * IR * sqrt(N)，假设 N=60 (5年月度数据)
        if result.ic != 0 and result.ir > 0:
            n_periods = 60  # 假设回测期数
            icir_value = abs(result.ic) * result.ir * math.sqrt(n_periods) / 10
            result.icir = round(icir_value, 4)

        # ── Predicted Fitness ──
        # 基于 ICIR 预测 Fitness
        if result.icir > 0:
            # 经验公式: fitness ≈ 0.5 + icir * 0.3 (有上限)
            predicted = min(0.5 + result.icir * 0.3, 3.0)
            result.predicted_fitness = round(predicted, 4)

        # ── Confidence 计算 ──
        # 置信度基于指标一致性
        confidence_factors = []

        # Sharpe 与 IC 的一致性
        if 0 < sharpe <= 2.0:
            confidence_factors.append(0.8)
        elif sharpe > 2.0:
            confidence_factors.append(0.6)  # 可能过拟合

        # Turnover 合理性
        if 0.05 <= turnover <= 0.50:
            confidence_factors.append(0.9)
        elif turnover > 0.50 or turnover < 0.02:
            confidence_factors.append(0.6)

        # Fitness 与预测的一致性
        if fitness > 0 and result.predicted_fitness > 0:
            diff_ratio = abs(fitness - result.predicted_fitness) / max(fitness, result.predicted_fitness)
            if diff_ratio < 0.3:
                confidence_factors.append(0.9)
            else:
                confidence_factors.append(0.6)

        if confidence_factors:
            result.confidence = round(sum(confidence_factors) / len(confidence_factors), 3)

        return result

    def compute_multi_faceted_reward(
        self,
        metrics: dict,
        expr: str = "",
    ) -> MultiFacetedReward:
        """计算多面奖励 (参考 Alpha SAGE Multi-Faceted Reward Function)

        R_total = w1*signal + w2*stability + w3*efficiency + w4*uniqueness - w5*simplicity

        权重配置:
          signal_quality: 0.30 (Sharpe-based component)
          stability: 0.25 (Fitness-based component)
          efficiency: 0.20 (Return-per-turnover |R|/TO)
          uniqueness: 0.15 (Decorrelation bonus)
          simplicity: 0.10 (Complexity penalty)

        Args:
            metrics: WQ 返回的基本指标字典
            expr: 因子表达式字符串 (用于复杂度分析，可选)

        Returns:
            MultiFacetedReward 数据类实例
        """
        sharpe = float(metrics.get("sharpe", 0))
        fitness = float(metrics.get("fitness", 0))
        turnover = max(float(metrics.get("turnover", 0.01)), 0.001)  # 避免除零
        returns = float(metrics.get("returns", 0))

        reward = MultiFacetedReward()
        weights = MultiFacetedReward.WEIGHTS

        # ── Signal Quality (0-1) ──
        # 基于 Sharpe ratio 归一化
        # Sharpe >= 2.0 → 1.0, Sharpe <= 0 → 0.0
        if sharpe >= 2.0:
            reward.signal_quality = 1.0
        elif sharpe > 0:
            reward.signal_quality = min(sharpe / 2.0, 1.0)
        else:
            reward.signal_quality = 0.0

        # ── Stability (0-1) ──
        # 基于 Fitness 归一化
        # Fitness >= 2.0 → 1.0, Fitness <= 0 → 0.0
        if fitness >= 2.0:
            reward.stability = 1.0
        elif fitness > 0:
            reward.stability = min(fitness / 2.0, 1.0)
        else:
            reward.stability = 0.0

        # ── Efficiency (0-1) ──
        # Return-per-turnover: |returns| / turnover
        # 目标值: efficiency > 0.3 为高效因子
        if turnover > 0 and returns != 0:
            efficiency_value = abs(returns) / turnover
            # 归一化到 0-1 (假设效率 > 0.5 为满分)
            reward.efficiency = min(efficiency_value / 0.5, 1.0)
        elif turnover > 0 and sharpe > 0:
            # 从 Sharpe 推断效率
            reward.efficiency = min(sharpe / 2.0 * 0.8, 1.0)
        else:
            reward.efficiency = 0.0

        # ── Uniqueness (0-1) ──
        # 去相关奖励 (基于因子特征估算)
        # 这里使用启发式方法：从表达式特征推断
        uniqueness_score = self._estimate_uniqueness(expr, metrics)
        reward.uniqueness = uniqueness_score

        # ── Simplicity Penalty (0-1, 越大越复杂) ──
        # 复杂度惩罚 (基于 AST 深度或表达式长度)
        complexity_penalty = self._estimate_complexity_penalty(expr)
        reward.simplicity = complexity_penalty

        # ── Total Reward ──
        # 加权求和 (simplicity 是惩罚项)
        reward.total_reward = (
            weights["signal_quality"] * reward.signal_quality
            + weights["stability"] * reward.stability
            + weights["efficiency"] * reward.efficiency
            + weights["uniqueness"] * reward.uniqueness
            - weights["simplicity"] * reward.simplicity
        )
        reward.total_reward = max(0.0, min(1.0, reward.total_reward))

        return reward

    @staticmethod
    def _estimate_uniqueness(expr: str, metrics: dict) -> float:
        """估算因子的独特性/去相关性

        启发式规则:
          - 使用非价格字段 (earnings, sales 等) → 更独特
          - 使用罕见算子组合 → 更独特
          - 低换手率 + 高 Sharpe → 可能是独特信号

        Args:
            expr: 因子表达式
            metrics: 指标字典

        Returns:
            独特性分数 (0-1)
        """
        if not expr:
            return 0.5  # 默认中等

        expr_lower = expr.lower()
        uniqueness = 0.5  # 基础分

        # 非价格字段加分
        unique_fields = [
            "earnings",
            "sales",
            "revenue",
            "cap",
            "analyst",
            "estimate",
            "guidance",
            "institutional",
            "short_interest",
            "options",
            "implied",
        ]
        for field in unique_fields:
            if field in expr_lower:
                uniqueness += 0.08
                break

        # 罕见算子加分
        rare_operators = [
            "ts_regression",
            "ts_av_diff",
            "ts_corr",
            "ts_skewness",
            "ts_kurtosis",
            "ts_product",
            "signed_power",
            "tanh",
            "sigmoid",
        ]
        rare_count = sum(1 for op in rare_operators if op in expr_lower)
        uniqueness += min(rare_count * 0.05, 0.15)

        # 表达式长度适中加分 (太简单可能拥挤)
        if 50 < len(expr) < 200:
            uniqueness += 0.05

        # 低换手率 + 正收益加分
        turnover = metrics.get("turnover", 0.5)
        sharpe = metrics.get("sharpe", 0)
        if turnover < 0.30 and sharpe > 1.0:
            uniqueness += 0.08

        return min(uniqueness, 1.0)

    @staticmethod
    def _estimate_complexity_penalty(expr: str) -> float:
        """估算表达式复杂度 (用于惩罚过复杂的因子)

        规则:
          - 表达式长度越长 → 越复杂
          - 嵌套层级越深 → 越复杂
          - 算子数量越多 → 越复杂

        Args:
            expr: 因子表达式

        Returns:
            复杂度惩罚分数 (0-1, 1=最复杂)
        """
        if not expr:
            return 0.0

        penalty = 0.0

        # 长度惩罚 (>100 字符开始扣分)
        length = len(expr)
        if length > 300:
            penalty += 0.3
        elif length > 200:
            penalty += 0.2
        elif length > 150:
            penalty += 0.1

        # 嵌套深度惩罚 (通过括号匹配估算)
        max_nesting = 0
        current_nesting = 0
        for char in expr:
            if char == "(":
                current_nesting += 1
                max_nesting = max(max_nesting, current_nesting)
            elif char == ")":
                current_nesting -= 1

        if max_nesting > 6:
            penalty += 0.25
        elif max_nesting > 4:
            penalty += 0.15
        elif max_nesting > 3:
            penalty += 0.08

        # 算子数量惩罚
        import re

        operators = re.findall(r"ts_\w+|rank|group_neutralize|tanh|sigmoid|signed_power", _expr_lower := expr.lower())
        operator_count = len(set(operators))  # 去重后的算子数量
        if operator_count > 8:
            penalty += 0.2
        elif operator_count > 6:
            penalty += 0.12
        elif operator_count > 4:
            penalty += 0.05

        return min(penalty, 1.0)

    # ══════════════════════════════════════════════════════════════════════
    # 🆕 增强评估方法 (v1.0.0+)
    # ══════════════════════════════════════════════════════════════════════

    def multi_layer_evaluate(self, metrics: dict) -> dict:
        """三层评估决策 (RD-Agent 模式)

        Layer 1: 值评估 (Value Evaluation)
            - 检查基本指标是否有效 (sharpe>0, fitness>0, turnover合理)
            - 快速过滤明显异常的数据

        Layer 2: 代码/结构评估 (Code Evaluation)
            - 分析因子的结构特征 (复杂度、字段多样性、算子组合)
            - 识别潜在问题 (过拟合风险、换手率异常等)

        Layer 3: 最终决策 (Final Decision)
            - 综合 Layer 1+2 结果，给出 PASS/IMPROVE/REJECT 决策
            - 生成详细的改进建议

        Returns:
            {
                "layer1_passed": bool,
                "layer1_details": str,
                "layer2_score": float,  # 0-1
                "layer2_issues": list[str],
                "final_decision": "PASS" | "IMPROVE" | "REJECT",
                "confidence": float,  # 0-1
                "diagnosis": str,
            }
        """
        # 提取指标
        sharpe = float(metrics.get("sharpe", 0))
        fitness = float(metrics.get("fitness", 0))
        turnover = float(metrics.get("turnover", 0))
        drawdown = abs(float(metrics.get("drawdown", 0)))
        returns = float(metrics.get("returns", 0))

        # ── Layer 1: 值评估 ──
        layer1_issues = []
        layer1_passed = True

        if sharpe <= 0:
            layer1_issues.append("Sharpe <= 0: 无正收益")
            layer1_passed = False
        elif sharpe < BRAIN_THRESHOLDS["min_sharpe"]:
            layer1_issues.append(f"Sharpe ({sharpe:.2f}) < 最低要求 ({BRAIN_THRESHOLDS['min_sharpe']})")

        if fitness <= 0:
            layer1_issues.append("Fitness <= 0: 质量指标无效")
            layer1_passed = False
        elif fitness < BRAIN_THRESHOLDS["min_fitness"]:
            layer1_issues.append(f"Fitness ({fitness:.2f}) < 最低要求 ({BRAIN_THRESHOLDS['min_fitness']})")

        if turnover <= 0 or turnover > BRAIN_THRESHOLDS["platform_max_turnover"]:
            layer1_issues.append(f"Turnover ({turnover:.1%}) 超出平台限制")
            layer1_passed = False

        if drawdown > BRAIN_THRESHOLDS["max_drawdown"]:
            layer1_issues.append(f"Drawdown ({drawdown:.1%}) 超过安全阈值")

        if returns <= 0 and sharpe > 0:
            layer1_issues.append("警告: Sharpe>0 但 Returns<=0，可能数据异常")

        layer1_details = "; ".join(layer1_issues) if layer1_issues else "所有基本指标正常"

        # ── Layer 2: 结构/质量评估 ──
        layer2_issues = []
        layer2_score = 1.0

        # 过拟合风险检测
        if sharpe > 2.5 and drawdown > 0.20:
            layer2_issues.append("过拟合风险: Sharpe过高但回撤大")
            layer2_score -= 0.25

        if turnover > 0.60:
            layer2_issues.append("换手率过高: 可能过度交易")
            layer2_score -= 0.15

        if turnover < 0.02:
            layer2_issues.append("换手率过低: 信号可能过于平滑")
            layer2_score -= 0.10

        if fitness < 0.5 * sharpe:
            layer2_issues.append("Fitness偏低: 收益质量不佳")
            layer2_score -= 0.20

        if drawdown > 0.25:
            layer2_issues.append("高风险: 回撤过大")
            layer2_score -= 0.20

        # 稳定性评估
        if 0.8 <= sharpe <= 1.5 and 0.8 <= fitness <= 1.5:
            layer2_score += 0.05  # 稳健因子加分

        layer2_score = max(0.0, min(1.0, layer2_score))

        # ── Layer 3: 最终决策 (含策略 D 增强) ──
        # 🆕 计算 ICIR 和多面奖励
        icir_metrics = self.infer_icir_metrics(metrics)
        multi_reward = self.compute_multi_faceted_reward(metrics)

        if not layer1_passed:
            final_decision = "REJECT"
            confidence = 0.9
        elif layer2_score >= 0.85 and layer1_passed:
            final_decision = "PASS"
            confidence = min(1.0, 0.7 + layer2_score * 0.3)
        else:
            final_decision = "IMPROVE"
            confidence = 0.6 + layer2_score * 0.2

        # 生成诊断信息 (增强版)
        diagnosis_parts = []
        if sharpe >= 1.5:
            diagnosis_parts.append("HIGH_SHARPE")
        elif sharpe >= 1.0:
            diagnosis_parts.append("MODERATE_SHARPE")
        else:
            diagnosis_parts.append("LOW_SHARPE")

        if fitness >= 1.2:
            diagnosis_parts.append("HIGH_FITNESS")
        elif fitness >= 0.8:
            diagnosis_parts.append("MODERATE_FITNESS")
        else:
            diagnosis_parts.append("LOW_FITNESS")

        if turnover > 0.50:
            diagnosis_parts.append("HIGH_TURNOVER")
        elif turnover < 0.10:
            diagnosis_parts.append("LOW_TURNOVER")

        if drawdown > 0.20:
            diagnosis_parts.append("HIGH_DRAWDOWN")

        # 🆕 策略 D: 新增分类
        # HIGH_ICIR_LOW_FITNESS: ICIR 高但 Fitness 低
        if icir_metrics.is_high_icir and fitness < 1.0:
            diagnosis_parts.append("HIGH_ICIR_LOW_FITNESS")
            final_decision = "IMPROVE"  # 强制改进（特殊处理）

        # EFFICIENT_ALPHA: 高效因子
        if multi_reward.is_efficient_alpha:
            diagnosis_parts.append("EFFICIENT_ALPHA")

        diagnosis = "_".join(diagnosis_parts) if diagnosis_parts else "UNKNOWN"

        return {
            "layer1_passed": layer1_passed,
            "layer1_details": layer1_details,
            "layer2_score": round(layer2_score, 3),
            "layer2_issues": layer2_issues,
            "final_decision": final_decision,
            "confidence": round(confidence, 3),
            "diagnosis": diagnosis,
            # 🆕 策略 D 新增字段
            "icir_metrics": icir_metrics.to_dict(),
            "multi_faceted_reward": multi_reward.to_dict(),
            "is_high_icir_low_fitness": icir_metrics.is_low_fitness_high_icir,
            "is_efficient_alpha": multi_reward.is_efficient_alpha,
        }

    def infer_advanced_metrics(self, metrics: dict) -> AdvancedMetrics:
        """从 WQ 返回的基本指标推断高级指标

        由于 WQ 不直接返回 IC/RankIC 等，
        我们可以从 Sharpe 和 Fitness 进行合理推断：

        推断逻辑:
        - Sharpe ≈ IC * sqrt(252) / std(factor_returns)  [近似关系]
        - Fitness ≈ 综合质量分 (可能包含 ICIR 成分)
        - Win Rate ≈ 从 Sharpe 符号稳定性推断
        - Risk Level → 从 drawdown + turnover 综合判断
        """
        sharpe = float(metrics.get("sharpe", 0))
        fitness = float(metrics.get("fitness", 0))
        turnover = float(metrics.get("turnover", 0))
        drawdown = abs(float(metrics.get("drawdown", 0)))
        returns = float(metrics.get("returns", 0))

        advanced = AdvancedMetrics()

        # ── IC 推断 (基于 Sharpe 近似关系) ──
        # 假设: Sharpe ≈ IC * sqrt(252) / σ_factor_returns
        # 对于典型因子, σ_factor_returns ≈ 0.15-0.25
        # 因此 IC ≈ Sharpe * 0.20 / sqrt(252) ≈ Sharpe * 0.0126
        if sharpe > 0:
            estimated_ic = sharpe * 0.0126
            advanced.ic = round(estimated_ic, 4)
            # Rank IC 通常略高于 IC (更稳健)
            advanced.rank_ic = round(estimated_ic * 1.1, 4)

        # ── IR 推断 (Information Ratio) ──
        # IR = mean_IC / std_IC, 通常在 0.5-2.0 之间
        # 从 Fitness 推断: Fitness 高通常意味着 IR 较高
        if fitness > 0:
            # 假设 IR 与 Fitness 正相关
            estimated_ir = min(2.0, fitness * 0.8)
            advanced.ir = round(estimated_ir, 3)

        # ── ICIR 推断 ──
        # ICIR = IC * sqrt(n) / std_IC, 其中 n 为观察期数
        # 假设 n=60 (约5年月度数据), std_IC ≈ IC/IR
        if advanced.ic and advanced.ir and advanced.ir > 0:
            import math

            n_periods = 60  # 假设 5 年月度数据
            std_ic_estimated = abs(advanced.ic) / advanced.ir
            if std_ic_estimated > 0:
                icir_value = abs(advanced.ic) * math.sqrt(n_periods) / std_ic_estimated
                advanced.icir = round(icir_value, 3)
                advanced.rank_icir = round(icir_value * 1.05, 3)

        # ── Win Rate 推断 ──
        # 从 Sharpe 推断月度胜率
        # 经验公式: win_rate ≈ norm.cdf(sharpe / sqrt(12)) 的近似
        if sharpe > 0:
            import math

            # 使用近似: P(X>0) 当 X~N(μ,σ), μ=sharpe/sqrt(12), σ=1
            z_score = sharpe / math.sqrt(12)
            # 近似标准正态 CDF
            win_rate_approx = 1.0 / (1.0 + math.exp(-0.07056 * z_score**3 - 1.5976 * z_score))
            advanced.win_rate = round(win_rate_approx, 3)

        # ── Stability 推断 ──
        # 从 Fitness 和 Turnover 推断稳定性
        # 高 Fitness + 中等 Turnover = 高稳定性
        if fitness > 0 and turnover > 0:
            stability_score = fitness * 0.4
            if 0.10 <= turnover <= 0.40:
                stability_score += 0.3
            elif turnover > 0.50:
                stability_score -= 0.2
            advanced.stability = round(min(1.0, max(0.0, stability_score)), 3)

        # ── Risk Level 判定 ──
        risk_factors = []
        if drawdown > 0.30:
            risk_factors.append(("CRITICAL", 3))
        elif drawdown > 0.20:
            risk_factors.append(("HIGH", 2))
        elif drawdown > 0.15:
            risk_factors.append(("MEDIUM", 1))

        if turnover > 0.70:
            risk_factors.append(("CRITICAL", 2))
        elif turnover > 0.50:
            risk_factors.append(("HIGH", 1))

        if sharpe < 0:
            risk_factors.append(("CRITICAL", 3))
        elif sharpe < 0.5:
            risk_factors.append(("HIGH", 1))

        if risk_factors:
            risk_factors.sort(key=lambda x: x[1], reverse=True)
            advanced.risk_level = risk_factors[0][0]
        else:
            if sharpe >= 1.5 and fitness >= 1.2 and drawdown < 0.10:
                advanced.risk_level = "LOW"

        # ── Improvement Potential 计算 ──
        # 基于各维度距离理想值的差距
        potential = 0.0

        # Sharpe 潜力 (目标: 2.0)
        if sharpe < 2.0:
            potential += (2.0 - sharpe) / 2.0 * 0.35

        # Fitness 潜力 (目标: 2.0)
        if fitness < 2.0:
            potential += (2.0 - fitness) / 2.0 * 0.25

        # Turnover 优化潜力 (目标: 0.15-0.30)
        if turnover < 0.10 or turnover > 0.40:
            potential += 0.15

        # Drawdown 优化潜力 (目标: <0.10)
        if drawdown > 0.10:
            potential += min(0.25, (drawdown - 0.10) * 1.5)

        advanced.improvement_potential = round(min(1.0, potential), 3)

        # ── Quantile Returns 模拟 ──
        # 基于 Sharpe 和 Returns 模拟五档收益分布
        if returns != 0 and sharpe > 0:
            avg_return = returns * 252  # 年化
            spread_estimate = abs(avg_return) * 0.4
            advanced.quantile_returns = {
                "Q1": f"{-(spread_estimate * 0.8):.2%}",
                "Q2": f"{-(spread_estimate * 0.3):.2%}",
                "Q3": f"{(spread_estimate * 0.1):.2%}",
                "Q4": f"{(spread_estimate * 0.5):.2%}",
                "Q5": f"{(spread_estimate * 1.0):.2%}",
                "spread": f"{spread_estimate * 1.8:.2%}",
            }

        # ── Overall Diagnosis ──
        profile = self.classify_factor_profile(metrics)
        advanced.overall_diagnosis = profile

        return advanced

    def classify_factor_profile(self, metrics: dict) -> str:
        """因子画像分类 (用于改进策略选择)

        分类类别:
        - "PERFECT": Sharpe≥1.5, Fitness≥1.2, Turnover<30% → 无需改进
        - "HIGH_SHARPE_LOW_FITNESS": Sharpe≥1.25, Fitness<1.0 → 专注降低换手率/过拟合
        - "NEAR_PASS": 0.8≤Sharpe<1.25, Fitness≥0.7 → EA/Prompt 注入优先
        - "LOW_SHARPE_HIGH_TURNOVER": Sharpe<0.8, Turnover>50% → 降低换手率
        - "NOISE": Sharpe<0, Fitness<0.5 → 丢弃或完全重写
        - "OVERFIT": Sharpe高但Drawdown>20% → 正则化/简化表达式
        - "STUCK": 连续多次改进无提升 → 触发 EA 全局搜索
        """
        sharpe = float(metrics.get("sharpe", 0))
        fitness = float(metrics.get("fitness", 0))
        turnover = float(metrics.get("turnover", 0))
        drawdown = abs(float(metrics.get("drawdown", 0)))

        # NOISE: 最差情况
        if sharpe < 0 and fitness < 0.5:
            return "NOISE"

        # PERFECT: 最优情况
        if sharpe >= 1.5 and fitness >= 1.2 and turnover < 0.30 and drawdown < 0.15:
            return "PERFECT"

        # OVERFIT: 高Sharpe但高风险
        if sharpe >= 1.25 and drawdown > 0.20:
            return "OVERFIT"

        # HIGH_SHARPE_LOW_FITNESS: Sharpe不错但质量低
        if sharpe >= 1.25 and fitness < 1.0:
            return "HIGH_SHARPE_LOW_FITNESS"

        # LOW_SHARPE_HIGH_TURNOVER: 低收益高换手
        if sharpe < 0.8 and turnover > 0.50:
            return "LOW_SHARPE_HIGH_TURNOVER"

        # NEAR_PASS: 接近及格线
        if 0.8 <= sharpe < 1.25 and fitness >= 0.7:
            return "NEAR_PASS"

        # STUCK: 需要外部信息判断（这里用默认逻辑）
        if 0.5 <= sharpe < 0.8 and 0.5 <= fitness < 0.8:
            return "STUCK"

        # 默认分类
        if sharpe >= 1.0:
            return "MODERATE_GOOD"
        elif sharpe >= 0.5:
            return "MODERATE_WEAK"
        else:
            return "WEAK_NEEDS_WORK"


# ── 便捷函数 ──────────────────────────────────────────────────────────────


def quick_score(sharpe: float, fitness: float, turnover: float, **kwargs) -> ScoreReport:
    """快速评分函数（简化接口）"""
    scorer = OfficialScoringAdapter()
    metrics = {
        "sharpe": sharpe,
        "fitness": fitness,
        "turnover": turnover,
        **kwargs,
    }
    return scorer.compute_score(metrics)


def evaluate_alpha_quality(
    expression: str,
    metrics: dict,
    validator=None,
) -> tuple:
    """完整的 Alpha 质量评估流程

    Args:
        expression: FASTEXPR 表达式
        metrics: WQ 模拟结果指标
        validator: WQExpressionValidator 实例（可选）

    Returns:
        (validation_result, score_report) 元组
    """
    from openalpha_brain.validation.wq_expression_validator import WQExpressionValidator

    # 1. 表达式验证
    if validator is None:
        validator = WQExpressionValidator()
    validation_result = validator.validate_full(expression)

    # 2. 评分计算
    scorer = OfficialScoringAdapter()
    score_report = scorer.compute_score(metrics)

    return validation_result, score_report
