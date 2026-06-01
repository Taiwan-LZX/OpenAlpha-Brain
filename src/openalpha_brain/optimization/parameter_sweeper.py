"""ParameterSweeper — 自动数值参数扫描优化器。

Inspired by worldquant-miner/alpha_expression_miner.py (zhutoutoutousan, 140 commits)

来自 worldquant-miner 工程实践 (140 commits, 2026-02):
自动识别表达式中的数值字面量（窗口、系数、阈值），
在合理范围内生成变体矩阵，批量回测选优。

工作流程:
1. parse_numeric_literals(expr) → 提取所有数值参数及上下文
2. generate_variants(params_config, range_pct, step) → 生成变体表达式列表
3. batch_evaluate(variants, brain_client) → 并发提交 BRAIN 回测（受 Semaphore(3) 限制）
4. select_best(results, metric='sharpe') → 选最优替换原表达式

参数类型分类:
  - window: 时间序列窗口大小 (ts_mean, ts_decay_linear 等)
  - coefficient: 运算系数/权重 (signed_power, scale 等)
  - threshold: 阈值参数 (trade_when, hump 等)
  - epsilon: 极小值保护参数
  - decay_exponent: 衰减指数 (ts_decay_exp_window 的指数部分)

使用示例:
    sweeper = ParameterSweeper(default_range_pct=10, default_step=1, max_variants=200)
    params = sweeper.parse_numeric_literals("ts_decay_linear(ts_rank(volume, 20), 10)")
    variants = sweeper.generate_variants("ts_decay_linear(ts_rank(volume, 20), 10)")
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

EPSILON = 1e-8


class ParamType(Enum):
    """参数类型枚举."""

    WINDOW = "window"
    COEFFICIENT = "coefficient"
    THRESHOLD = "threshold"
    EPSILON = "epsilon"
    DECAY_EXPONENT = "decay_exponent"


@dataclass
class NumericLiteral:
    """提取的数值字面量信息."""

    value: float
    context: str
    position: int
    param_type: ParamType
    original_str: str


@dataclass
class VariantResult:
    """单个变体的回测结果."""

    expression: str
    metric_value: float
    metrics: dict = field(default_factory=dict)


@dataclass
class SweepResult:
    """参数扫描完整结果."""

    original_expression: str
    best_expression: str | None
    best_metric_value: float
    metric_name: str
    total_variants_tested: int
    improvement_pct: float
    results: list[VariantResult] = field(default_factory=list)
    params_scanned: list[NumericLiteral] = field(default_factory=list)


class ParameterSweeper:
    """自动数值参数扫描优化器.

    Design Principles:
      1. 智能识别数值参数类型（基于上下文和位置）
      2. 为不同类型参数生成合理的搜索范围
      3. 控制变体数量防止组合爆炸（max_variants 硬上限）
      4. 支持异步并发回测评估
      5. 所有数值操作加 epsilon 保护
    """

    PARAM_TYPE_CONFIG = {
        ParamType.WINDOW: {
            "range_pct": 0.50,
            "step": 1,
            "min_val": 2,
            "max_val": 252,
            "description": "lookback window",
        },
        ParamType.COEFFICIENT: {
            "range_pct": 1.00,
            "step": 0.01,
            "min_val": -5.0,
            "max_val": 5.0,
            "description": "coefficient/weight",
        },
        ParamType.THRESHOLD: {
            "range_pct": 2.00,
            "step": 0.001,
            "min_val": 0.001,
            "max_val": 1.0,
            "description": "threshold parameter",
        },
        ParamType.EPSILON: {
            "range_pct": 2.00,
            "step": 0.001,
            "min_val": 0.001,
            "max_val": 1.0,
            "description": "epsilon protection",
        },
        ParamType.DECAY_EXPONENT: {
            "range_pct": 0.50,
            "step": 0.1,
            "min_val": 0.1,
            "max_val": 3.0,
            "description": "decay exponent",
        },
    }

    WINDOW_OPERATORS = [
        "ts_mean",
        "ts_std_dev",
        "ts_sum",
        "ts_min",
        "ts_max",
        "ts_arg_max",
        "ts_arg_min",
        "ts_rank",
        "ts_delta",
        "ts_regression",
        "ts_corr",
        "ts_covariance",
        "ts_av_diff",
        "ts_zscore",
        "ts_skewness",
        "ts_kurtosis",
        "ts_product",
        "ts_decay_linear",
        "ts_decay_exp_window",
    ]

    COEFFICIENT_PATTERNS = [
        r"signed_power\s*\(\s*[^,]+\s*,\s*([\d.]+)\s*\)",
        r"scale\s*\(\s*[^,]+\s*,\s*([\d.]+)\s*\)",
    ]

    THRESHOLD_PATTERNS = [
        r"trade_when\s*\([^,]+,\s*([\d.]+)\s*,\s*[\d.]+\s*\)",
        r"hump\s*\([^,]+,\s*([\d.]+)\s*\)",
        r"truncate\s*\([^,]+,\s*([\d.-]+)\s*\)",
    ]

    def __init__(
        self,
        default_range_pct: float = 10,
        default_step: float = 1,
        max_variants: int = 200,
    ):
        """初始化 ParameterSweeper.

        Args:
            default_range_pct: 默认搜索范围百分比 (10 表示 ±10%)
            default_step: 默认步长
            max_variants: 最大变体数量硬上限
        """
        self.default_range_pct = default_range_pct / 100.0 if default_range_pct > 1 else default_range_pct
        self.default_step = default_step
        self.max_variants = max(max_variants, 1)
        self._semaphore = asyncio.Semaphore(3)
        self._evaluate_fn: Any = None

    def set_evaluate_fn(self, fn: Any) -> None:
        """设置评估函数（用于批量回测）.

        Args:
            fn: 异步评估函数，签名为 async fn(expression: str) -> dict
                返回 dict 包含 sharpe, fitness, turnover 等指标
        """
        self._evaluate_fn = fn

    def parse_numeric_literals(self, expression: str) -> list[NumericLiteral]:
        """提取表达式中的所有数值字面量及上下文信息.

        使用正则表达式匹配数值参数，并根据周围的操作符和上下文
        判断参数类型（window/coefficient/threshold/epsilon/decay_exponent）。

        Args:
            expression: Alpha 表达式字符串

        Returns:
            NumericLiteral 对象列表，每个包含 value, context, position, param_type
        """
        literals = []

        pattern = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"

        for match in re.finditer(pattern, expression):
            value_str = match.group(1)
            try:
                value = float(value_str)
            except (ValueError, OSError):
                continue

            if value == 0 and "." not in value_str:
                continue

            position = match.start()
            start = max(0, position - 30)
            end = min(len(expression), position + len(value_str) + 30)
            context = expression[start:end]

            param_type = self._classify_param_type(expression, position, value, context)

            literals.append(
                NumericLiteral(
                    value=value,
                    context=context,
                    position=position,
                    param_type=param_type,
                    original_str=value_str,
                )
            )

        logger.debug(
            "[PARAM-SWEEP] Parsed %d numeric literals from expression (%d chars)",
            len(literals),
            len(expression),
        )
        return literals

    def _classify_param_type(
        self,
        expression: str,
        position: int,
        value: float,
        context: str,
    ) -> ParamType:
        """根据上下文判断参数类型.

        Args:
            expression: 完整表达式
            position: 数值在表达式中的位置
            value: 数值本身
            context: 周围30字符的上下文

        Returns:
            ParamType 枚举值
        """
        preceding_text = expression[:position]

        for op in self.WINDOW_OPERATORS:
            pattern = rf"{op}\s*\([^)]*$"
            if re.search(pattern, preceding_text):
                if op == "ts_decay_exp_window" and "," in preceding_text.rsplit("(", 1)[-1]:
                    return ParamType.DECAY_EXPONENT
                return ParamType.WINDOW

        for coeff_pattern in self.COEFFICIENT_PATTERNS:
            match = re.search(coeff_pattern, context)
            if match and abs(float(match.group(1)) - value) < EPSILON:
                return ParamType.COEFFICIENT

        for thresh_pattern in self.THRESHOLD_PATTERNS:
            match = re.search(thresh_pattern, context)
            if match and abs(float(match.group(1)) - value) < EPSILON:
                return ParamType.THRESHOLD

        if 0 < value < 0.01:
            return ParamType.EPSILON

        if 2 <= value <= 252 and value == int(value):
            return ParamType.WINDOW

        if abs(value) <= 5.0:
            return ParamType.COEFFICIENT

        return ParamType.THRESHOLD

    def generate_variants(
        self,
        expression: str,
        params_config: dict | None = None,
    ) -> list[str]:
        """生成参数变体表达式列表.

        根据解析出的数值参数类型，为每个参数生成合理的变体范围，
        然后通过笛卡尔积生成所有组合（受 max_variants 上限控制）。

        参数类型搜索范围配置:
          - window type (lookback): range ±50%, step 1, min=2, max=252
          - coefficient: range ±100%, step 0.01, min=-5, max=5
          - threshold/epsilon: range ±200%, step 0.001, min=0.001, max=1.0
          - decay exponent: range ±50%, step 0.1, min=0.1, max=3.0

        Args:
            expression: 原始 Alpha 表达式
            params_config: 可选的参数级别配置覆盖，格式为:
                {position: {"range_pct": float, "step": float, "min_val": float, "max_val": float}}

        Returns:
            变体表达式列表（不包含原始表达式）
        """
        literals = self.parse_numeric_literals(expression)

        if not literals:
            logger.info("[PARAM-SWEEP] No numeric literals found in expression")
            return []

        param_variations = []
        for lit in literals:
            config = self.PARAM_TYPE_CONFIG[lit.param_type].copy()

            if params_config and lit.position in params_config:
                config.update(params_config[lit.position])

            range_pct = config["range_pct"]
            step = config["step"]
            min_val = config["min_val"]
            max_val = config["max_val"]

            variations = self._generate_single_param_variations(lit.value, range_pct, step, min_val, max_val)

            if variations:
                param_variations.append(
                    {
                        "literal": lit,
                        "variations": variations,
                    }
                )

        if not param_variations:
            return []

        variants = self._cartesian_generate(expression, param_variations)

        logger.info(
            "[PARAM-SWEEP] Generated %d variants (cap=%d) from %d parameters",
            len(variants),
            self.max_variants,
            len(param_variations),
        )
        return variants[: self.max_variants]

    def _generate_single_param_variations(
        self,
        value: float,
        range_pct: float,
        step: float,
        min_val: float,
        max_val: float,
    ) -> list[float]:
        """为单个参数生成变体值列表.

        Args:
            value: 原始参数值
            range_pct: 搜索范围百分比（如 0.5 表示 ±50%）
            step: 步长
            min_val: 允许的最小值
            max_val: 允许的最大值

        Returns:
            变体值列表（不包含原始值）
        """
        abs_range = max(abs(value) * range_pct, EPSILON)

        lower_bound = max(value - abs_range, min_val)
        upper_bound = min(value + abs_range, max_val)

        if upper_bound - lower_bound < step + EPSILON:
            return []

        variations = []
        current = lower_bound + step

        while current < upper_bound - EPSILON:
            if abs(current - value) > EPSILON:
                variations.append(current)
            current += step

        if len(variations) > 20:
            step_size = len(variations) // 10
            variations = variations[::step_size][:10]

        return variations

    def _cartesian_generate(
        self,
        expression: str,
        param_variations: list[dict],
    ) -> list[str]:
        """通过受限笛卡尔积生成变体表达式.

        为避免组合爆炸，采用以下策略:
          1. 单参数变化优先（每次只改一个参数）
          2. 双参数组合次之
          3. 总数受 max_variants 限制

        Args:
            expression: 原始表达式
            param_variations: 参数变体列表

        Returns:
            变体表达式列表
        """
        variants = []
        seen = set()

        for pv in param_variations:
            lit = pv["literal"]
            for new_value in pv["variations"]:
                variant = self._replace_value_at_position(expression, lit.position, lit.original_str, new_value)

                if variant and variant not in seen:
                    seen.add(variant)
                    variants.append(variant)

                if len(variants) >= self.max_variants:
                    return variants

        if len(param_varizations := param_variations) >= 2 and len(variants) < self.max_variants:
            for i in range(min(len(param_varizations), 3)):
                for j in range(i + 1, min(len(param_varizations), 4)):
                    pv1 = param_variations[i]
                    pv2 = param_variations[j]

                    for v1 in pv1["variations"][:5]:
                        for v2 in pv2["variations"][:5]:
                            intermediate = self._replace_value_at_position(
                                expression,
                                pv1["literal"].position,
                                pv1["literal"].original_str,
                                v1,
                            )

                            if intermediate:
                                variant = self._replace_value_at_position(
                                    intermediate,
                                    pv2["literal"].position,
                                    pv2["literal"].original_str,
                                    v2,
                                )

                                if variant and variant not in seen:
                                    seen.add(variant)
                                    variants.append(variant)

                                if len(variants) >= self.max_variants:
                                    return variants

        return variants

    def _replace_value_at_position(
        self,
        expression: str,
        position: int,
        original_str: str,
        new_value: float,
    ) -> str | None:
        """替换表达式中指定位置的数值.

        Args:
            expression: 原始表达式
            position: 数值的起始位置
            original_str: 原始数值字符串
            new_value: 新的数值

        Returns:
            替换后的表达式，如果失败返回 None
        """
        try:
            if new_value == int(new_value):
                new_str = str(int(new_value))
            else:
                new_str = f"{new_value:.6g}"

            new_expr = expression[:position] + new_str + expression[position + len(original_str) :]

            if new_expr == expression:
                return None

            return new_expr
        except (OSError, ValueError, RuntimeError):
            return None

    async def batch_evaluate(self, variants: list[str]) -> list[VariantResult]:
        """并发批量评估变体表达式.

        使用 Semaphore(3) 限制并发数，避免过载 BRAIN API。

        Args:
            variants: 变体表达式列表

        Returns:
            VariantResult 列表，包含每个变体的指标
        """
        if not variants:
            return []

        if self._evaluate_fn is None:
            logger.warning("[PARAM-SWEEP] No evaluate_fn set, cannot batch evaluate")
            return []

        results = []
        tasks = [self._evaluate_single_variant(v) for v in variants]

        completed_results = await asyncio.gather(*tasks, return_exceptions=True)

        for variant, result in zip(variants, completed_results):
            if isinstance(result, Exception):
                logger.debug("[PARAM-SWEEP] Evaluation failed for variant: %s", result)
                continue

            if result is not None:
                results.append(result)

        logger.info(
            "[PARAM-SWEEP] Batch evaluation complete: %d/%d succeeded",
            len(results),
            len(variants),
        )
        return results

    async def _evaluate_single_variant(self, variant: str) -> VariantResult | None:
        """评估单个变体（带信号量限制）.

        Args:
            variant: 变体表达式

        Returns:
            VariantResult 或 None（如果评估失败）
        """
        async with self._semaphore:
            try:
                result_dict = await self._evaluate_fn(variant)

                if not isinstance(result_dict, dict):
                    return None

                sharpe = float(result_dict.get("sharpe", 0) or 0)
                fitness = float(result_dict.get("fitness", 0) or 0)
                turnover = result_dict.get("turnover")

                return VariantResult(
                    expression=variant,
                    metric_value=sharpe,
                    metrics={
                        "sharpe": sharpe,
                        "fitness": fitness,
                        "turnover": turnover,
                    },
                )
            except Exception as e:
                logger.debug("[PARAM-SWEEP] Single evaluation error: %s", e)
                return None

    def select_best(
        self,
        results: list[VariantResult],
        metric: str = "sharpe",
    ) -> tuple[str, dict] | None:
        """从结果中选择最优变体.

        Args:
            results: VariantResult 列表
            metric: 评价指标名称 ('sharpe', 'fitness')

        Returns:
            (best_expression, best_metrics) 元组，如果没有有效结果则返回 None
        """
        if not results:
            return None

        valid_results = [r for r in results if r.metric_value > EPSILON]

        if not valid_results:
            logger.info("[PARAM-SWEEP] No valid results with positive %s", metric)
            return None

        best = max(valid_results, key=lambda r: r.metric_value)

        logger.info(
            "[PARAM-SWEEP] Best variant selected: %s=%.4f (%d candidates)",
            metric,
            best.metric_value,
            len(valid_results),
        )
        return (best.expression, best.metrics)

    async def sweep(
        self,
        expression: str,
        current_sharpe: float = 0.0,
        brain_client: Any = None,
        metric: str = "sharpe",
    ) -> SweepResult:
        """执行完整的参数扫描流程.

        这是主要入口方法，整合了:
          1. 解析数值参数
          2. 生成变体
          3. 批量评估（如果提供了 brain_client）
          4. 选择最优

        Args:
            expression: 原始 Alpha 表达式
            current_sharpe: 当前 Sharpe ratio（用于计算改进幅度）
            brain_client: 可选的 BRAIN 客户端（用于实际回测）
            metric: 评价指标

        Returns:
            SweepResult 包含完整的扫描结果
        """
        t0 = time.perf_counter()

        params = self.parse_numeric_literals(expression)
        variants = self.generate_variants(expression)

        if not variants:
            return SweepResult(
                original_expression=expression,
                best_expression=None,
                best_metric_value=current_sharpe,
                metric_name=metric,
                total_variants_tested=0,
                improvement_pct=0.0,
                params_scanned=params,
            )

        results = []

        if brain_client is not None and hasattr(brain_client, "submit"):
            self.set_evaluate_fn(brain_client.submit)
            results = await self.batch_evaluate(variants)

        best_expr, best_metrics = self.select_best(results, metric)
        best_value = best_metrics.get(metric, 0) if best_metrics else 0.0

        improvement = 0.0
        if current_sharpe > EPSILON and best_value > EPSILON:
            improvement = (best_value - current_sharpe) / max(abs(current_sharpe), EPSILON)

        ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "[PARAM-SWEEP] Sweep complete: %d params, %d variants, %dms, best_%s=%.4f (%+.1f%%)",
            len(params),
            len(variants),
            int(ms),
            metric,
            best_value,
            improvement * 100,
        )

        return SweepResult(
            original_expression=expression,
            best_expression=best_expr,
            best_metric_value=best_value,
            metric_name=metric,
            total_variants_tested=len(results),
            improvement_pct=improvement,
            results=results,
            params_scanned=params,
        )


def get_parameter_sweeper(
    default_range_pct: float = 10,
    default_step: float = 1,
    max_variants: int = 200,
) -> ParameterSweeper:
    """工厂函数用于创建 ParameterSweeper 实例.

    Args:
        default_range_pct: 默认搜索范围百分比
        default_step: 默认步长
        max_variants: 最大变体数量

    Returns:
        配置好的 ParameterSweeper 实例
    """
    return ParameterSweeper(
        default_range_pct=default_range_pct,
        default_step=default_step,
        max_variants=max_variants,
    )
