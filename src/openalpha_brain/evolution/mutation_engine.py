"""Brain-Aware Mutation Engine for WQ BRAIN Platform.

Adapted from QuantGPT MutationEngine with WQ-specific diagnostics.
Provides structured failure diagnosis and targeted mutation strategies
to guide LLM-based factor improvement in the feedback loop.

Key Features:
  - Composite score calculation from WQ metrics (Sharpe, Fitness, Turnover)
  - Pattern-based failure mode detection (IC direction, nesting depth, etc.)
  - Strategy-specific prompt generation for each mutation type
  - Operator replacement suggestions with domain knowledge
  - No pandas dependency (pure string operations)
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MutationStrategy(Enum):
    """Mutation strategy types for factor improvement."""

    MUTATE_WINDOW = "mutate_window"
    MUTATE_OPERATOR = "mutate_operator"
    MUTATE_NORMALIZATION = "mutate_normalization"
    MUTATE_SIGNAL_TYPE = "mutate_signal_type"
    MUTATE_NONLINEAR = "mutate_nonlinear"
    MUTATE_INTERACTION = "mutate_interaction"
    SIMPLIFY = "simplify"
    REGENERATE_FULL = "regenerate_full"


@dataclass
class Diagnosis:
    """Diagnosis result with recommended strategy."""

    strategy: MutationStrategy
    reason: str
    details: dict


_OPERATOR_REPLACEMENTS = {
    "ts_mean": ["ts_decay_linear", "ts_sum", "ts_median"],
    "ts_std_dev": ["ts_mean", "ts_rank", "ts_mad"],
    "ts_delta": ["ts_av_diff", "ts_regression"],
    "ts_corr": ["ts_cov", "ts_rank"],
    "rank": ["zscore", "scale", "group_rank"],
    "decay_linear": ["ts_mean", "ts_sum"],
}


class BrainAwareMutationEngine:
    """Adapted mutation engine for WQ BRAIN platform.

    Diagnoses failure modes from WQ backtest metrics and selects targeted
    mutation strategies to guide LLM-based factor improvement.

    Usage:
        engine = BrainAwareMutationEngine()
        diagnosis = engine.diagnose(expression, wq_metrics, wq_checks)
        prompt = engine.generate_mutation_prompt(diagnosis, expression)
    """

    def __init__(self):
        self._operator_replacements = _OPERATOR_REPLACEMENTS
        self._nonlinear_ops = {"tanh", "sigmoid", "sign_power", "log", "sqrt"}
        self._normalization_ops = {"rank", "zscore", "scale", "group_zscore", "group_rank"}

    def diagnose(
        self,
        expression: str,
        wq_metrics: dict,
        wq_checks: list,
    ) -> Diagnosis:
        """Diagnose failure mode from WQ metrics and checks.

        Args:
            expression: Original factor expression
            wq_metrics: Dict with sharpe, fitness, turnover, returns, drawdown
            wq_checks: List of {name, value, limit, result} dicts

        Returns:
            Diagnosis with strategy, reason, and details
        """
        score = self.compute_score(wq_metrics)
        ic_mean = wq_metrics.get("ic_mean", 0)
        ic_ir = wq_metrics.get("ic_ir", 0)
        nesting = self.count_nesting(expression)
        has_norm = self.has_normalization(expression)
        has_nonlinear = self.has_nonlinear(expression)

        logger.debug(
            "[MUTATION-ENGINE] Diagnosing | score=%.1f ic=%.4f ir=%.2f nesting=%d norm=%s nonlinear=%s",
            score,
            ic_mean,
            ic_ir,
            nesting,
            has_norm,
            has_nonlinear,
        )

        if score < 20:
            return Diagnosis(
                strategy=MutationStrategy.REGENERATE_FULL,
                reason=f"极低评分({score:.1f}/100), 需要完全重写",
                details={"score": score, "ic_mean": ic_mean},
            )

        if abs(ic_mean) < 0.005:
            return Diagnosis(
                strategy=MutationStrategy.MUTATE_OPERATOR,
                reason=f"IC接近零({ic_mean:.4f}), 当前算子无预测能力",
                details={
                    "ic_mean": ic_mean,
                    "suggested_replacements": self._suggest_replacements(expression),
                },
            )

        if ic_mean < -0.01:
            return Diagnosis(
                strategy=MutationStrategy.MUTATE_SIGNAL_TYPE,
                reason=f"IC为负({ic_mean:.4f}), 因子方向反转",
                details={"ic_mean": ic_mean},
            )

        if nesting > 8:
            return Diagnosis(
                strategy=MutationStrategy.SIMPLIFY,
                reason=f"嵌套层数过深({nesting}层), 需适当简化",
                details={"nesting_depth": nesting},
            )

        if ic_ir < 0.5 and not has_norm:
            return Diagnosis(
                strategy=MutationStrategy.MUTATE_NORMALIZATION,
                reason=f"IR较低({ic_ir:.2f})且无标准化, 建议添加rank/zscore",
                details={"ic_ir": ic_ir, "has_normalization": has_norm},
            )

        if 20 <= score < 50 and not has_nonlinear:
            return Diagnosis(
                strategy=MutationStrategy.MUTATE_NONLINEAR,
                reason=f"评分中等({score:.1f})且无非线性变换, 建议引入tanh/power等",
                details={"score": score, "has_nonlinear": False},
            )

        if self._is_single_signal(expression):
            return Diagnosis(
                strategy=MutationStrategy.MUTATE_INTERACTION,
                reason="单信号因子, 建议组合多个信号源增强预测能力",
                details={"signal_count": 1},
            )

        return Diagnosis(
            strategy=MutationStrategy.MUTATE_WINDOW,
            reason="默认策略: 调整时序窗口参数以优化IC/IR",
            details={
                "ic_mean": ic_mean,
                "ic_ir": ic_ir,
                "current_windows": self._extract_windows(expression),
            },
        )

    def generate_mutation_prompt(
        self,
        diagnosis: Diagnosis,
        expression: str,
        inspiration_exprs: list[str] | None = None,
    ) -> str:
        """Generate targeted LLM improvement prompt based on diagnosis.

        Args:
            diagnosis: Diagnosis from diagnose() method
            expression: Original factor expression
            inspiration_exprs: Optional list of example expressions for reference

        Returns:
            Formatted prompt string for LLM
        """
        strategy = diagnosis.strategy

        parts = [
            "你是一个量化因子表达式优化专家。基于诊断结果，使用定向突变策略改进因子。",
            "",
            "## 输出格式要求（必须严格遵守）",
            "只返回一个因子表达式，不要任何解释、分析或推理过程。",
            "不要使用 markdown 代码块、反引号或引号包裹。",
            "你的回复必须是恰好一行可执行的因子表达式。",
            "",
            "## 复杂度限制",
            "- 函数嵌套层数不能超过 10 层",
            "- 表达式总长度不能超过 500 个字符",
            "",
            "## 多样性要求",
            "- 新表达式必须与当前表达式结构不同",
            "- 禁止仅修改常数或窗口参数的微小变化",
            "- 鼓励使用非线性变换（tanh, sigmoid, power）",
            "- 鼓励组合多个信号源（量价交互、动量+波动等）",
            "",
            "## 当前因子",
            f"{expression}",
            "",
            "## 诊断结果",
            f"策略: {strategy.value}",
            f"原因: {diagnosis.reason}",
            "",
        ]

        if strategy == MutationStrategy.MUTATE_WINDOW:
            windows = diagnosis.details.get("current_windows", [])
            parts.extend(
                [
                    "## 突变指令: 调整时序窗口",
                    f"当前窗口参数: {windows}",
                    "请尝试不同的窗口长度（5/10/20/40/60），保留核心算子结构。",
                    "建议:",
                    "- 短期动量: ts_delta(close, 5) → ts_delta(close, 10)",
                    "- 中期趋势: ts_mean(volume, 20) → ts_mean(volume, 40)",
                    "- 波动率: ts_std_dev(close, 20) → ts_std_dev(close, 60)",
                ]
            )

        elif strategy == MutationStrategy.MUTATE_OPERATOR:
            replacements = diagnosis.details.get("suggested_replacements", {})
            parts.extend(
                [
                    "## 突变指令: 替换核心算子",
                    f"建议替换方案: {replacements}",
                    "当前算子无预测能力，请替换为其他类型的时序/截面算子。",
                    "示例:",
                    "- ts_mean → ts_decay_linear (加入时间衰减)",
                    "- rank → zscore (改用标准化)",
                    "- ts_delta → ts_av_diff (平均差分替代简单差分)",
                ]
            )

        elif strategy == MutationStrategy.MUTATE_NORMALIZATION:
            parts.extend(
                [
                    "## 突变指令: 添加标准化",
                    "请在最外层添加 rank() 或 zscore()，或在关键子表达式上添加 scale() / tanh()。",
                    "示例:",
                    "- group_neutralize(rank(expr), industry)",
                    "- group_neutralize(zscore(expr), sector)",
                    "- tanh(ts_zscore(close, 20))",
                ]
            )

        elif strategy == MutationStrategy.MUTATE_SIGNAL_TYPE:
            parts.extend(
                [
                    "## 突变指令: 翻转因子方向",
                    "因子IC为负，请在表达式前添加 -1 * 或调整信号逻辑。",
                    "示例:",
                    "- 原始: rank(ts_delta(close, 20))",
                    "- 反转: -rank(ts_delta(close, 20))",
                    "- 或: rank(-ts_delta(close, 20))",
                    "注意: 如果使用了 group_neutralize，确保反转逻辑正确。",
                ]
            )

        elif strategy == MutationStrategy.MUTATE_NONLINEAR:
            parts.extend(
                [
                    "## 突变指令: 引入非线性变换",
                    "当前因子仅使用线性运算。请引入非线性变换增强表达能力：",
                    "- tanh(x): 压缩极端值，增强鲁棒性",
                    "- power(x, 0.5) 或 sign_power(x, 0.5): 弱化极端值影响",
                    "- sigmoid(x): S型映射，适合二值化信号",
                    "组合示例:",
                    "- rank(tanh(ts_delta(close, 20) / ts_std_dev(close, 20)))",
                    "- group_neutralize(signed_power(rank(ts_mean(volume, 10)), 0.5), industry)",
                ]
            )

        elif strategy == MutationStrategy.MUTATE_INTERACTION:
            parts.extend(
                [
                    "## 突变指令: 组合多信号源",
                    "当前因子仅使用单一信号。请组合多个信号源：",
                    "- 量价交互: rank(volume_signal) * rank(price_signal)",
                    "- 动量+波动: rank(momentum) * rank(-volatility)",
                    "- 条件组合: where(vol_condition, signal_a, signal_b)",
                    "- 加权组合: 0.6*rank(signal_a) + 0.4*rank(signal_b)",
                ]
            )

        elif strategy == MutationStrategy.SIMPLIFY:
            nesting_depth = diagnosis.details.get("nesting_depth", 0)
            parts.extend(
                [
                    "## 突变指令: 适当简化表达式",
                    f"当前嵌套深度: {nesting_depth} 层",
                    "请减少嵌套到6-8层以内，移除冗余变换，保留核心预测信号。",
                    "建议:",
                    "- 移除重复的 rank()/zscore() 包裹",
                    "- 合并相邻的 ts_* 算子",
                    "- 使用更简洁的表达方式表达相同逻辑",
                ]
            )

        elif strategy == MutationStrategy.REGENERATE_FULL:
            parts.extend(
                [
                    "## 突变指令: 完全重写",
                    "当前因子完全无效，请从零开始设计一个新的因子表达式。",
                    "建议尝试以下因子类别:",
                    "- 动量因子: ts_delta, ts_regression, ts_av_diff",
                    "- 反转因子: -ts_delta, ts_decay_linear (负相关)",
                    "- 波动率因子: ts_std_dev, ts_corr (与volume交互)",
                    "- 量价因子: volume * price_change 的各种组合",
                    "- 质量因子: earnings, sales, revenue 相关字段",
                ]
            )

        if inspiration_exprs:
            parts.extend(
                [
                    "",
                    "## 参考表达式（灵感来源）",
                    "以下是一些成功的因子表达式，可以参考其结构但不要直接复制:",
                ]
            )
            for i, expr in enumerate(inspiration_exprs[:3], 1):
                parts.append(f"{i}. {expr}")

        parts.extend(
            [
                "",
                "请生成改进后的因子表达式：",
            ]
        )

        return "\n".join(parts)

    def suggest_operator_replacement(self, expr: str, target_op: str) -> list[str]:
        """Suggest replacement candidates for a target operator.

        Args:
            expr: Factor expression (for context)
            target_op: Operator name to replace

        Returns:
            List of candidate replacement operators
        """
        return self._operator_replacements.get(target_op, [target_op])

    def count_nesting(self, expr: str) -> int:
        """Calculate expression nesting depth.

        Args:
            expr: Factor expression string

        Returns:
            Maximum nesting depth (integer)
        """
        max_depth = 0
        current = 0
        for ch in expr:
            if ch == "(":
                current += 1
                max_depth = max(max_depth, current)
            elif ch == ")":
                current -= 1
        return max_depth

    def has_normalization(self, expr: str) -> bool:
        """Check if expression contains normalization operators.

        Args:
            expr: Factor expression string

        Returns:
            True if normalization operator found
        """
        expr_lower = expr.lower()
        return any(op + "(" in expr_lower for op in self._normalization_ops)

    def has_nonlinear(self, expr: str) -> bool:
        """Check if expression contains nonlinear transforms.

        Args:
            expr: Factor expression string

        Returns:
            True if nonlinear operator found
        """
        expr_lower = expr.lower()
        return any(op + "(" in expr_lower for op in self._nonlinear_ops)

    def compute_score(self, wq_metrics: dict) -> float:
        """Compute composite score from WQ metrics (0-100).

        Scoring formula:
          - Sharpe contribution: 0-50 points (sharpe * 20, capped)
          - Fitness contribution: 0-30 points (fitness * 15, capped)
          - Turnover penalty/bonus: 0-20 points

        Args:
            wq_metrics: Dict with sharpe, fitness, turnover keys

        Returns:
            Composite score between 0 and 100
        """
        sharpe = wq_metrics.get("sharpe", 0)
        fitness = wq_metrics.get("fitness", 0)
        turnover = wq_metrics.get("turnover", 50)

        sharpe_score = max(0, min(50, sharpe * 20))
        fitness_score = max(0, min(30, fitness * 15))

        if 1 <= turnover <= 50:
            to_score = 20
        elif turnover <= 70:
            to_score = 10
        elif turnover >= 1:
            to_score = 5
        else:
            to_score = 0

        total = sharpe_score + fitness_score + to_score
        logger.debug(
            "[MUTATION-ENGINE] Score computation | sharpe=%.2f fitness=%.2f turnover=%.1f → total=%.1f",
            sharpe,
            fitness,
            turnover,
            total,
        )
        return total

    def _suggest_replacements(self, expr: str) -> dict[str, list[str]]:
        """Suggest operator replacements based on expression content.

        Args:
            expr: Factor expression

        Returns:
            Dict mapping operators to their replacement candidates
        """
        suggestions = {}
        expr_lower = expr.lower()
        for op, replacements in self._operator_replacements.items():
            if op + "(" in expr_lower:
                suggestions[op] = replacements
        return suggestions

    def _is_single_signal(self, expr: str) -> bool:
        """Check if expression uses only one base variable.

        Args:
            expr: Factor expression

        Returns:
            True if only one base variable detected
        """
        base_vars = {"open", "high", "low", "close", "volume", "amount", "vwap"}
        expr_lower = expr.lower()
        used = [v for v in base_vars if v in expr_lower]
        return len(used) <= 1

    def _extract_windows(self, expr: str) -> list[int]:
        """Extract window parameters from time-series operators.

        Args:
            expr: Factor expression

        Returns:
            Sorted list of unique window values
        """
        pattern = r"ts_\w+\([^,]+,\s*(\d+)\)"
        matches = re.findall(pattern, expr)
        return sorted(set(int(m) for m in matches))
