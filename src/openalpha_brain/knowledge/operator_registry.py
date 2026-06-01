from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class OperatorCategory(Enum):
    TEMPORAL = "temporal"
    CROSS_SECTIONAL = "cross_sectional"
    GROUP = "group"
    MATH = "math"
    CONDITIONAL = "conditional"
    LOGICAL = "logical"
    VECTOR = "vector"
    TRANSFORMATIONAL = "transformational"


@dataclass
class OperatorDef:
    name: str
    category: OperatorCategory
    min_args: int
    max_args: int
    args_type: list[str] = field(default_factory=list)
    requires_lookback: bool = False
    default_lookback: int | None = None
    description: str = ""
    risk_level: str = "medium"
    alternatives: list[str] = field(default_factory=list)
    forbidden_combos: list[str] = field(default_factory=list)
    use_count: int = 0
    success_count: int = 0
    avg_sharpe: float = 0.0


class OperatorRegistry:
    """WQ BRAIN 算子注册表 — 统一管理 66 个算子"""

    def __init__(self):
        self._operators: dict[str, OperatorDef] = {}
        self._by_category: dict[OperatorCategory, list[str]] = {}
        self._load_builtin_operators()

    def _load_builtin_operators(self):
        self.register(OperatorDef(
            name="ts_mean", category=OperatorCategory.TEMPORAL,
            min_args=2, max_args=2, args_type=["field", "window"],
            requires_lookback=True, default_lookback=20,
            description="时间序列均值",
            risk_level="medium", alternatives=["ts_decay_linear", "ts_sum", "ts_median"],
        ))

        self.register(OperatorDef(
            name="ts_decay_linear", category=OperatorCategory.TEMPORAL,
            min_args=2, max_args=3, args_type=["field", "window", "decay"],
            requires_lookback=True, default_lookback=10,
            description="线性衰减加权 — 过审因子核心算子 ⭐",
            risk_level="low", alternatives=["ts_mean", "ts_sum"],
        ))

        self.register(OperatorDef(
            name="ts_std_dev", category=OperatorCategory.TEMPORAL,
            min_args=2, max_args=2, args_type=["field", "window"],
            requires_lookback=True, default_lookback=20,
            description="时间序列标准差",
            risk_level="medium", alternatives=["ts_av_diff", "ts_zscore"],
        ))

        self.register(OperatorDef(
            name="ts_zscore", category=OperatorCategory.TEMPORAL,
            min_args=2, max_args=2, args_type=["field", "window"],
            requires_lookback=True, default_lookback=20,
            description="时间序列 Z-Score 标准化",
            risk_level="low", alternatives=["ts_av_diff", "ts_rank"],
        ))

        self.register(OperatorDef(
            name="ts_rank", category=OperatorCategory.TEMPORAL,
            min_args=2, max_args=3, args_type=["field", "window", "constant"],
            requires_lookback=True, default_lookback=20,
            description="时间序列排名",
            risk_level="low", alternatives=["ts_quantile", "ts_zscore"],
        ))

        self.register(OperatorDef(
            name="ts_delay", category=OperatorCategory.TEMPORAL,
            min_args=2, max_args=2, args_type=["field", "days"],
            requires_lookback=True, default_lookback=1,
            description="时间延迟 — 获取 d 天前的值",
            risk_level="medium", alternatives=["ts_delta"],
        ))

        self.register(OperatorDef(
            name="ts_sum", category=OperatorCategory.TEMPORAL,
            min_args=2, max_args=2, args_type=["field", "window"],
            requires_lookback=True, default_lookback=20,
            description="时间序列求和",
            risk_level="medium", alternatives=["ts_mean", "ts_decay_linear"],
        ))

        self.register(OperatorDef(
            name="ts_corr", category=OperatorCategory.TEMPORAL,
            min_args=3, max_args=3, args_type=["field_x", "field_y", "window"],
            requires_lookback=True, default_lookback=20,
            description="时间序列相关性",
            risk_level="low", alternatives=["ts_covariance", "ts_regression"],
        ))

        self.register(OperatorDef(
            name="ts_delta", category=OperatorCategory.TEMPORAL,
            min_args=2, max_args=2, args_type=["field", "window"],
            requires_lookback=True, default_lookback=1,
            description="时间序列差分 — 动量因子核心 ⭐",
            risk_level="high", alternatives=["ts_decay_linear", "ts_av_diff"],
            forbidden_combos=["ts_delta(close,", "ts_delta(volume,"],
        ))

        self.register(OperatorDef(
            name="ts_av_diff", category=OperatorCategory.TEMPORAL,
            min_args=2, max_args=2, args_type=["field", "window"],
            requires_lookback=True, default_lookback=20,
            description="均值偏离 — x - ts_mean(x, d)",
            risk_level="low", alternatives=["ts_zscore", "ts_decay_linear"],
        ))

        self.register(OperatorDef(
            name="ts_quantile", category=OperatorCategory.TEMPORAL,
            min_args=2, max_args=3, args_type=["field", "window", "driver"],
            requires_lookback=True, default_lookback=60,
            description="时间序列分位数变换",
            risk_level="low", alternatives=["ts_rank", "ts_zscore"],
        ))

        self.register(OperatorDef(
            name="ts_regression", category=OperatorCategory.TEMPORAL,
            min_args=3, max_args=5, args_type=["y", "x", "window", "lag", "rettype"],
            requires_lookback=True, default_lookback=20,
            description="时间序列回归 — 提取 beta/alpha/residual",
            risk_level="low", alternatives=["ts_corr", "ts_covariance"],
        ))

        self.register(OperatorDef(
            name="ts_arg_max", category=OperatorCategory.TEMPORAL,
            min_args=2, max_args=2, args_type=["field", "window"],
            requires_lookback=True, default_lookback=20,
            description="距最大值出现的天数",
            risk_level="medium", alternatives=["ts_arg_min", "ts_rank"],
        ))

        self.register(OperatorDef(
            name="ts_arg_min", category=OperatorCategory.TEMPORAL,
            min_args=2, max_args=2, args_type=["field", "window"],
            requires_lookback=True, default_lookback=20,
            description="距最小值出现的天数",
            risk_level="medium", alternatives=["ts_arg_max", "ts_rank"],
        ))

        self.register(OperatorDef(
            name="ts_skewness", category=OperatorCategory.TEMPORAL,
            min_args=2, max_args=2, args_type=["field", "window"],
            requires_lookback=True, default_lookback=20,
            description="时间序列偏度",
            risk_level="medium", alternatives=["ts_kurtosis", "ts_std_dev"],
        ))

        self.register(OperatorDef(
            name="ts_kurtosis", category=OperatorCategory.TEMPORAL,
            min_args=2, max_args=2, args_type=["field", "window"],
            requires_lookback=True, default_lookback=20,
            description="时间序列峰度",
            risk_level="medium", alternatives=["ts_skewness", "ts_std_dev"],
        ))

        self.register(OperatorDef(
            name="ts_scale", category=OperatorCategory.TEMPORAL,
            min_args=2, max_args=3, args_type=["field", "window", "constant"],
            requires_lookback=True, default_lookback=20,
            description="时间序列缩放到 [0,1]",
            risk_level="low", alternatives=["ts_rank", "ts_zscore"],
        ))

        self.register(OperatorDef(
            name="ts_product", category=OperatorCategory.TEMPORAL,
            min_args=2, max_args=2, args_type=["field", "window"],
            requires_lookback=True, default_lookback=20,
            description="时间序列乘积 — 几何平均相关",
            risk_level="medium", alternatives=["ts_sum", "ts_mean"],
        ))

        self.register(OperatorDef(
            name="ts_covariance", category=OperatorCategory.TEMPORAL,
            min_args=3, max_args=3, args_type=["field_y", "field_x", "window"],
            requires_lookback=True, default_lookback=20,
            description="时间序列协方差",
            risk_level="medium", alternatives=["ts_corr", "ts_regression"],
        ))

        self.register(OperatorDef(
            name="ts_backfill", category=OperatorCategory.TEMPORAL,
            min_args=2, max_args=3, args_type=["field", "lookback", "k"],
            requires_lookback=True, default_lookback=5,
            description="用最近有效值填充 NaN",
            risk_level="low", alternatives=["kth_element"],
        ))

        self.register(OperatorDef(
            name="ts_count_nans", category=OperatorCategory.TEMPORAL,
            min_args=2, max_args=2, args_type=["field", "window"],
            requires_lookback=True, default_lookback=20,
            description="统计窗口内 NaN 数量",
            risk_level="low", alternatives=[],
        ))

        self.register(OperatorDef(
            name="kth_element", category=OperatorCategory.TEMPORAL,
            min_args=3, max_args=4, args_type=["field", "window", "k", "ignore"],
            requires_lookback=True, default_lookback=5,
            description="获取第 K 个历史值",
            risk_level="low", alternatives=["ts_backfill", "ts_delay"],
        ))

        self.register(OperatorDef(
            name="hump", category=OperatorCategory.TRANSFORMATIONAL,
            min_args=1, max_args=2, args_type=["field", "hump_size"],
            requires_lookback=False, default_lookback=None,
            description="限制变化幅度 — 降低换手率",
            risk_level="low", alternatives=[],
        ))

        self.register(OperatorDef(
            name="last_diff_value", category=OperatorCategory.TEMPORAL,
            min_args=2, max_args=2, args_type=["field", "window"],
            requires_lookback=True, default_lookback=20,
            description="最近一个不同于当前值的值",
            risk_level="medium", alternatives=["ts_delay", "kth_element"],
        ))

        self.register(OperatorDef(
            name="ts_step", category=OperatorCategory.TEMPORAL,
            min_args=1, max_args=1, args_type=["counter"],
            requires_lookback=False, default_lookback=None,
            description="日计数器 — 每天递增",
            risk_level="low", alternatives=[],
        ))

        self.register(OperatorDef(
            name="days_from_last_change", category=OperatorCategory.TEMPORAL,
            min_args=1, max_args=1, args_type=["field"],
            requires_lookback=False, default_lookback=None,
            description="距离上次值改变的天数",
            risk_level="low", alternatives=["ts_step"],
        ))

        self.register(OperatorDef(
            name="rank", category=OperatorCategory.CROSS_SECTIONAL,
            min_args=1, max_args=2, args_type=["field", "rate"],
            requires_lookback=False, default_lookback=None,
            description="截面排名 — 归一化到 [0,1] ⭐",
            risk_level="low", alternatives=["zscore", "scale", "group_rank"],
        ))

        self.register(OperatorDef(
            name="zscore", category=OperatorCategory.CROSS_SECTIONAL,
            min_args=1, max_args=1, args_type=["field"],
            requires_lookback=False, default_lookback=None,
            description="截面 Z-Score 标准化",
            risk_level="low", alternatives=["rank", "normalize", "group_zscore"],
        ))

        self.register(OperatorDef(
            name="scale", category=OperatorCategory.CROSS_SECTIONAL,
            min_args=1, max_args=4, args_type=["field", "scale", "longscale", "shortscale"],
            requires_lookback=False, default_lookback=None,
            description="缩放到目标仓位规模",
            risk_level="low", alternatives=["rank", "normalize"],
        ))

        self.register(OperatorDef(
            name="winsorize", category=OperatorCategory.CROSS_SECTIONAL,
            min_args=1, max_args=2, args_type=["field", "std"],
            requires_lookback=False, default_lookback=None,
            description="截断极端值 — 默认 4 倍标准差",
            risk_level="low", alternatives=["normalize", "quantile"],
        ))

        self.register(OperatorDef(
            name="normalize", category=OperatorCategory.CROSS_SECTIONAL,
            min_args=1, max_args=3, args_type=["field", "useStd", "limit"],
            requires_lookback=False, default_lookback=None,
            description="截面标准化 — 去均值/除标准差",
            risk_level="low", alternatives=["zscore", "winsorize"],
        ))

        self.register(OperatorDef(
            name="quantile", category=OperatorCategory.CROSS_SECTIONAL,
            min_args=1, max_args=3, args_type=["field", "driver", "sigma"],
            requires_lookback=False, default_lookback=None,
            description="分位数变换 — 降低异常值影响",
            risk_level="low", alternatives=["rank", "winsorize"],
        ))

        self.register(OperatorDef(
            name="group_neutralize", category=OperatorCategory.GROUP,
            min_args=2, max_args=2, args_type=["field", "group"],
            requires_lookback=False, default_lookback=None,
            description="组内去均值 — 行业/板块中性 ⭐",
            risk_level="low", alternatives=["group_zscore", "group_mean"],
        ))

        self.register(OperatorDef(
            name="group_rank", category=OperatorCategory.GROUP,
            min_args=2, max_args=2, args_type=["field", "group"],
            requires_lookback=False, default_lookback=None,
            description="组内排名",
            risk_level="low", alternatives=["rank", "group_zscore"],
        ))

        self.register(OperatorDef(
            name="group_zscore", category=OperatorCategory.GROUP,
            min_args=2, max_args=2, args_type=["field", "group"],
            requires_lookback=False, default_lookback=None,
            description="组内 Z-Score",
            risk_level="low", alternatives=["group_neutralize", "zscore"],
        ))

        self.register(OperatorDef(
            name="group_mean", category=OperatorCategory.GROUP,
            min_args=2, max_args=3, args_type=["field", "weight", "group"],
            requires_lookback=False, default_lookback=None,
            description="组内均值（加权调和平均）",
            risk_level="low", alternatives=["group_neutralize"],
        ))

        self.register(OperatorDef(
            name="group_scale", category=OperatorCategory.GROUP,
            min_args=2, max_args=2, args_type=["field", "group"],
            requires_lookback=False, default_lookback=None,
            description="组内归一化到 [0,1]",
            risk_level="low", alternatives=["group_rank", "scale"],
        ))

        self.register(OperatorDef(
            name="group_backfill", category=OperatorCategory.GROUP,
            min_args=3, max_args=4, args_type=["field", "group", "window", "std"],
            requires_lookback=True, default_lookback=5,
            description="组内回填 NaN — 用 winsorized mean 填充",
            risk_level="low", alternatives=["ts_backfill"],
        ))

        self.register(OperatorDef(
            name="trade_when", category=OperatorCategory.CONDITIONAL,
            min_args=3, max_args=3, args_type=["value", "condition", "exit_condition"],
            requires_lookback=False, default_lookback=None,
            description="条件交易 — 满足条件时更新信号 ⭐",
            risk_level="low", alternatives=[],
        ))

        self.register(OperatorDef(
            name="add", category=OperatorCategory.MATH,
            min_args=2, max_args=0, args_type=["x", "y"],
            requires_lookback=False, default_lookback=None,
            description="加法 x + y",
            risk_level="low", alternatives=["multiply"],
        ))

        self.register(OperatorDef(
            name="subtract", category=OperatorCategory.MATH,
            min_args=2, max_args=0, args_type=["x", "y"],
            requires_lookback=False, default_lookback=None,
            description="减法 x - y",
            risk_level="low", alternatives=["add", "reverse"],
        ))

        self.register(OperatorDef(
            name="multiply", category=OperatorCategory.MATH,
            min_args=2, max_args=0, args_type=["x", "y"],
            requires_lookback=False, default_lookback=None,
            description="乘法 x * y",
            risk_level="low", alternatives=["power"],
        ))

        self.register(OperatorDef(
            name="divide", category=OperatorCategory.MATH,
            min_args=2, max_args=2, args_type=["x", "y"],
            requires_lookback=False, default_lookback=None,
            description="除法 x / y",
            risk_level="low", alternatives=["inverse", "power"],
        ))

        self.register(OperatorDef(
            name="power", category=OperatorCategory.MATH,
            min_args=2, max_args=2, args_type=["x", "y"],
            requires_lookback=False, default_lookback=None,
            description="幂运算 x ^ y",
            risk_level="medium", alternatives=["signed_power", "sqrt"],
        ))

        self.register(OperatorDef(
            name="sqrt", category=OperatorCategory.MATH,
            min_args=1, max_args=1, args_type=["x"],
            requires_lookback=False, default_lookback=None,
            description="平方根 √x",
            risk_level="low", alternatives=["power(x, 0.5)"],
        ))

        self.register(OperatorDef(
            name="log", category=OperatorCategory.MATH,
            min_args=1, max_args=1, args_type=["x"],
            requires_lookback=False, default_lookback=None,
            description="自然对数 ln(x) — 对数变换 ⭐",
            risk_level="low", alternatives=[],
        ))

        self.register(OperatorDef(
            name="abs", category=OperatorCategory.MATH,
            min_args=1, max_args=1, args_type=["x"],
            requires_lookback=False, default_lookback=None,
            description="绝对值 |x|",
            risk_level="low", alternatives=["signed_power(x, 2)"],
        ))

        self.register(OperatorDef(
            name="min", category=OperatorCategory.MATH,
            min_args=2, max_args=0, args_type=["x", "y"],
            requires_lookback=False, default_lookback=None,
            description="最小值",
            risk_level="low", alternatives=["max"],
        ))

        self.register(OperatorDef(
            name="max", category=OperatorCategory.MATH,
            min_args=2, max_args=0, args_type=["x", "y"],
            requires_lookback=False, default_lookback=None,
            description="最大值",
            risk_level="low", alternatives=["min"],
        ))

        self.register(OperatorDef(
            name="sign", category=OperatorCategory.MATH,
            min_args=1, max_args=1, args_type=["x"],
            requires_lookback=False, default_lookback=None,
            description="符号函数 — 返回 -1/0/1",
            risk_level="low", alternatives=[],
        ))

        self.register(OperatorDef(
            name="reverse", category=OperatorCategory.MATH,
            min_args=1, max_args=1, args_type=["x"],
            requires_lookback=False, default_lookback=None,
            description="取反 -x",
            risk_level="low", alternatives=["multiply(x, -1)"],
        ))

        self.register(OperatorDef(
            name="inverse", category=OperatorCategory.MATH,
            min_args=1, max_args=1, args_type=["x"],
            requires_lookback=False, default_lookback=None,
            description="倒数 1/x",
            risk_level="medium", alternatives=["divide(1, x)", "power(x, -1)"],
        ))

        self.register(OperatorDef(
            name="signed_power", category=OperatorCategory.MATH,
            min_args=2, max_args=2, args_type=["x", "y"],
            requires_lookback=False, default_lookback=None,
            description="有符号幂 — 保持 x 的符号",
            risk_level="low", alternatives=["power"],
        ))

        self.register(OperatorDef(
            name="densify", category=OperatorCategory.MATH,
            min_args=1, max_args=1, args_type=["field"],
            requires_lookback=False, default_lookback=None,
            description="稠密化分组字段 — 减少桶数量",
            risk_level="low", alternatives=[],
        ))

        self.register(OperatorDef(
            name="and", category=OperatorCategory.LOGICAL,
            min_args=2, max_args=2, args_type=["input1", "input2"],
            requires_lookback=False, default_lookback=None,
            description="逻辑与 AND",
            risk_level="low", alternatives=[],
        ))

        self.register(OperatorDef(
            name="or", category=OperatorCategory.LOGICAL,
            min_args=2, max_args=2, args_type=["input1", "input2"],
            requires_lookback=False, default_lookback=None,
            description="逻辑或 OR",
            risk_level="low", alternatives=[],
        ))

        self.register(OperatorDef(
            name="not", category=OperatorCategory.LOGICAL,
            min_args=1, max_args=1, args_type=["x"],
            requires_lookback=False, default_lookback=None,
            description="逻辑非 NOT",
            risk_level="low", alternatives=[],
        ))

        self.register(OperatorDef(
            name="greater", category=OperatorCategory.LOGICAL,
            min_args=2, max_args=2, args_type=["input1", "input2"],
            requires_lookback=False, default_lookback=None,
            description="大于 >",
            risk_level="low", alternatives=["less"],
        ))

        self.register(OperatorDef(
            name="less", category=OperatorCategory.LOGICAL,
            min_args=2, max_args=2, args_type=["input1", "input2"],
            requires_lookback=False, default_lookback=None,
            description="小于 <",
            risk_level="low", alternatives=["greater"],
        ))

        self.register(OperatorDef(
            name="equal", category=OperatorCategory.LOGICAL,
            min_args=2, max_args=2, args_type=["input1", "input2"],
            requires_lookback=False, default_lookback=None,
            description="等于 ==",
            risk_level="low", alternatives=["not_equal"],
        ))

        self.register(OperatorDef(
            name="not_equal", category=OperatorCategory.LOGICAL,
            min_args=2, max_args=2, args_type=["input1", "input2"],
            requires_lookback=False, default_lookback=None,
            description="不等 !=",
            risk_level="low", alternatives=["equal"],
        ))

        self.register(OperatorDef(
            name="greater_equal", category=OperatorCategory.LOGICAL,
            min_args=2, max_args=2, args_type=["input1", "input2"],
            requires_lookback=False, default_lookback=None,
            description="大于等于 >=",
            risk_level="low", alternatives=["less_equal"],
        ))

        self.register(OperatorDef(
            name="less_equal", category=OperatorCategory.LOGICAL,
            min_args=2, max_args=2, args_type=["input1", "input2"],
            requires_lookback=False, default_lookback=None,
            description="小于等于 <=",
            risk_level="low", alternatives=["greater_equal"],
        ))

        self.register(OperatorDef(
            name="is_nan", category=OperatorCategory.LOGICAL,
            min_args=1, max_args=1, args_type=["input"],
            requires_lookback=False, default_lookback=None,
            description="判断是否为 NaN",
            risk_level="low", alternatives=[],
        ))

        self.register(OperatorDef(
            name="if_else", category=OperatorCategory.LOGICAL,
            min_args=3, max_args=3, args_type=["condition", "true_value", "false_value"],
            requires_lookback=False, default_lookback=None,
            description="条件分支 if-else ⭐",
            risk_level="low", alternatives=["trade_when"],
        ))

        self.register(OperatorDef(
            name="vec_sum", category=OperatorCategory.VECTOR,
            min_args=1, max_args=1, args_type=["vector_field"],
            requires_lookback=False, default_lookback=None,
            description="向量求和",
            risk_level="low", alternatives=["vec_avg"],
        ))

        self.register(OperatorDef(
            name="vec_avg", category=OperatorCategory.VECTOR,
            min_args=1, max_args=1, args_type=["vector_field"],
            requires_lookback=False, default_lookback=None,
            description="向量平均值",
            risk_level="low", alternatives=["vec_sum"],
        ))

        self.register(OperatorDef(
            name="bucket", category=OperatorCategory.TRANSFORMATIONAL,
            min_args=2, max_args=5, args_type=["field", "range", "buckets", "skipBoth", "NaNGroup"],
            requires_lookback=False, default_lookback=None,
            description="分桶 — 将排名值离散化为组别",
            risk_level="low", alternatives=["group_neutralize", "group_rank"],
        ))

    def register(self, op_def: OperatorDef):
        self._operators[op_def.name] = op_def
        cat = op_def.category
        if cat not in self._by_category:
            self._by_category[cat] = []
        self._by_category[cat].append(op_def.name)

    def get(self, name: str) -> OperatorDef | None:
        return self._operators.get(name)

    def get_by_category(self, cat: OperatorCategory) -> list[OperatorDef]:
        return [self._operators[n] for n in self._by_category.get(cat, [])]

    def get_temporal_operators(self) -> list[OperatorDef]:
        return self.get_by_category(OperatorCategory.TEMPORAL)

    def get_low_risk_operators(self) -> list[OperatorDef]:
        return [op for op in self._operators.values() if op.risk_level == "low"]

    def get_alternatives(self, op_name: str) -> list[str]:
        op = self._operators.get(op_name)
        return op.alternatives if op else []

    def validate_expression_operators(self, expr: str) -> tuple[bool, list[str]]:
        errors = []
        found_ops = set(re.findall(r'([a-zA-Z_]\w*)\s*\(', expr.lower()))
        known_ops = self._operators.keys()
        for op_name in found_ops:
            if op_name not in known_ops and not op_name.startswith(('ts_', 'group_')):
                if op_name in ['rank', 'zscore', 'scale', 'tanh', 'sigmoid', 'signed_power',
                              'log', 'abs', 'vec_sum', 'vec_avg', 'trade_when', 'bucket',
                              'if_else', 'hump', 'winsorize', 'normalize', 'quantile',
                              'add', 'subtract', 'multiply', 'divide', 'power', 'sqrt',
                              'min', 'max', 'sign', 'reverse', 'inverse', 'densify',
                              'and', 'or', 'not', 'greater', 'less', 'equal',
                              'not_equal', 'greater_equal', 'less_equal', 'is_nan']:
                    continue
                errors.append(f"未知算子: {op_name}")
        return len(errors) == 0, errors

    def count_operators(self, expr: str) -> int:
        return len(re.findall(
            r'(?:ts_\w+|group_\w+|rank|zscore|scale|tanh|sigmoid|signed_power|log|abs|vec_\w+|trade_when|bucket|if_else|hump|winsorize|normalize|quantile)\s*\(',
            expr.lower()
        ))

    def get_forbidden_patterns(self) -> list[tuple[str, str]]:
        patterns = [
            (r'ts_delta\s*\(\s*close\s*,', 'ts_delta(close) 单独使用过于拥挤'),
            (r'ts_delta\s*\(\s*volume\s*,', 'ts_delta(volume) 单独使用过于拥挤'),
            (r'rank\s*\(\s*ts_mean\s*\([^)]{0,20}\)\s*\)', 'rank(ts_mean(x)) 结构过于简单'),
        ]
        all_patterns = list(patterns)
        for op in self._operators.values():
            for combo in op.forbidden_combos:
                all_patterns.append((re.escape(combo), f'{op.name} 禁止组合'))
        return all_patterns

    def record_usage(self, op_name: str, sharpe: float):
        op = self._operators.get(op_name)
        if op:
            op.use_count += 1
            if sharpe > 0:
                op.success_count += 1
                total = op.success_count
                if total > 0:
                    op.avg_sharpe = (op.avg_sharpe * (total - 1) + sharpe) / total

    def get_stats(self) -> dict:
        return {
            "total_operators": len(self._operators),
            "most_used": sorted(self._operators.values(), key=lambda x: x.use_count, reverse=True)[:5],
            "highest_success_rate": sorted(
                [o for o in self._operators.values() if o.use_count > 0],
                key=lambda x: x.success_count / x.use_count,
                reverse=True
            )[:5],
        }

    def get_all_operator_names(self) -> set[str]:
        return set(self._operators.keys())

    def get_operators_requiring_lookback(self) -> list[OperatorDef]:
        return [op for op in self._operators.values() if op.requires_lookback]

    def suggest_alternative_for_high_risk(self, op_name: str) -> str | None:
        op = self._operators.get(op_name)
        if op and op.risk_level == "high" and op.alternatives:
            low_risk_alts = [a for a in op.alternatives if a in self._operators]
            alt_op = self._operators.get(low_risk_alts[0]) if low_risk_alts else None
            if alt_op and alt_op.risk_level != "high":
                return low_risk_alts[0]
        return None


_registry_instance: OperatorRegistry | None = None


def get_operator_registry() -> OperatorRegistry:
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = OperatorRegistry()
    return _registry_instance
