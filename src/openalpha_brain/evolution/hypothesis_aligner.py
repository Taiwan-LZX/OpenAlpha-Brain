"""
R² Hypothesis Alignment Scorer for AlphaAgent (KDD'25)

實作 AlphaAgent 論文中的核心創新：R² hypothesis alignment scoring。
衡量 FactorAgent 生成的 FASTEXPR 表達式與 IdeaAgent 產生的經濟假設之間的語義一致性，
防止「說一套做一套」的語義漂移（semantic drift）。

評分維度：
1. Operator Match (30%) — 運算子與假設模板的匹配度
2. Field Match (20%) — 欄位與假設預期欄位的匹配度
3. Direction Consistency (25%) — 表達式方向與聲稱方向的一致性
4. Structural Fit (25%) — 整體結構複雜度與假設的適配度

使用方式：
    aligner = HypothesisAligner()
    result = aligner.align(expression, "momentum_long")
    print(result["r2_score"], result["alignment_level"])
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from openalpha_brain.monitoring.algorithm_telemetry import AlgorithmTelemetryCollector
from openalpha_brain.utils.algo_logger import Timer, algo_log, log_call

logger = logging.getLogger(__name__)


@dataclass
class HypothesisTemplate:
    """單一假設模板，描述某種 alpha 策略類型的預期特徵。"""

    name: str
    expected_operators: list[str] = field(default_factory=list)
    avoid_operators: list[str] = field(default_factory=list)
    expected_fields: list[str] = field(default_factory=list)
    structural_signature: str = ""
    direction_bias: float = 0.0
    description: str = ""


class HypothesisAligner:
    """R² Hypothesis Alignment Scorer for AlphaAgent-style consistency check.

    Scores how well a generated FASTEXPR expression aligns with its claimed
    economic hypothesis (direction/strategy type). Prevents "say one thing,
    do another" semantic drift.

    核心機制：
    - 維護 8+ 種內建假設模板（動量、均值復歸、價值因子等）
    - 從四個維度計算 R² 對齊分數
    - 支援反向推斷：從表達式自動偵測最可能的假設類型
    - 生成 LLM 可讀的反饋文字，嵌入 mutation prompt 引導修正
    """

    _PRICE_FIELDS: set[str] = {"close", "open", "high", "low", "vwap", "returns"}
    _VOLUME_FIELDS: set[str] = {"volume", "adv20"}
    _FUNDAMENTAL_FIELDS: set[str] = {
        "cap",
        "sales",
        "assets",
        "equity",
        "revenue",
        "earnings",
        "sharesout",
        "liabilities",
        "debt",
    }
    _TECHNICAL_OPS: set[str] = {
        "ts_delta",
        "ts_decay_linear",
        "ts_rank",
        "ts_regression",
        "ts_corr",
        "ts_covariance",
    }
    _FUNDAMENTAL_OPS: set[str] = {
        "ts_mean",
        "ts_std_dev",
        "ts_zscore",
        "ts_sum",
        "ts_product",
        "ts_av_diff",
        "ts_skewness",
        "ts_kurtosis",
    }

    _OPERATOR_RE = re.compile(
        r"\b(ts_\w+|rank|group_neutralize|group_rank|zscore|group_zscore|scale|signed_power|abs|log|sign|delta|correlation|covariance|ts_product|ts_sum|ts_min|ts_max|ts_argmax|ts_argmin|ts_skewness|ts_kurtosis|ts_av_diff|ts_std_dev|regression)\b"
    )
    _FIELD_RE = re.compile(
        r"\b(close|open|high|low|vwap|volume|adv\d+|returns|cap|sales|assets|equity|revenue|earnings|sharesout|vwap\d+)\b",
        re.IGNORECASE,
    )
    _NUMBER_RE = re.compile(r"\b(\d+)\b")

    def __init__(self) -> None:
        self._templates: dict[str, HypothesisTemplate] = self._build_template_library()
        self._tel = AlgorithmTelemetryCollector.get_instance()
        self._direction_keywords: dict[str, list[str]] = {
            "momentum_long": ["momentum", "trend", "continuation", "breakout", "動量", "趨勢", "延續"],
            "momentum_short": ["reversal", "contrarian", "mean-reversion", "short", "反轉", "空頭"],
            "mean_reversion": ["mean-reversion", "revert", "oversold", "overbought", "均值復歸", "回歸"],
            "value_factor": ["value", "cheap", "book-to-market", "pb", "pe", "價值", "便宜"],
            "quality_factor": ["quality", "profitability", "roa", "roe", "debt", "品質", "盈利能力"],
            "low_volatility": ["low-vol", "volatility", "beta", "risk", "低波動", "風險"],
            "volume_price": ["volume-price", "liquidity", "flow", "量價", "流動性"],
            "earnings_momentum": ["earnings", "surprise", "revision", "analyst", "盈利動量", "盈餘驚喜"],
            "temporal": ["temporal", "time-series", "時間序列", "時序", "technical", "技術面", "pattern"],
            "cross_sectional": ["cross-sectional", "截面", "橫截面", "relative", "相對排名"],
            "industry_rotation": ["industry", "rotation", "sector", "行業輪動", "行業", "板塊"],
            "lead_lag": ["lead-lag", "lead_lag", "leading", "lagging", "information diffusion", "領先滯後", "傳導"],
            "liquidity": ["liquidity", "amihud", "illiquidity", "turnover", "流動性", "非流動性"],
            "sentiment": ["sentiment", "behavioral", "overreaction", "herding", "情緒", "行為", "過度反應"],
            "size": ["size", "small-cap", "market-cap", "小盤", "市值", "規模"],
            "growth": ["growth", "expansion", "momentum earnings", "成長", "增長", "擴張"],
        }
        self._fallback_weights: dict[str, float] = {
            "field_overlap": 0.30,
            "operator_similarity": 0.25,
            "structural_complexity": 0.20,
            "direction_keyword": 0.25,
        }

    def _build_template_library(self) -> dict[str, HypothesisTemplate]:
        """建構內建假設模板庫，涵蓋 8 種常見 alpha 策略類型。"""
        templates: dict[str, HypothesisTemplate] = {
            "momentum_long": HypothesisTemplate(
                name="momentum_long",
                expected_operators=["ts_delta", "ts_decay_linear", "ts_rank", "ts_zscore", "ts_regression"],
                avoid_operators=["ts_mean", "ts_sum"],
                expected_fields=["close", "returns", "vwap"],
                structural_signature=r"ts_delta\(.+,\s*\d+\)|ts_rank\(.+,\s*\d+\)",
                direction_bias=+1.0,
                description="動量多頭策略：追漲殺跌，利用價格延續性獲利",
            ),
            "momentum_short": HypothesisTemplate(
                name="momentum_short",
                expected_operators=["ts_delta", "ts_zscore", "ts_rank", "ts_regression"],
                avoid_operators=["ts_mean", "ts_sum"],
                expected_fields=["close", "returns", "vwap"],
                structural_signature=r"-?\s*ts_delta\(.+,\s*\d+\)|-?\s*ts_zscore\(.+,\s*\d+\)",
                direction_bias=-1.0,
                description="動量空頭/反轉策略：利用過度反應後的價格修正",
            ),
            "mean_reversion": HypothesisTemplate(
                name="mean_reversion",
                expected_operators=["ts_mean", "ts_decay_linear", "ts_zscore", "ts_std_dev", "ts_av_diff"],
                avoid_operators=["ts_delta", "ts_rank"],
                expected_fields=["close", "vwap", "returns", "adv20"],
                structural_signature=r".+-\s*ts_mean\(.+,\s*\d+\)|ts_zscore\(.+,\s*\d+\).*(?:<|>)",
                direction_bias=-1.0,
                description="均值復歸策略：價格偏離均值後預期回歸",
            ),
            "value_factor": HypothesisTemplate(
                name="value_factor",
                expected_operators=["rank", "group_rank", "zscore", "group_zscore", "log", "scale"],
                avoid_operators=["ts_delta", "ts_decay_linear"],
                expected_fields=["cap", "sales", "assets", "earnings", "revenue"],
                structural_signature=r"(?:rank|zscore)\(.*(?:cap|sales|earnings|revenue))",
                direction_bias=-1.0,
                description="價值因子：買入低估資產，賣出高估資產",
            ),
            "quality_factor": HypothesisTemplate(
                name="quality_factor",
                expected_operators=["rank", "group_rank", "zscore", "scale", "log"],
                avoid_operators=["ts_delta", "ts_decay_linear"],
                expected_fields=["earnings", "sales", "assets", "revenue", "cap"],
                structural_signature=r"(?:rank|zscore)\(.*(?:earnings|sales|revenue))",
                direction_bias=+1.0,
                description="品質因子：買入高品質（高 ROE/穩定盈利）公司",
            ),
            "low_volatility": HypothesisTemplate(
                name="low_volatility",
                expected_operators=["ts_std_dev", "ts_av_diff", "ts_zscore", "ts_decay_linear"],
                avoid_operators=["ts_delta", "ts_rank"],
                expected_fields=["close", "returns", "vwap", "volume"],
                structural_signature=r"ts_std_dev\(.+,\s*\d+\)|ts_av_diff\(.+,\s*\d+\)",
                direction_bias=-1.0,
                description="低波動異常：低波動股票長期報酬率高於預期",
            ),
            "volume_price": HypothesisTemplate(
                name="volume_price",
                expected_operators=["ts_delta", "ts_corr", "correlation", "ts_decay_linear", "ts_rank"],
                avoid_operators=[],
                expected_fields=["volume", "close", "vwap", "adv20"],
                structural_signature=r"(?:corr|correlation)\(.+,.+\)|ts_delta\(volume",
                direction_bias=+1.0,
                description="量價關係：成交量確認價格趨勢或預示反轉",
            ),
            "earnings_momentum": HypothesisTemplate(
                name="earnings_momentum",
                expected_operators=["ts_delta", "ts_decay_linear", "rank", "group_rank"],
                avoid_operators=["ts_mean", "ts_std_dev"],
                expected_fields=["earnings", "sales", "revenue", "cap"],
                structural_signature=r"ts_delta\(.+?(?:earnings|sales|revenue).*,\s*\d+\)",
                direction_bias=+1.0,
                description="盈利動量：盈利修訂方向具有持續性",
            ),
            "temporal": HypothesisTemplate(
                name="temporal",
                expected_operators=["ts_delta", "ts_decay_linear", "ts_zscore", "ts_rank", "ts_mean", "ts_std_dev"],
                avoid_operators=[],
                expected_fields=["close", "vwap", "returns", "volume", "adv20"],
                structural_signature=r"ts_(?:delta|decay_linear|zscore|rank)\(.+,\s*\d+\)",
                direction_bias=0.0,
                description="時間序列模式：利用價格/成交量的時間結構特徵（純技術面）",
            ),
            "cross_sectional": HypothesisTemplate(
                name="cross_sectional",
                expected_operators=["rank", "group_rank", "zscore", "group_zscore", "group_neutralize", "scale"],
                avoid_operators=["ts_delta", "ts_decay_linear"],
                expected_fields=["close", "vwap", "volume", "cap", "sales", "assets"],
                structural_signature=r"(?:rank|group_rank|zscore|group_zscore)\(.*(?:close|vwap|volume))",
                direction_bias=0.0,
                description="截面因子：利用股票間橫截面排名和相對位置關係",
            ),
            "industry_rotation": HypothesisTemplate(
                name="industry_rotation",
                expected_operators=["group_neutralize", "group_rank", "group_zscore", "group_mean", "ts_delta", "rank"],
                avoid_operators=[],
                expected_fields=["close", "vwap", "volume", "cap", "industry"],
                structural_signature=r"group_(?:neutralize|rank|zscore)\(.*,\s*industry\)",
                direction_bias=0.0,
                description="行業輪動：利用行業內相對強弱進行選股，通常配合 group 運算子",
            ),
            "lead_lag": HypothesisTemplate(
                name="lead_lag",
                expected_operators=["ts_delta", "ts_delay", "ts_corr", "correlation", "ts_decay_linear"],
                avoid_operators=[],
                expected_fields=["close", "vwap", "volume", "returns"],
                structural_signature=r"ts_(?:delta|delay|corr)\(.+,\s*\d+\)",
                direction_bias=+1.0,
                description="領先滯後效應：利用資產間的資訊傳導延遲獲利",
            ),
            "liquidity": HypothesisTemplate(
                name="liquidity",
                expected_operators=["ts_mean", "ts_decay_linear", "rank", "group_rank", "ts_delta"],
                avoid_operators=[],
                expected_fields=["volume", "adv20", "close", "vwap"],
                structural_signature=r"(?:adv\d+|volume)",
                direction_bias=+1.0,
                description="流動性因子：利用成交量與流動性特徵預測報酬",
            ),
            "sentiment": HypothesisTemplate(
                name="sentiment",
                expected_operators=["ts_delta", "ts_decay_linear", "ts_zscore", "rank", "group_rank"],
                avoid_operators=[],
                expected_fields=["close", "volume", "vwap", "returns"],
                structural_signature=r"ts_(?:delta|decay_linear|zscore)\(.+,\s*\d+\)",
                direction_bias=+1.0,
                description="情緒因子：利用市場情緒指標（如量價背離、極端收益）捕捉過度反應",
            ),
            "size": HypothesisTemplate(
                name="size",
                expected_operators=["rank", "group_rank", "zscore", "group_zscore", "log", "scale"],
                avoid_operators=["ts_delta", "ts_decay_linear"],
                expected_fields=["cap", "sharesout", "volume", "adv20"],
                structural_signature=r"(?:rank|zscore)\(.*(?:cap|sharesout))",
                direction_bias=-1.0,
                description="規模因子：小盤股效應，小市值公司長期報酬率較高",
            ),
            "growth": HypothesisTemplate(
                name="growth",
                expected_operators=["ts_delta", "ts_decay_linear", "rank", "group_rank", "ts_regression"],
                avoid_operators=["ts_mean", "ts_std_dev"],
                expected_fields=["sales", "revenue", "earnings", "assets", "cap"],
                structural_signature=r"ts_delta\(.+?(?:sales|revenue|earnings).*,\s*\d+\)",
                direction_bias=+1.0,
                description="成長因子：利用營收/盈利增長率選股，高增長公司獲得超額報酬",
            ),
        }
        return templates

    @algo_log(label="HypothesisAligner.align")
    def align(
        self,
        expression: str,
        hypothesis: str,
        direction: str = "",
    ) -> dict[str, Any]:
        eid = None
        try:
            eid = self._tel.record_enter_sync("HypothesisAligner", cycle_id="unknown", expr_id=hash(expression) % 10000)
            t0 = time.perf_counter()
            with Timer("align_computation"):
                expr_summary = expression[:80] + ("..." if len(expression) > 80 else "")
                log_call(
                    "align_input",
                    input={"expression": expr_summary, "hypothesis": hypothesis, "direction": direction},
                )

                template_key = self._resolve_hypothesis_key(hypothesis)
                template = self._templates.get(template_key)
                is_fallback = template is None

                if is_fallback:
                    logger.warning(
                        "HypothesisAligner: unknown hypothesis '%s', using multi-signal fallback scoring", hypothesis
                    )
                    result = self._fallback_scoring(expression, hypothesis)
                    result["calibrated_score"] = self._calibrate_score(result["r2_score"], "unknown")
                    result["fallback_triggered"] = True
                    log_call(
                        "align_fallback",
                        input={"expression": expr_summary, "hypothesis": hypothesis, "template_key": template_key},
                        output={
                            k: result[k]
                            for k in ["r2_score", "calibrated_score", "alignment_level", "fallback_signal_details"]
                            if k in result
                        },
                        extra={"fallback_reason": f"unknown_template:{hypothesis}"},
                    )
                    ms = (time.perf_counter() - t0) * 1000
                    with contextlib.suppress(OSError, ValueError, RuntimeError):
                        self._tel.record_exit_sync(
                            "HypothesisAligner",
                            eid,
                            metrics={
                                "r2_score": round(result.get("r2_score", 0), 4),
                                "alignment_level": result.get("alignment_level", "unknown"),
                            },
                            duration_ms=ms,
                        )
                    return result

                expr_ops = self._extract_operators(expression)
                expr_fields = self._extract_fields(expression)

                op_match = self._score_operator_match(expr_ops, template)
                field_match = self._score_field_match(expr_fields, template)
                dir_consistency = self._score_direction_consistency(expression, template, direction)
                structural_fit = self._score_structural_fit(expression, template)

                raw_r2 = op_match * 0.30 + field_match * 0.20 + dir_consistency * 0.25 + structural_fit * 0.25
                r2_score = round(raw_r2, 4)
                calibrated_score = round(self._calibrate_score(r2_score, template_key), 4)

            alignment_level = self._classify_alignment_level(r2_score)
            diagnosis = self._build_diagnosis(op_match, field_match, dir_consistency, structural_fit, template)
            suggestions = self._build_suggestions(
                op_match, field_match, dir_consistency, structural_fit, template, expression
            )

            economy = self._check_categorical_consistency(
                template_key,
                expr_ops,
                expr_fields,
                expression,
            )

            result = {
                "r2_score": r2_score,
                "calibrated_score": calibrated_score,
                "alignment_level": alignment_level,
                "operator_match": round(op_match, 4),
                "field_match": round(field_match, 4),
                "direction_consistency": round(dir_consistency, 4),
                "structural_fit": round(structural_fit, 4),
                "economic_consistent": economy["is_consistent"],
                "economic_conflict_type": economy["conflict_type"],
                "economic_confidence": economy["confidence"],
                "diagnosis": diagnosis,
                "suggestions": suggestions,
                "matched_template": template_key,
                "fallback_triggered": False,
            }

            log_call(
                "align_dimensions",
                input={"expression": expr_summary, "template": template_key},
                output={
                    "r2_score": r2_score,
                    "calibrated_score": calibrated_score,
                    "alignment_level": alignment_level,
                    "op_match": round(op_match, 4),
                    "field_match": round(field_match, 4),
                    "dir_consistency": round(dir_consistency, 4),
                    "structural_fit": round(structural_fit, 4),
                },
                extra={"is_fallback": False, "template_matched": True},
            )
            ms = (time.perf_counter() - t0) * 1000
            with contextlib.suppress(OSError, ValueError, RuntimeError):
                self._tel.record_exit_sync(
                    "HypothesisAligner",
                    eid,
                    metrics={"r2_score": r2_score, "alignment_level": alignment_level},
                    duration_ms=ms,
                )
            return result
        except Exception as e:
            if eid:
                with contextlib.suppress(OSError, ValueError, RuntimeError):
                    self._tel.record_error_sync("HypothesisAligner", str(e), type(e).__name__)
            raise

    def detect_hypothesis_from_expression(self, expression: str) -> str:
        """從表達式自動偵測最可能的假設類型。

        用於驗證 FactorAgent 是否偏離了 IdeaAgent 的意圖。
        計算表達式與每個模板的對齊分數，返回最高分的模板名稱。

        Args:
            expression: FASTEXPR 格式的 alpha 因子表達式

        Returns:
            最可能的假設模板名稱（如 "momentum_long"）
        """
        best_key = "unknown"
        best_score = -1.0

        for key, _template in self._templates.items():
            partial = self.align(expression, key)
            if partial["r2_score"] > best_score:
                best_score = partial["r2_score"]
                best_key = key

        return best_key

    def build_alignment_feedback(self, alignment_result: dict[str, Any]) -> str:
        """生成 LLM 可讀的反饋文字，嵌入到 mutation prompt 中引導修正。

        Args:
            alignment_result: align() 方法返回的結果字典

        Returns:
            格式化的反饋文字，可直接拼接到 LLM prompt 中
        """
        r2 = alignment_result.get("r2_score", 0.0)
        level = alignment_result.get("alignment_level", "unknown")
        diagnosis = alignment_result.get("diagnosis", "")
        suggestions = alignment_result.get("suggestions", [])
        matched = alignment_result.get("matched_template", "")

        lines = [
            f"[HYPOTHESIS ALIGNMENT CHECK] R²={r2:.3f} ({level.upper()})",
            f"Target hypothesis: {matched}",
            f"Diagnosis: {diagnosis}",
        ]

        if suggestions:
            lines.append("Suggestions:")
            for s in suggestions[:5]:
                lines.append(f"  - {s}")

        if level == "contradictory":
            lines.insert(1, "WARNING: Expression contradicts the claimed hypothesis!")

        return "\n".join(lines)

    def _resolve_hypothesis_key(self, hypothesis: str) -> str:
        """將假設輸入解析為標準模板鍵值。支援名稱匹配和關鍵詞模糊匹配。"""
        hypothesis_lower = hypothesis.lower().strip()

        if hypothesis_lower in self._templates:
            return hypothesis_lower

        for key, keywords in self._direction_keywords.items():
            for kw in keywords:
                if kw in hypothesis_lower:
                    return key

        alias_map = {
            "momentum": "momentum_long",
            "trend_following": "momentum_long",
            "reversal": "momentum_short",
            "contrarian": "momentum_short",
            "mr": "mean_reversion",
            "mean_revert": "mean_reversion",
            "value": "value_factor",
            "bm": "value_factor",
            "quality": "quality_factor",
            "lowvol": "low_volatility",
            "min_variance": "low_volatility",
            "vp": "volume_price",
            "obv": "volume_price",
            "em": "earnings_momentum",
            "eps_momentum": "earnings_momentum",
        }

        for alias, canonical in alias_map.items():
            if alias in hypothesis_lower.replace("-", "").replace("_", ""):
                return canonical

        return hypothesis_lower

    def _extract_operators(self, expression: str) -> list[str]:
        """從 FASTEXPR 表達式中提取所有運算子名稱。"""
        return list(dict.fromkeys(self._OPERATOR_RE.findall(expression)))

    def _extract_fields(self, expression: str) -> list[str]:
        """從 FASTEXPR 表達式中提取所有欄位名稱。"""
        return [f.lower() for f in dict.fromkeys(self._FIELD_RE.findall(expression))]

    def _score_operator_match(self, expr_ops: list[str], template: HypothesisTemplate) -> float:
        """計算運算子匹配度 (維度 1, 權重 30%)。

        使用加權 Jaccard similarity：
        - 出現 expected_operators 中的運算子加分（核心運算子加權更高）
        - 出現 avoid_operators 中的運算子扣分
        """
        if not expr_ops:
            return 0.3

        expected_list = template.expected_operators
        avoid_set = {o.lower() for o in template.avoid_operators}
        expr_set = {o.lower() for o in expr_ops}

        core_expected = {o.lower() for o in expected_list[:2]}
        full_expected = {o.lower() for o in expected_list}

        core_hits = len(core_expected & expr_set)
        full_hits = len(full_expected & expr_set)
        penalty_count = len(avoid_set & expr_set)

        if core_hits >= 1 and len(expected_list) > 0:
            base_score = 0.5 + (full_hits / len(expected_list)) * 0.5
        elif full_hits > 0:
            base_score = 0.3 + (full_hits / len(expected_list)) * 0.4
        else:
            base_score = 0.15

        penalty = penalty_count * 0.25

        return max(0.0, min(1.0, base_score - penalty))

    def _score_field_match(self, expr_fields: list[str], template: HypothesisTemplate) -> float:
        """計算欄位匹配度 (維度 2, 權重 20%)。

        使用召回率 + 同族字段獎勵代替嚴格交集比，避免因缺少少數預期欄位而大幅扣分。
        """
        if not expr_fields:
            return 0.3

        expected_set = {f.lower() for f in template.expected_fields}
        expr_set = {f.lower() for f in expr_fields}

        if not expected_set:
            return 0.5

        overlap = len(expected_set & expr_set)
        recall = overlap / len(expr_set) if expr_set else 0.0

        family_map: dict[str, set[str]] = {
            "price": {"close", "open", "high", "low", "vwap"},
            "volume": {"volume", "adv20"},
            "fundamental": {
                "cap",
                "sales",
                "assets",
                "equity",
                "revenue",
                "earnings",
                "sharesout",
                "liabilities",
                "debt",
            },
            "return": {"returns"},
        }

        family_bonus = 0.0
        for expr_f in expr_set:
            expr_family = None
            for family_name, members in family_map.items():
                if expr_f in members:
                    expr_family = family_name
                    break
            if expr_family is not None and any(m in expected_set for m in family_map[expr_family]):
                family_bonus += 0.1

        precision = overlap / len(expected_set) if expected_set else 0.0
        raw = 0.5 * recall + 0.3 * precision + min(family_bonus, 0.2)
        return max(0.0, min(1.0, raw))

    def _score_direction_consistency(
        self,
        expression: str,
        template: HypothesisTemplate,
        declared_direction: str = "",
    ) -> float:
        """計算方向一致性 (維度 3, 權重 25%)。

        解析表達式的隱含方向並與模板的 direction_bias 比較：
        - 最外層負號表示反向
        - ts_delta 的正負意義
        - rank 內部的遞增/遞減模式
        """
        expr_stripped = expression.strip()
        inferred_bias = self._infer_expression_direction(expr_stripped)
        target_bias = template.direction_bias

        if declared_direction:
            decl_lower = declared_direction.lower().strip()
            if decl_lower in ("short", "sell", "negative", "-1", "空頭"):
                target_bias = -abs(target_bias)
            elif decl_lower in ("long", "buy", "positive", "+1", "多頭"):
                target_bias = abs(target_bias)

        if target_bias == 0.0:
            return 1.0

        alignment = inferred_bias * target_bias
        if alignment > 0:
            return 0.8 + min(0.2, abs(alignment) * 0.2)
        elif alignment < 0:
            return max(0.1, 0.4 - abs(alignment) * 0.3)
        else:
            return 0.6

    def _infer_expression_direction(self, expression: str) -> float:
        """推斷表達式的隱含方向偏置 (+1 多頭, -1 空頭, 0 中性)。"""
        has_leading_neg = bool(re.match(r"^-\s*\w+", expression))

        neg_after_group_neutralize = False
        gn_match = re.match(r"^group_neutralize\s*\(\s*-(.+)", expression)
        if gn_match:
            neg_after_group_neutralize = True

        ts_delta_positives = len(re.findall(r"ts_delta\s*\([^,)]+\s*,\s*\d+\)", expression))
        wrapped_neg_deltas = len(re.findall(r"-\s*ts_delta\s*\(", expression))

        rank_inner_neg = bool(re.search(r"rank\s*\(\s*-", expression))

        sign_op_count = len(re.findall(r"\bsign\s*\(", expression))

        if neg_after_group_neutralize or has_leading_neg:
            return -1.0
        if wrapped_neg_deltas > ts_delta_positives or rank_inner_neg:
            return -0.7
        if sign_op_count > 0 and ts_delta_positives > 0:
            return 0.5
        if ts_delta_positives > 0 and not has_leading_neg:
            return 1.0

        inner_expr = expression
        outer_match = re.match(r"^(?:group_neutralize|rank)\s*\(\s*(.+)\s*\)$", expression, re.DOTALL)
        if outer_match:
            inner_expr = outer_match.group(1).strip()

        if inner_expr.startswith("-"):
            return -1.0

        return 0.5

    def _score_structural_fit(self, expression: str, template: HypothesisTemplate) -> float:
        """計算結構適配度 (維度 4, 權重 25%)。

        評估整體結構複雜度與假設是否匹配：
        - 巢狀深度是否合理
        - 運算子數量是否在預期範圍
        - 組成型態 (ratio/additive/multiplicative) 是否符合假設
        """
        depth = self._compute_nesting_depth(expression)
        op_count = len(self._extract_operators(expression))
        num_count = len(self._NUMBER_RE.findall(expression))
        has_ratio = "/" in expression or "div" in expression.lower()
        has_multiplication = "*" in expression

        depth_penalty = 0.0
        if depth > 6:
            depth_penalty = (depth - 6) * 0.05
        elif depth < 2:
            depth_penalty = 0.05

        op_density = op_count / max(1, num_count + 1)
        density_score = 1.0 - abs(op_density - 0.5) * 0.5

        sig_score = 0.5
        if template.structural_signature:
            try:
                if re.search(template.structural_signature, expression, re.IGNORECASE):
                    sig_score = 1.0
                else:
                    partial_patterns = template.structural_signature.split("|")
                    matches = sum(1 for p in partial_patterns if re.search(p.strip(), expression, re.IGNORECASE))
                    sig_score = 0.3 + (matches / len(partial_patterns)) * 0.5
            except re.error:
                sig_score = 0.5

        structure_type_score = 0.8
        if template.name in ("value_factor", "quality_factor") and not (has_ratio or has_multiplication):
            structure_type_score = 0.5
        elif template.name == "mean_reversion" and has_ratio:
            structure_type_score = 1.0

        raw = (density_score * 0.3 + sig_score * 0.4 + structure_type_score * 0.3) - depth_penalty
        return max(0.0, min(1.0, raw))

    def _compute_nesting_depth(self, expression: str) -> int:
        """計算表達式的括號巢狀深度。"""
        max_depth = 0
        current_depth = 0
        for ch in expression:
            if ch == "(":
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif ch == ")":
                current_depth -= 1
        return max_depth

    def _classify_alignment_level(self, r2_score: float) -> str:
        """根據 R² 分數分類對齊等級。"""
        if r2_score >= 0.75:
            return "strong"
        elif r2_score >= 0.50:
            return "moderate"
        elif r2_score >= 0.30:
            return "weak"
        else:
            return "contradictory"

    def _check_categorical_consistency(
        self,
        template_key: str,
        expr_ops: list[str],
        expr_fields: list[str],
        expression: str,
    ) -> dict[str, Any]:
        op_set = {o.lower() for o in expr_ops}
        field_set = {f.lower() for f in expr_fields}
        depth = self._compute_nesting_depth(expression)

        conflict_type: str | None = None
        confidence = 1.0
        issues: list[str] = []

        has_price = bool(field_set & self._PRICE_FIELDS)
        bool(field_set & self._VOLUME_FIELDS)
        has_fundamental_field = bool(field_set & self._FUNDAMENTAL_FIELDS)
        has_technical_op = bool(op_set & self._TECHNICAL_OPS)
        has_fundamental_op = bool(op_set & self._FUNDAMENTAL_OPS)

        is_momentum = "momentum" in template_key
        is_value = "value" in template_key
        is_quality = "quality" in template_key
        is_earnings_mom = "earnings" in template_key
        is_lowvol = "low_volatility" in template_key

        if is_momentum and has_fundamental_field and not has_price:
            conflict_type = "field_mismatch"
            confidence -= 0.35
            issues.append("動量策略通常使用價格/成交量欄位，但表達式使用了基本面欄位")

        if (is_value or is_quality) and not has_fundamental_field and has_price:
            conflict_type = "field_mismatch" if conflict_type is None else conflict_type
            confidence -= 0.35
            issues.append("價值/品質因子需要基本面欄位，但表達式只有價格欄位")

        if is_earnings_mom and not has_fundamental_field:
            conflict_type = "field_mismatch" if conflict_type is None else conflict_type
            confidence -= 0.35
            issues.append("盈利動量策略應使用盈利相關欄位（earnings/revenue/sales），未偵測到")

        if (is_value or is_quality) and has_technical_op and not has_fundamental_op:
            conflict_type = "operator_mismatch" if conflict_type is None else conflict_type
            confidence -= 0.30
            issues.append("價值/品質因子不應依賴技術面運算子，應使用 rank/zscore/scale 等橫截面運算子")

        if is_lowvol and "ts_delta" in op_set:
            conflict_type = "operator_mismatch" if conflict_type is None else conflict_type
            confidence -= 0.30
            issues.append("低波動策略不應使用 ts_delta（趨勢追蹤運算子），應使用 ts_std_dev/ts_av_diff")

        if is_lowvol and depth > 5:
            conflict_type = "complexity_mismatch" if conflict_type is None else conflict_type
            confidence -= 0.20
            issues.append(f"低波動策略巢狀深度={depth} 過高，真正的低波動因子通常 depth ≤ 4")

        confidence = max(0.0, min(1.0, confidence))
        is_consistent = confidence >= 0.60 and conflict_type is None
        explanation = (
            f"[CATEGORICAL] template='{template_key}' → consistent={is_consistent} (confidence={confidence:.2f})"
        )
        if issues:
            explanation += f" | Issues: {'; '.join(issues)}"

        return {
            "is_consistent": is_consistent,
            "conflict_type": conflict_type,
            "confidence": round(confidence, 4),
            "explanation": explanation,
        }

    def _build_diagnosis(
        self,
        op_match: float,
        field_match: float,
        dir_consistency: float,
        structural_fit: float,
        template: HypothesisTemplate,
    ) -> str:
        """組裝人類可讀的診斷說明。"""
        parts = []

        if op_match >= 0.8:
            parts.append(f"運算子高度匹配{template.name}策略模式")
        elif op_match >= 0.5:
            parts.append(f"運算子部分匹配{template.name}策略，缺少關鍵運算子")
        else:
            parts.append(f"運算子與{template.name}策略嚴重不符")

        if field_match >= 0.7:
            parts.append("使用的欄位符合假設預期")
        elif field_match >= 0.3:
            parts.append("欄位使用部分符合，但非最佳選擇")
        else:
            parts.append("使用的欄位與假設預期差異較大")

        if dir_consistency >= 0.8:
            parts.append("表達式方向與假設一致")
        elif dir_consistency >= 0.5:
            parts.append("方向存在些微偏差")
        else:
            parts.append("警告：表達式方向可能與聲稱的假設矛盾")

        if structural_fit >= 0.7:
            parts.append("結構複雜度合理")
        else:
            parts.append("結構複雜度可能偏高或偏低")

        return "；".join(parts)

    def _build_suggestions(
        self,
        op_match: float,
        field_match: float,
        dir_consistency: float,
        structural_fit: float,
        template: HypothesisTemplate,
        expression: str,
    ) -> list[str]:
        """根據各維度分數生成改善建議列表。"""
        suggestions = []

        if op_match < 0.5:
            missing = [
                op
                for op in template.expected_operators[:3]
                if op.lower() not in {o.lower() for o in self._extract_operators(expression)}
            ]
            if missing:
                suggestions.append(f"考慮加入 {template.name} 策略的核心運算子：{', '.join(missing)}")
            if template.avoid_operators:
                found_avoid = [
                    op
                    for op in template.avoid_operators
                    if op.lower() in {o.lower() for o in self._extract_operators(expression)}
                ]
                if found_avoid:
                    suggestions.append(f"避免使用與 {template.name} 衝突的運算子：{', '.join(found_avoid)}")

        if field_match < 0.4:
            suggested_fields = template.expected_fields[:3]
            suggestions.append(f"改用更符合 {template.name} 假設的欄位：{', '.join(suggested_fields)}")

        if dir_consistency < 0.5:
            if template.direction_bias > 0:
                suggestions.append("移除表達式前導負號以符合多頭方向")
            else:
                suggestions.append("加入前導負號或使用反向運算子以符合空頭/復歸方向")

        if structural_fit < 0.5:
            depth = self._compute_nesting_depth(expression)
            if depth > 5:
                suggestions.append("簡化表達式結構，降低巢狀深度以提高可解釋性和執行效率")
            else:
                suggestions.append("增加時間序列運算子或標準化層以提升結構完整度")

        if not suggestions:
            suggestions.append("表達式與假設對齊良好，無需重大調整")

        return suggestions

    def _fallback_scoring(self, expression: str, hypothesis: str) -> dict[str, Any]:
        """當假設無法匹配任何已知模板時的多信號融合備用評分邏輯。

        融合四個信號：
        1. field_overlap — 欄位與假設名稱/關鍵詞推斷預期欄位的重疊率
        2. operator_similarity — 操作符與所有模板操作符集合的相似度
        3. structural_complexity — 表達式結構複雜度是否在合理區間
        4. direction_keyword — 假設字串中方向關鍵詞與表達式隱含方向的匹配
        """
        expr_ops = self._extract_operators(expression)
        expr_fields = self._extract_fields(expression)
        weights = self._fallback_weights

        signal_details: dict[str, float] = {}

        field_overlap_score = self._fallback_signal_field_overlap(expr_fields, hypothesis)
        signal_details["field_overlap"] = round(field_overlap_score, 4)

        operator_sim_score = self._fallback_signal_operator_similarity(expr_ops)
        signal_details["operator_similarity"] = round(operator_sim_score, 4)

        structural_score = self._fallback_signal_structural_complexity(expression)
        signal_details["structural_complexity"] = round(structural_score, 4)

        direction_kw_score = self._fallback_signal_direction_keyword(expression, hypothesis)
        signal_details["direction_keyword"] = round(direction_kw_score, 4)

        raw_r2 = (
            field_overlap_score * weights["field_overlap"]
            + operator_sim_score * weights["operator_similarity"]
            + structural_score * weights["structural_complexity"]
            + direction_kw_score * weights["direction_keyword"]
        )
        r2_score = round(max(0.0, min(1.0, raw_r2)), 4)

        log_call(
            "fallback_scoring",
            input={"expression": expression[:80], "hypothesis": hypothesis},
            output={"r2_score": r2_score},
            extra={
                "signal_details": signal_details,
                "weights": weights,
                "trigger_reason": f"unknown_template:{hypothesis}",
            },
        )

        return {
            "r2_score": r2_score,
            "alignment_level": self._classify_alignment_level(r2_score),
            "operator_match": round(operator_sim_score, 4),
            "field_match": round(field_overlap_score, 4),
            "direction_consistency": round(direction_kw_score, 4),
            "structural_fit": round(structural_score, 4),
            "diagnosis": f"未知假設 '{hypothesis}'，使用多信號融合備用評分邏輯",
            "suggestions": [f"請確認假設 '{hypothesis}' 是否為有效的策略類型"],
            "matched_template": "unknown",
            "fallback_signal_details": signal_details,
        }

    def _fallback_signal_field_overlap(self, expr_fields: list[str], hypothesis: str) -> float:
        """fallback 信號 1：欄位重疊率 — 根據假設字串推斷預期欄位族並計算重疊。"""
        if not expr_fields:
            return 0.3

        expr_set = {f.lower() for f in expr_fields}
        hypothesis_lower = hypothesis.lower()

        inferred_field_set: set[str] = set()
        if any(kw in hypothesis_lower for kw in ["price", "價格", "close", "momentum", "動量", "trend", "趨勢"]):
            inferred_field_set |= self._PRICE_FIELDS
        if any(kw in hypothesis_lower for kw in ["volume", "量", "流動", "liquidity", "flow"]):
            inferred_field_set |= self._VOLUME_FIELDS
        if any(
            kw in hypothesis_lower
            for kw in [
                "value",
                "價值",
                "quality",
                "品質",
                "fundamental",
                "基本面",
                "earnings",
                "盈利",
                "sales",
                "營收",
                "growth",
                "成長",
                "cap",
                "市值",
                "size",
            ]
        ):
            inferred_field_set |= self._FUNDAMENTAL_FIELDS

        if not inferred_field_set:
            inferred_field_set = self._PRICE_FIELDS | self._VOLUME_FIELDS

        overlap = len(expr_set & inferred_field_set)
        if not expr_set:
            return 0.3
        recall = overlap / len(expr_set)
        precision = overlap / len(inferred_field_set) if inferred_field_set else 0.0
        return max(0.0, min(1.0, 0.5 * recall + 0.5 * precision))

    def _fallback_signal_operator_similarity(self, expr_ops: list[str]) -> float:
        """fallback 信號 2：操作符相似度 — 計算表達式操作符與所有模板操作符集合的 Jaccard 相似度。"""
        if not expr_ops:
            return 0.3

        expr_set = {o.lower() for o in expr_ops}
        all_expected: set[str] = set()
        all_avoid: set[str] = set()
        for tmpl in self._templates.values():
            all_expected |= {o.lower() for o in tmpl.expected_operators}
            all_avoid |= {o.lower() for o in tmpl.avoid_operators}

        hits = len(expr_set & all_expected)
        penalties = len(expr_set & all_avoid)
        base = hits / len(all_expected) if all_expected else 0.3
        penalty = penalties * 0.2
        return max(0.0, min(1.0, base + 0.3 - penalty))

    def _fallback_signal_structural_complexity(self, expression: str) -> float:
        """fallback 信號 3：結構複雜度差異 — 評估表達式複雜度是否落在合理區間 [2, 6]。"""
        depth = self._compute_nesting_depth(expression)
        len(self._extract_operators(expression))

        if depth < 2:
            return 0.4 + (depth / 2) * 0.2
        elif 2 <= depth <= 5:
            return 0.85
        elif depth == 6:
            return 0.7
        else:
            return max(0.2, 0.7 - (depth - 6) * 0.1)

    def _fallback_signal_direction_keyword(self, expression: str, hypothesis: str) -> float:
        """fallback 信號 4：方向關鍵詞匹配 — 假設中的方向關鍵詞與表達式隱含方向的一致性。"""
        hypothesis_lower = hypothesis.lower()
        inferred_dir = self._infer_expression_direction(expression)

        long_kws = ["long", "多頭", "buy", "positive", "bullish", "漲", "動量", "momentum", "trend", "趨勢"]
        short_kws = [
            "short",
            "空頭",
            "sell",
            "negative",
            "bearish",
            "跌",
            "反轉",
            "reversal",
            "contrarian",
            "復歸",
            "revert",
        ]

        long_hits = sum(1 for kw in long_kws if kw in hypothesis_lower)
        short_hits = sum(1 for kw in short_kws if kw in hypothesis_lower)

        if long_hits > short_hits:
            target_bias = 1.0
        elif short_hits > long_hits:
            target_bias = -1.0
        else:
            target_bias = 0.0

        if target_bias == 0.0:
            return 0.65

        alignment = inferred_dir * target_bias
        if alignment > 0:
            return 0.75 + min(0.25, abs(alignment) * 0.25)
        elif alignment < 0:
            return max(0.15, 0.45 - abs(alignment) * 0.25)
        else:
            return 0.55

    def _calibrate_score(self, raw_score: float, template_key: str) -> float:
        """對原始 R² 分數進行跨模板歸一化校準。

        不同模板的得分分佈不同（如 temporal 模板較寬鬆，value_factor 較嚴格），
        使用模板特定的校準參數映射到 [0, 1] 區間。
        """
        calibration_params: dict[str, dict[str, float]] = {
            "momentum_long": {"offset": -0.02, "scale": 1.05},
            "momentum_short": {"offset": -0.03, "scale": 1.08},
            "mean_reversion": {"offset": -0.01, "scale": 1.03},
            "value_factor": {"offset": 0.02, "scale": 0.96},
            "quality_factor": {"offset": 0.01, "scale": 0.98},
            "low_volatility": {"offset": 0.0, "scale": 1.0},
            "volume_price": {"offset": -0.02, "scale": 1.04},
            "earnings_momentum": {"offset": -0.01, "scale": 1.02},
            "temporal": {"offset": 0.05, "scale": 0.92},
            "cross_sectional": {"offset": 0.0, "scale": 1.0},
            "industry_rotation": {"offset": -0.02, "scale": 1.04},
            "lead_lag": {"offset": -0.01, "scale": 1.02},
            "liquidity": {"offset": 0.0, "scale": 1.0},
            "sentiment": {"offset": -0.02, "scale": 1.05},
            "size": {"offset": 0.01, "scale": 0.98},
            "growth": {"offset": -0.01, "scale": 1.03},
            "unknown": {"offset": 0.0, "scale": 1.0},
        }

        params = calibration_params.get(template_key, calibration_params["unknown"])
        calibrated = (raw_score + params["offset"]) * params["scale"]
        return max(0.0, min(1.0, calibrated))

    @algo_log(label="HypothesisAligner.score_alignment")
    def score_alignment(self, expression: str, hypothesis: str, direction: str = "") -> dict[str, Any]:
        """公開的對齊評分接口，等同 align() 但提供更精簡的返回格式。

        Args:
            expression: FASTEXPR 格式的 alpha 因子表達式
            hypothesis: 假設名稱或自然語言描述
            direction: 可選方向字串

        Returns:
            與 align() 返回相同的完整結果字典
        """
        with Timer("score_alignment_computation"):
            result = self.align(expression, hypothesis, direction)
            log_call(
                "score_alignment_summary",
                input={"expression": expression[:80], "hypothesis": hypothesis},
                output={
                    "r2_score": result.get("r2_score"),
                    "calibrated_score": result.get("calibrated_score"),
                    "alignment_level": result.get("alignment_level"),
                    "fallback_triggered": result.get("fallback_triggered"),
                },
            )
            return result

    @algo_log(label="HypothesisAligner.llm_verify_alignment")
    async def llm_verify_alignment(
        self,
        expression: str,
        hypothesis: str,
        rule_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Send rule-based alignment result to LLM for semantic verification.

        Only invoked for ambiguous boundary cases where regex/keyword scoring
        is inconclusive (r2 in [0.40, 0.70] or alignment_level == "weak").

        Args:
            expression: FASTEXPR alpha factor expression
            hypothesis: Claimed economic hypothesis name or description
            rule_result: Existing align() result dict to verify

        Returns:
            Dict with keys:
            - llm_verified: bool
            - llm_adjustment: float (clamped to [-0.15, +0.15])
            - llm_comment: str
            - adjusted_r2_score: float
            - adjusted_alignment_level: str
        """
        r2_score = rule_result.get("r2_score", 0.0)
        rule_result.get("alignment_level", "unknown")
        rule_result.get("diagnosis", "")

        logger.info(
            "[HYP-ALIGN-LLM] Sending verification request for expr=%s hyp=%s r2=%.3f",
            expression[:60],
            hypothesis,
            r2_score,
        )

        raise NotImplementedError(
            "llm_verify_alignment requires an LLM callback. Use align_with_llm(llm_fn=...) instead."
        )

    @algo_log(label="HypothesisAligner.align_with_llm")
    async def align_with_llm(
        self,
        expression: str,
        hypothesis: str,
        direction: str = "",
        llm_fn: Any = None,
    ) -> dict[str, Any]:
        eid = None
        try:
            eid = await self._tel.record_enter(
                "HypothesisAligner", cycle_id="unknown", expr_id=hash(expression) % 10000
            )
            t0 = time.perf_counter()
            llm_verified = False

            with Timer("align_with_llm"):
                rule_result = self.align(expression, hypothesis, direction)

                base: dict[str, Any] = {
                    **rule_result,
                    "llm_verified": False,
                    "llm_adjustment": 0.0,
                    "llm_comment": "",
                }

                if llm_fn is None:
                    logger.debug("[HYP-ALIGN-LLM] No llm_fn provided, returning rule-only result")
                    ms = (time.perf_counter() - t0) * 1000
                    with contextlib.suppress(OSError, ValueError, RuntimeError):
                        await self._tel.record_exit(
                            "HypothesisAligner",
                            eid,
                            metrics={"llm_verified": llm_verified, "adjustment": 0.0},
                            duration_ms=ms,
                        )
                    return base

                r2 = rule_result.get("r2_score", 0.0)
                level = rule_result.get("alignment_level", "")

                in_boundary = 0.40 <= r2 <= 0.70
                is_weak = level == "weak"

                if not (in_boundary or is_weak):
                    logger.info(
                        "[HYP-ALIGN-LLM] Skipping LLM: r2=%.3f level=%s (outside boundary [0.40, 0.70] and not weak)",
                        r2,
                        level,
                    )
                    ms = (time.perf_counter() - t0) * 1000
                    with contextlib.suppress(OSError, ValueError, RuntimeError):
                        await self._tel.record_exit(
                            "HypothesisAligner",
                            eid,
                            metrics={"llm_verified": llm_verified, "adjustment": 0.0},
                            duration_ms=ms,
                        )
                    return base

            logger.info(
                "[HYP-ALIGN-LLM] Boundary zone detected: r2=%.3f level=%s → invoking LLM",
                r2,
                level,
            )

            r2_score = rule_result.get("r2_score", 0.0)
            diag = rule_result.get("diagnosis", "")
            lvl = rule_result.get("alignment_level", "unknown")

            prompt = (
                "Verify this alpha factor alignment assessment.\n\n"
                f"Expression: {expression}\n"
                f"Claimed hypothesis: {hypothesis}\n"
                f"Rule-based R² score: {r2_score:.4f} ({lvl})\n"
                f"Rule-based diagnosis: {diag}\n\n"
                "Does the expression TRULY implement the claimed economic hypothesis?\n"
                "Consider: operator semantics, field relevance, signal direction, structural appropriateness.\n\n"
                "Reply format:\n"
                "AGREE: <yes/partial/no>\n"
                "ADJUSTMENT: <-0.15 to +0.15>\n"
                "COMMENT: <one sentence>"
            )

            try:
                raw_response = await asyncio.wait_for(
                    llm_fn(prompt),
                    timeout=10.0,
                )
            except TimeoutError:
                logger.warning("[HYP-ALIGN-LLM] LLM call timed out after 10s, using rule-only result")
                return base
            except (aiohttp.ClientError, ValueError, json.JSONDecodeError):
                return base

            response_text = str(raw_response).strip() if raw_response is not None else ""

            parsed = self._parse_llm_verification_response(response_text)

            adjustment = float(parsed.get("adjustment", 0.0))
            adjustment = max(-0.15, min(0.15, adjustment))
            agree = parsed.get("agree", "").lower()
            comment = parsed.get("comment", "")

            new_r2 = round(max(0.0, min(1.0, r2 + adjustment)), 4)
            old_level = level

            if agree == "no":
                override_level = "contradictory" if new_r2 < 0.30 else "weak"
            elif agree == "yes":
                override_level = self._classify_alignment_level(new_r2)
            else:
                override_level = self._classify_alignment_level(new_r2)

            logger.info(
                "[HYP-ALIGN-LLM] Verification complete: agree=%s adj=%.3f r2: %.4f→%.4f level: %s→%s | %s",
                agree,
                adjustment,
                r2,
                new_r2,
                old_level,
                override_level,
                comment,
            )

            result = {
                **base,
                "r2_score": new_r2,
                "alignment_level": override_level,
                "llm_verified": True,
                "llm_adjustment": round(adjustment, 4),
                "llm_comment": comment,
            }
            llm_verified = True
            ms = (time.perf_counter() - t0) * 1000
            with contextlib.suppress(OSError, ValueError, RuntimeError):
                await self._tel.record_exit(
                    "HypothesisAligner",
                    eid,
                    metrics={"llm_verified": llm_verified, "adjustment": round(adjustment, 4)},
                    duration_ms=ms,
                )
            return result
        except Exception as e:
            if eid:
                with contextlib.suppress(OSError, ValueError, RuntimeError):
                    await self._tel.record_error("HypothesisAligner", str(e), type(e).__name__)
            raise

    def _parse_llm_verification_response(self, response_text: str) -> dict[str, str]:
        """Parse structured LLM verification response into typed fields.

        Handles minor formatting variations (extra spaces, missing fields).
        """
        result: dict[str, str] = {
            "agree": "",
            "adjustment": "0.0",
            "comment": "",
        }

        agree_match = re.search(r"AGREE[:\s]+(yes|partial|no)", response_text, re.IGNORECASE)
        if agree_match:
            result["agree"] = agree_match.group(1).lower()

        adj_match = re.search(r"ADJUSTMENT[:\s]*([+-]?[\d.]+)", response_text, re.IGNORECASE)
        if adj_match:
            result["adjustment"] = adj_match.group(1)

        comment_match = re.search(r"COMMENT[:\s]+(.+)", response_text, re.IGNORECASE | re.DOTALL)
        if comment_match:
            result["comment"] = comment_match.group(1).strip()

        return result


# ========== 內嵌測試用例 ==========

if __name__ == "__main__":
    aligner = HypothesisAligner()

    # 測試案例 1：動量多頭表達式 — 應得高分
    expr1 = "group_neutralize(rank(ts_delta(close, 5)), industry)"
    r1 = aligner.align(expr1, "momentum_long")
    assert r1["r2_score"] >= 0.50, f"Test 1 failed: momentum_long score={r1['r2_score']} < 0.50"
    assert r1["alignment_level"] in ("strong", "moderate", "weak"), f"Test 1 failed: level={r1['alignment_level']}"
    print(f"[PASS] Test 1 — 動量多頭: R²={r1['r2_score']:.3f}, level={r1['alignment_level']}")
    print(f"       {r1['diagnosis']}")

    # 測試案例 2：均值復歸表達式 — 應得分數合理
    expr2 = "group_neutralize(rank(-ts_zscore(close, 20)), industry)"
    r2 = aligner.align(expr2, "mean_reversion")
    assert r2["r2_score"] >= 0.40, f"Test 2 failed: mean_reversion score={r2['r2_score']} < 0.40"
    print(f"[PASS] Test 2 — 均值復歸: R²={r2['r2_score']:.3f}, level={r2['alignment_level']}")
    print(f"       {r2['diagnosis']}")

    # 測試案例 3：方向矛盾 — 聲稱動量但表達式是均值復歸結構
    expr3 = "group_neutralize(rank(ts_mean(close, 20) - close), industry)"
    r3 = aligner.align(expr3, "momentum_long")
    assert r3["r2_score"] <= 0.65, f"Test 3 failed: contradictory should be low, got {r3['r2_score']}"
    print(f"[PASS] Test 3 — 方向矛盾: R²={r3['r2_score']:.3f}, level={r3['alignment_level']}")
    print(f"       {r3['diagnosis']}")

    # 測試案例 4：反向偵測
    detected = aligner.detect_hypothesis_from_expression(expr1)
    print(f"[PASS] Test 4 — 反向偵測: '{expr1[:50]}...' → {detected}")
    assert detected == "industry_rotation", (
        f"Test 4 failed: expression with group_neutralize(..., industry) should be industry_rotation, detected={detected}"  # noqa: E501
    )

    # 測試案例 5：反饋文字生成
    feedback = aligner.build_alignment_feedback(r3)
    assert "HYPOTHESIS ALIGNMENT CHECK" in feedback, "Test 5 failed: missing header"
    assert "R²=" in feedback, "Test 5 failed: missing r2_score"
    print(f"[PASS] Test 5 — 反饋文字:\n{feedback}")

    # 測試案例 6：價值因子
    expr6 = "group_neutralize(rank(log(cap / sales)), industry)"
    r6 = aligner.align(expr6, "value_factor")
    assert r6["field_match"] > 0.3, f"Test 6 failed: value factor field match too low: {r6['field_match']}"
    print(f"[PASS] Test 6 — 價值因子: R²={r6['r2_score']:.3f}, field_match={r6['field_match']:.3f}")

    print("\n=== 所有測試通過 ===")
