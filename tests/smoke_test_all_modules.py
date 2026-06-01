"""
全局冒烟测试 — 验证所有新模块能正常导入和基本工作

Round 3: Smoke Test All Modules
目标：确保所有核心模块可导入、可初始化、基本功能正常
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_all_imports():
    """所有模块都能成功 import"""
    from openalpha_brain.knowledge.operator_registry import (
        get_operator_registry, OperatorRegistry, OperatorDef, OperatorCategory
    )
    from openalpha_brain.knowledge.field_proxy_map import (
        get_field_proxy_map, FieldProxyMap
    )
    from openalpha_brain.generation.template_reasoning_generator import (
        TemplateReasoningGenerator, ReasoningResult
    )
    from openalpha_brain.generation.alpha_logics import (
        AlphaLogicLibrary, ThreeBlockTemplate
    )
    from openalpha_brain.validation.wq_format_repair import (
        WQFormatRepair, RepairDiagnosis, auto_repair_wq_expression
    )
    from openalpha_brain.validation.anti_overfit_detector import (
        LightweightAntiOverfitDetector,
        FullAntiOverfitDetector,
        AntiOverfitResult,
    )
    from openalpha_brain.validation.wq_expression_validator import (
        WQExpressionValidator, ValidationResult, CheckResult
    )
    from openalpha_brain.validation.official_scorer import (
        OfficialScoringAdapter, ScoreReport, quick_score, evaluate_alpha_quality
    )
    from openalpha_brain.evolution.mutation_engine import (
        BrainAwareMutationEngine, MutationStrategy, Diagnosis
    )
    from openalpha_brain.core.feedback_orchestrator import (
        FeedbackLoopOrchestrator, DecisionAction, GeneratedAlpha, CycleResult
    )
    from openalpha_brain.services.slot_manager import (
        SlotManager, PriorityTier, SlotState, SubmissionTask
    )


def test_operator_registry():
    """OperatorRegistry 初始化 + 基本查询"""
    from openalpha_brain.knowledge.operator_registry import get_operator_registry

    reg = get_operator_registry()
    assert reg is not None
    stats = reg.get_stats()
    assert stats["total_operators"] >= 60, f"Expected >= 60 operators, got {stats['total_operators']}"

    ts_ops = reg.get_temporal_operators()
    assert len(ts_ops) >= 15, f"Expected >= 15 temporal operators, got {len(ts_ops)}"

    low_risk = reg.get_low_risk_operators()
    assert len(low_risk) >= 5, f"Expected >= 5 low-risk operators, got {len(low_risk)}"


def test_field_proxy_map():
    """FieldProxyMap 初始化"""
    from openalpha_brain.knowledge.field_proxy_map import get_field_proxy_map

    fpm = get_field_proxy_map()
    assert fpm is not None


def test_expression_validator():
    """WQExpressionValidator 基本验证"""
    from openalpha_brain.validation.wq_expression_validator import WQExpressionValidator

    v = WQExpressionValidator()

    expr = (
        "group_neutralize("
        "  rank(ts_decay_linear(ts_delta(signed_power(-close / debt, 2), 5), 10)),"
        "  industry"
        ")"
    )
    r1 = v.validate_full(expr)
    assert r1.passed is True, f"Valid expression should pass: {r1.errors}"

    r2 = v.validate_full("bad_func(xyz)")
    assert r2.passed is False, "Invalid expression should fail"


def test_official_scorer():
    """OfficialScorer 基本评分"""
    from openalpha_brain.validation.official_scorer import OfficialScoringAdapter

    s = OfficialScoringAdapter()
    r = s.compute_score({
        "sharpe": 1.5,
        "fitness": 1.2,
        "turnover": 0.25,
        "drawdown": 0.10,
    })
    assert r.overall_score > 50, f"Good factor should score > 50, got {r.overall_score}"
    assert r.grade in ["A+", "A", "A-", "B+"], f"Unexpected grade: {r.grade}"


def test_format_repair():
    """WQFormatRepair 基本修复"""
    from openalpha_brain.validation.wq_format_repair import WQFormatRepair

    r = WQFormatRepair()
    d = r.diagnose("lookback required", "rank(ts_mean(close))")
    assert d.error_type == "lookback", f"Expected 'lookback', got {d.error_type}"

    repaired = r.repair(d)
    assert "20" in repaired or "10" in repaired, f"Should insert default window, got: {repaired}"


def test_mutation_engine():
    """MutationEngine 基本诊断"""
    from openalpha_brain.evolution.mutation_engine import (
        BrainAwareMutationEngine, MutationStrategy
    )

    e = BrainAwareMutationEngine()
    d = e.diagnose("rank(ts_delta(close, 20))", {"sharpe": -0.5}, [])
    assert isinstance(d.strategy, MutationStrategy), \
        f"Expected MutationStrategy, got {type(d.strategy)}"

    prompt = e.generate_mutation_prompt(d, "rank(ts_delta(close, 20))")
    assert len(prompt) > 50, f"Prompt should have content, got length {len(prompt)}"


def test_anti_overfit():
    """AntiOverfitDetector 基本检测"""
    from openalpha_brain.validation.anti_overfit_detector import LightweightAntiOverfitDetector

    d = LightweightAntiOverfitDetector()
    r = d.evaluate({"sharpe": 1.5, "fitness": 1.2, "turnover": 0.25})
    assert r.score >= 60, f"Good metrics should score >= 60, got {r.score}"

    r2 = d.evaluate({
        "sharpe": -2.0,
        "fitness": -1.5,
        "turnover": 0.95,
        "drawdown": -0.80,
        "checks": [
            {"name": "sharpe_positive", "result": False},
            {"name": "turnover_platform", "result": False},
            {"name": "fitness_efficient", "result": False},
        ]
    })
    assert r2.score < 60, f"Bad metrics should score < 60, got {r2.score}"


def test_alpha_logic_library():
    """AlphaLogicLibrary + ThreeBlockTemplate"""
    from openalpha_brain.generation.alpha_logics import AlphaLogicLibrary

    lib = AlphaLogicLibrary()
    logics = lib.get_top_logics(n=5)
    assert len(logics) >= 3, f"Expected >= 3 logics, got {len(logics)}"

    t = lib.get_three_block_template("value_regression")
    if t is None:
        t = lib.get_three_block_template(list(lib._templates.keys())[0])
    assert t is not None, "Should have at least one template"

    assembled = t.assemble(price_field="close", fundamental_field="debt", decay_lb=10)
    assert "group_neutralize" in assembled, "Assembled expression should contain group_neutralize"
    assert "debt" in assembled, "Assembled expression should contain debt"


def test_integration_op_reg_in_validator():
    """集成测试：OperatorRegistry 在 Validator 中使用"""
    from openalpha_brain.knowledge.operator_registry import get_operator_registry
    from openalpha_brain.validation.wq_expression_validator import WQExpressionValidator

    reg = get_operator_registry()
    op_names = set(reg._operators.keys())
    v = WQExpressionValidator(operator_registry=op_names)

    expr = (
        "group_neutralize("
        "  rank(ts_decay_linear(ts_delta(signed_power(-close / debt, 2), 5), 10)),"
        "  industry"
        ")"
    )
    result = v.validate_full(expr)
    assert result.passed is True, f"Integration test should pass: {result.errors}"


def run_all_smoke_tests():
    """运行所有冒烟测试"""
    tests = [
        ("imports", test_all_imports),
        ("operator_registry", test_operator_registry),
        ("field_proxy_map", test_field_proxy_map),
        ("expression_validator", test_expression_validator),
        ("official_scorer", test_official_scorer),
        ("format_repair", test_format_repair),
        ("mutation_engine", test_mutation_engine),
        ("anti_overfit", test_anti_overfit),
        ("alpha_logic_lib", test_alpha_logic_library),
        ("integration", test_integration_op_reg_in_validator),
    ]

    passed = 0
    failed = 0
    errors = []

    print("=" * 60)
    print("🔥 SMOKE TEST ALL MODULES — Round 3")
    print("=" * 60)
    print()

    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  ✅ {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            failed += 1
            errors.append((name, str(e)))

    print()
    print("=" * 60)
    print(f"SMOKE TEST RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    if failed > 0:
        print("\n⚠️  SOME TESTS FAILED — needs investigation:")
        for name, err in errors:
            print(f"  • {name}: {err}")
        return False
    else:
        print("\n🎉 ALL SMOKE TESTS PASSED")
        return True


if __name__ == "__main__":
    success = run_all_smoke_tests()
    sys.exit(0 if success else 1)
