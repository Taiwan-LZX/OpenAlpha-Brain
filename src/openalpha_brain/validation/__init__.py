# validation package
from openalpha_brain.validation.ast_validator import (
    ASTValidator,
    ValidationResult,
)
from openalpha_brain.validation.wq_format_repair import (
    WQFormatRepair,
    RepairDiagnosis,
    create_wq_format_repairer,
    auto_repair_wq_expression,
)
from openalpha_brain.validation.anti_overfit_detector import (
    LightweightAntiOverfitDetector,
    FullAntiOverfitDetector,
    TestResult,
    AntiOverfitResult,
)
from openalpha_brain.validation.wq_expression_validator import (
    WQExpressionValidator,
    CheckResult as WQCheckResult,
    ValidationResult as WQValidationResult,
)
from openalpha_brain.validation.official_scorer import (
    OfficialScoringAdapter,
    ScoreReport,
    CheckItem,
    quick_score,
    evaluate_alpha_quality,
)

__all__ = [
    # 原有模块
    "ASTValidator",
    "ValidationResult",
    "WQFormatRepair",
    "RepairDiagnosis",
    "create_wq_format_repairer",
    "auto_repair_wq_expression",
    "LightweightAntiOverfitDetector",
    "FullAntiOverfitDetector",
    "TestResult",
    "AntiOverfitResult",
    # 新增: WQ 表达式验证器
    "WQExpressionValidator",
    "WQCheckResult",
    "WQValidationResult",
    # 新增: 官方评分适配器
    "OfficialScoringAdapter",
    "ScoreReport",
    "CheckItem",
    "quick_score",
    "evaluate_alpha_quality",
]
