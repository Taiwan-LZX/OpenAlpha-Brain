from __future__ import annotations

CRITICAL_ALGORITHMS = {
    "mab_select", "rag_retrieve", "conversation_summarizer", "ast_repair",
    "brain_submit", "add_record", "whitelist_update", "experience_card_retrieval",
    "market_state_infer", "signal_arbiter_rerank",
}

IMPORTANT_ALGORITHMS = {
    "find_similar_by_embedding", "vector_duplicate_check", "semantic_alignment_check",
    "feature_map_add", "param_optimization", "alpha_channel_submit",
    "experience_distillation", "evidence_recording", "success_case_add",
    "pnl_stability_analysis", "garch_volatility", "overfit_detection",
    "originality_check", "complexity_control", "failure_fix_add",
    "mab_update", "logic_evolution",
}

CONDITIONAL_ALGORITHMS = {
    "multi_agent_orchestrator", "self_critique", "specialist_agent_create",
    "semantic_crossover", "gradient_mutation", "crossover_mutation_engine",
    "auto_debug_loop", "yearly_data_fetch", "correlation_fetch",
    "prod_correlation_check", "explore_unexplored_regions", "sample_inspiration",
    "find_similar_failures", "tool_search", "sample_distant_parent",
    "classify", "tool_conflict_detection", "quant_logic_review",
    "evidence_distillation", "trajectory_mutation", "semantic_space_mutation",
    "alpha_channel_batch", "mab_bias_adjustment",
}


def check_algorithm_connectivity(algo_call_counts, critical=None, important=None, conditional=None):
    critical = critical or CRITICAL_ALGORITHMS
    important = important or IMPORTANT_ALGORITHMS
    conditional = conditional or CONDITIONAL_ALGORITHMS

    critical_called = {name: algo_call_counts[name] for name in critical if name in algo_call_counts and algo_call_counts[name] > 0}
    important_called = {name: algo_call_counts[name] for name in important if name in algo_call_counts and algo_call_counts[name] > 0}
    conditional_called = {name: algo_call_counts[name] for name in conditional if name in algo_call_counts and algo_call_counts[name] > 0}

    critical_missing = sorted(name for name in critical if name not in critical_called)
    important_missing = sorted(name for name in important if name not in important_called)

    total_called = len(critical_called) + len(important_called)
    total_expected = len(critical) + len(important)
    coverage_pct = (total_called / total_expected * 100.0) if total_expected > 0 else 0.0

    return {
        "passed": len(critical_missing) == 0,
        "critical_missing": critical_missing,
        "important_missing": important_missing,
        "critical_called": critical_called,
        "important_called": important_called,
        "conditional_called": conditional_called,
        "total_algorithms_called": total_called,
        "coverage_pct": round(coverage_pct, 1),
    }


class AlgorithmConnectivityChecker:
    def __init__(self, required_critical=None, required_important=None):
        self.required_critical = required_critical or CRITICAL_ALGORITHMS
        self.required_important = required_important or IMPORTANT_ALGORITHMS
        self._snapshots = []

    def snapshot(self, label, algo_call_counts):
        report = check_algorithm_connectivity(
            algo_call_counts,
            critical=self.required_critical,
            important=self.required_important,
        )
        report["label"] = label
        self._snapshots.append(report)
        return report

    def final_check(self, algo_call_counts, min_coverage_pct=60.0):
        report = check_algorithm_connectivity(
            algo_call_counts,
            critical=self.required_critical,
            important=self.required_important,
        )
        report["min_coverage_pct"] = min_coverage_pct
        assert report["critical_missing"] == [], (
            f"Critical algorithms missing: {report['critical_missing']}"
        )
        important_coverage = (
            len(report["important_called"]) / len(self.required_important) * 100.0
            if self.required_important
            else 0.0
        )
        assert important_coverage >= min_coverage_pct, (
            f"Important algorithm coverage {important_coverage:.1f}% "
            f"below minimum {min_coverage_pct}%"
        )
        return report


def format_connectivity_report(report):
    lines = []
    lines.append("=" * 60)
    lines.append("  Algorithm Connectivity Report")
    lines.append("=" * 60)
    lines.append(f"  Passed: {report['passed']}")
    lines.append(f"  Coverage: {report['coverage_pct']:.1f}% ({report['total_algorithms_called']} algorithms)")
    lines.append("")
    lines.append(f"  Critical Called  : {', '.join(sorted(report['critical_called'].keys())) or '(none)'}")
    lines.append(f"  Critical Missing : {', '.join(report['critical_missing']) or '(none)'}")
    lines.append("")
    lines.append(f"  Important Called  : {', '.join(sorted(report['important_called'].keys())) or '(none)'}")
    lines.append(f"  Important Missing : {', '.join(report['important_missing']) or '(none)'}")
    lines.append("")
    lines.append(f"  Conditional Called: {', '.join(sorted(report['conditional_called'].keys())) or '(none)'}")
    if "min_coverage_pct" in report:
        lines.append("")
        lines.append(f"  Min Coverage Required: {report['min_coverage_pct']:.1f}%")
    if "label" in report:
        lines.append(f"  Snapshot Label: {report['label']}")
    lines.append("=" * 60)
    return "\n".join(lines)
