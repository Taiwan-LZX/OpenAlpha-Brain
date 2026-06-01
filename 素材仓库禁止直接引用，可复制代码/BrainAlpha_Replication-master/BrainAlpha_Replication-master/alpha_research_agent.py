#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

import alpha_research_pipeline as pipeline
from alpha_agent.config import (
    AgentConfig, AgentRuntimeConfig, AuthConfig, ModelConfig,
    GridConfig, RAGConfig, LLMGenConfig,
)
from alpha_agent.engine import AlphaResearchAgent, ResearchToolbox
from alpha_agent.evaluation import run_evaluation_suite
from alpha_agent.planner import HeuristicPlanner, OpenAIJsonPlanner
from local_env import load_local_dotenv
from worldquant_brain_cli import API_BASE, BrainApiError, DEFAULT_MAX_WAIT, DEFAULT_POLL_INTERVAL, DEFAULT_TIMEOUT


DEFAULT_AGENT_WORKDIR = Path(".alpha_agent")
DEFAULT_EVAL_CASES = Path("docs") / "eval_cases.json"

load_local_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Alpha Research Agent with planner-driven orchestration, tool use, and guarded submission flow."
    )
    add_auth_args(parser)
    add_common_agent_args(parser)
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full alpha research agent loop.")
    add_run_args(run_parser)
    run_parser.set_defaults(func=command_run)

    leaderboard_parser = subparsers.add_parser("leaderboard", help="Show current leaderboard from workdir.")
    leaderboard_parser.add_argument("--limit", type=int, default=10)
    leaderboard_parser.set_defaults(func=command_leaderboard)

    eval_parser = subparsers.add_parser("evaluate", help="Run baseline vs agent evaluation suite.")
    eval_parser.add_argument(
        "--cases",
        default=str(DEFAULT_EVAL_CASES),
        help="Path to evaluation JSON cases list.",
    )
    eval_parser.add_argument(
        "--output",
        default="",
        help="Optional output report path. Defaults to <workdir>/evaluation/report.json",
    )
    eval_parser.set_defaults(func=command_evaluate)

    return parser.parse_args()


def add_auth_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--email", default=os.getenv("WQB_EMAIL"), help="BRAIN account email.")
    parser.add_argument("--password", default=os.getenv("WQB_PASSWORD"), help="BRAIN account password.")
    parser.add_argument(
        "--cookie-header",
        default=os.getenv("WQB_COOKIE_HEADER"),
        help="Optional raw Cookie header. If set, email/password login is skipped.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("WQB_API_BASE", API_BASE),
        help=f"API base URL. Defaults to {API_BASE}.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("WQB_TIMEOUT", DEFAULT_TIMEOUT)),
        help="Single request timeout in seconds.",
    )


def add_common_agent_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--idea-library",
        default=str(pipeline.DEFAULT_IDEA_LIBRARY),
        help="Path to idea library JSON.",
    )
    parser.add_argument(
        "--fields-summary",
        default=str(pipeline.DEFAULT_FIELDS_SUMMARY),
        help="Path to cached fields summary JSON.",
    )
    parser.add_argument(
        "--workdir",
        default=str(DEFAULT_AGENT_WORKDIR),
        help="Agent working directory for results/state/logs.",
    )
    parser.add_argument(
        "--planner-provider",
        choices=("heuristic", "openai"),
        default=os.getenv("ALPHA_AGENT_PLANNER_PROVIDER", "heuristic"),
        help="Planning backend.",
    )
    parser.add_argument(
        "--planner-model",
        default=os.getenv("ALPHA_AGENT_PLANNER_MODEL", "gpt5.5"),
        help="Model name for planner provider.",
    )
    parser.add_argument(
        "--planner-temperature",
        type=float,
        default=float(os.getenv("ALPHA_AGENT_PLANNER_TEMPERATURE", "0.1")),
        help="Planner sampling temperature.",
    )
    parser.add_argument(
        "--planner-base-url",
        default=os.getenv("ALPHA_AGENT_PLANNER_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI-compatible base URL for planner provider.",
    )
    parser.add_argument(
        "--planner-api-key-env",
        default=os.getenv("ALPHA_AGENT_PLANNER_API_KEY_ENV", "OPENAI_API_KEY"),
        help="Environment variable name that stores planner API key.",
    )


def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--budget", type=int, default=int(os.getenv("ALPHA_AGENT_BUDGET", "24")))
    parser.add_argument("--max-iterations", type=int, default=int(os.getenv("ALPHA_AGENT_MAX_ITERATIONS", "12")))
    parser.add_argument("--seed-fraction", type=float, default=float(os.getenv("ALPHA_AGENT_SEED_FRACTION", "0.7")))
    parser.add_argument("--refine-top-k", type=int, default=int(os.getenv("ALPHA_AGENT_REFINE_TOP_K", "8")))
    parser.add_argument(
        "--robustness-top-k",
        type=int,
        default=int(os.getenv("ALPHA_AGENT_ROBUSTNESS_TOP_K", "3")),
    )
    parser.add_argument(
        "--robustness-score-threshold",
        type=float,
        default=float(os.getenv("ALPHA_AGENT_ROBUSTNESS_SCORE_THRESHOLD", "500")),
    )
    parser.add_argument("--family", action="append", default=[], help="Optional family filter (repeatable).")
    parser.add_argument(
        "--max-family-budget-share",
        type=float,
        default=float(os.getenv("ALPHA_AGENT_MAX_FAMILY_BUDGET_SHARE", "0.45")),
    )
    parser.add_argument(
        "--min-expression-novelty",
        type=float,
        default=float(os.getenv("ALPHA_AGENT_MIN_EXPRESSION_NOVELTY", "0.10")),
    )
    parser.add_argument("--shuffle-seeds", action="store_true", default=True, help="Shuffle generated seeds.")
    parser.add_argument("--no-shuffle-seeds", action="store_false", dest="shuffle_seeds")
    parser.add_argument("--random-seed", type=int, default=int(os.getenv("ALPHA_AGENT_RANDOM_SEED", "7")))
    parser.add_argument("--retries", type=int, default=int(os.getenv("ALPHA_AGENT_RETRIES", "2")))
    parser.add_argument("--sleep-between", type=float, default=float(os.getenv("ALPHA_AGENT_SLEEP_BETWEEN", "1.0")))
    parser.add_argument("--max-wait", type=float, default=float(os.getenv("WQB_MAX_WAIT", str(DEFAULT_MAX_WAIT))))
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=float(os.getenv("WQB_POLL_INTERVAL", str(DEFAULT_POLL_INTERVAL))),
    )
    parser.add_argument("--allow-pending-checks", action="store_true")
    parser.add_argument(
        "--submission-mode",
        choices=("disabled", "manual", "auto_approved"),
        default="disabled",
        help="Submission governance mode.",
    )
    parser.add_argument(
        "--interactive-approval",
        action="store_true",
        help="Prompt approval in terminal when submission-mode=manual.",
    )
    # ── Grid pipeline flags ──
    parser.add_argument("--grid-enabled", action="store_true", help="Enable BrainAlpha structured exploration grid.")
    parser.add_argument("--grid-budget", type=int, default=5, help="Number of grid cells to explore per iteration.")
    parser.add_argument("--grid-domain-family", default="", help="Focus grid exploration on a signal family.")
    parser.add_argument("--grid-diversity-threshold", type=float, default=0.15)
    parser.add_argument("--grid-max-repair-iterations", type=int, default=5)
    parser.add_argument("--grid-sharpe-threshold", type=float, default=0.5)
    parser.add_argument("--grid-delta-threshold", type=float, default=0.05)
    parser.add_argument("--grid-populate-library", action="store_true", default=None, help="Populate grid from library (overrides --grid-no-populate-library).")
    parser.add_argument("--grid-no-populate-library", action="store_true", help="Skip populating grid from library.")
    # ── RAG flags ──
    parser.add_argument("--rag-enabled", action="store_true", help="Enable TF-IDF RAG field encoder.")
    parser.add_argument("--rag-top-k", type=int, default=5)
    parser.add_argument("--rag-max-features", type=int, default=2000)
    # ── LLM Generation flags ──
    parser.add_argument("--llm-gen-enabled", action="store_true", default=os.getenv("ALPHA_AGENT_LLM_GEN_ENABLED", "false").lower() == "true", help="Enable LLM CoT expression generation (default from ALPHA_AGENT_LLM_GEN_ENABLED env).")
    parser.add_argument("--no-llm-gen-enabled", action="store_false", dest="llm_gen_enabled", help="Disable LLM CoT expression generation.")
    parser.add_argument("--llm-gen-diversity-threshold", type=float, default=0.15)
    parser.add_argument("--llm-gen-cot-enabled", action="store_true", default=True)


def command_run(args: argparse.Namespace) -> Dict[str, Any]:
    runtime = build_runtime(args)
    planner = build_planner(runtime.model)
    approval_callback = None
    if args.submission_mode == "manual" and args.interactive_approval:
        approval_callback = cli_manual_approval

    def _cli_progress(payload: Dict[str, Any]) -> None:
        t = payload.get("type", "")
        if t == "run_started":
            print(f"[Agent] Started  budget={payload.get('budget')}  max_iter={payload.get('max_iterations')}", flush=True)
        elif t == "planner_decision":
            print(f"[Agent] Iter {payload.get('iteration')}  stage={payload.get('stage')}  action={payload.get('action')}  batch={payload.get('batch_size')}", flush=True)
        elif t == "batch_started":
            print(f"[Agent]   Evaluating {payload.get('batch_size')} candidate(s)...", flush=True)
            for c in (payload.get("candidates") or []):
                expr = c.get("expression", "?")
                print(f"[Agent]     expression={expr}", flush=True)
        elif t == "candidate_started":
            print(f"[Agent]   [{payload.get('candidate_index')}/{payload.get('candidate_count')}] Submitting to Brain...", flush=True)
        elif t == "candidate_completed":
            r = payload.get("record", {})
            status = r.get("status") or r.get("alpha_status", "?")
            sharpe = r.get("sharpe")
            if sharpe is None:
                metrics = r.get("metrics", {})
                sharpe = metrics.get("sharpe", "?")
            print(f"[Agent]   [{payload.get('candidate_index')}/{payload.get('candidate_count')}] status={status}  sharpe={sharpe}", flush=True)
            if status == "error":
                err = r.get("error", {})
                print(f"[Agent]     error={err.get('message', '?')}", flush=True)
        elif t == "batch_completed":
            print(f"[Agent]   Batch done: {payload.get('details', {}).get('executed_batch')} evaluated", flush=True)

    agent = AlphaResearchAgent(
        runtime, planner=planner,
        approval_callback=approval_callback,
        progress_callback=_cli_progress,
    )
    result = agent.run()
    s = result.summary
    print(f"[Agent] Done: {s.get('evaluated_total', '?')} evaluated, best_sharpe={s.get('best_sharpe', '?')}", flush=True)
    return result.to_dict()


def command_leaderboard(args: argparse.Namespace) -> Dict[str, Any]:
    runtime = build_runtime(args, run_overrides={})
    toolbox = ResearchToolbox(runtime)
    leaderboard = [pipeline.compact_record(item) for item in toolbox.leaderboard(limit=args.limit)]
    return {
        "count": len(leaderboard),
        "workdir": str(runtime.agent.workdir),
        "leaderboard": leaderboard,
    }


def command_evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    runtime = build_runtime(args, run_overrides={})
    case_file = Path(args.cases)
    output_path = Path(args.output) if args.output else Path(runtime.agent.workdir) / "evaluation" / "report.json"
    return run_evaluation_suite(
        runtime=runtime,
        case_file=case_file,
        output_path=output_path,
    )


def _resolve_grid_populate(args: argparse.Namespace) -> bool:
    populate = getattr(args, "grid_populate_library", None)
    if populate is not None:
        return populate
    return not getattr(args, "grid_no_populate_library", False)


def build_runtime(args: argparse.Namespace, run_overrides: Optional[Dict[str, Any]] = None) -> AgentRuntimeConfig:
    auth = AuthConfig(
        email=args.email,
        password=args.password,
        cookie_header=args.cookie_header,
        base_url=args.base_url,
        timeout=args.timeout,
    )
    model = ModelConfig(
        provider=args.planner_provider,
        model=args.planner_model,
        temperature=args.planner_temperature,
        base_url=args.planner_base_url,
        api_key_env=args.planner_api_key_env,
    )
    grid = GridConfig(
        enabled=getattr(args, "grid_enabled", False),
        grid_budget=getattr(args, "grid_budget", 5),
        grid_domain_family=getattr(args, "grid_domain_family", ""),
        grid_diversity_threshold=getattr(args, "grid_diversity_threshold", 0.15),
        grid_max_repair_iterations=getattr(args, "grid_max_repair_iterations", 5),
        grid_sharpe_threshold=getattr(args, "grid_sharpe_threshold", 0.5),
        grid_delta_threshold=getattr(args, "grid_delta_threshold", 0.05),
        grid_populate_from_library=_resolve_grid_populate(args),
    )
    rag = RAGConfig(
        enabled=getattr(args, "rag_enabled", False),
        rag_top_k=getattr(args, "rag_top_k", 5),
        rag_max_features=getattr(args, "rag_max_features", 2000),
    )
    llm_gen = LLMGenConfig(
        enabled=getattr(args, "llm_gen_enabled", False),
        diversity_threshold=getattr(args, "llm_gen_diversity_threshold", 0.15),
        cot_enabled=getattr(args, "llm_gen_cot_enabled", True),
    )
    base_agent = AgentConfig(
        budget=getattr(args, "budget", 24),
        max_iterations=getattr(args, "max_iterations", 12),
        seed_fraction=getattr(args, "seed_fraction", 0.7),
        refine_top_k=getattr(args, "refine_top_k", 8),
        robustness_top_k=getattr(args, "robustness_top_k", 3),
        robustness_score_threshold=getattr(args, "robustness_score_threshold", 500.0),
        family_filter=tuple(getattr(args, "family", []) or ()),
        max_family_budget_share=getattr(args, "max_family_budget_share", 0.45),
        min_expression_novelty=getattr(args, "min_expression_novelty", 0.10),
        shuffle_seeds=getattr(args, "shuffle_seeds", True),
        random_seed=getattr(args, "random_seed", 7),
        retries=getattr(args, "retries", 2),
        sleep_between=getattr(args, "sleep_between", 1.0),
        max_wait=getattr(args, "max_wait", DEFAULT_MAX_WAIT),
        poll_interval=getattr(args, "poll_interval", DEFAULT_POLL_INTERVAL),
        allow_pending_checks=getattr(args, "allow_pending_checks", False),
        submission_mode=getattr(args, "submission_mode", "disabled"),
        workdir=Path(args.workdir),
        idea_library=Path(args.idea_library),
        fields_summary=Path(args.fields_summary),
        grid=grid,
        rag=rag,
        llm_gen=llm_gen,
    )
    if run_overrides:
        base_agent = replace(base_agent, **run_overrides)
    return AgentRuntimeConfig(auth=auth, model=model, agent=base_agent)


def build_planner(model_config: ModelConfig) -> Any:
    if model_config.provider == "openai":
        return OpenAIJsonPlanner(model_config=model_config)
    return HeuristicPlanner()


def cli_manual_approval(candidate: Dict[str, Any]) -> bool:
    alpha_id = candidate.get("alpha_id")
    score = candidate.get("score")
    family = candidate.get("family")
    prompt = f"Approve submit for alpha {alpha_id} (family={family}, score={score})? [y/N]: "
    answer = input(prompt).strip().lower()
    return answer in {"y", "yes"}


def render_output(payload: Any, *, pretty: bool) -> None:
    if pretty:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False))


def main() -> int:
    args = parse_args()
    try:
        payload = args.func(args)
        render_output(payload, pretty=args.pretty)
        return 0
    except BrainApiError as exc:
        message = str(exc)
        if exc.status is not None:
            message = f"{message} [status={exc.status}]"
        if exc.url:
            message = f"{message} [url={exc.url}]"
        print(message, file=sys.stderr)
        if exc.payload is not None:
            print(json.dumps(exc.payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
