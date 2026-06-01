from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from alpha_research_pipeline import DEFAULT_DATASETS_DIR, DEFAULT_FIELDS_SUMMARY, DEFAULT_IDEA_LIBRARY
from worldquant_brain_cli import API_BASE, DEFAULT_MAX_WAIT, DEFAULT_POLL_INTERVAL, DEFAULT_TIMEOUT


@dataclass(frozen=True)
class AuthConfig:
    email: Optional[str]
    password: Optional[str]
    cookie_header: Optional[str]
    base_url: str = API_BASE
    timeout: float = DEFAULT_TIMEOUT


@dataclass(frozen=True)
class ModelConfig:
    provider: str = "heuristic"
    model: str = "gpt5.5"
    temperature: float = 0.1
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    timeout: float = 60.0


@dataclass(frozen=True)
class GridConfig:
    enabled: bool = False
    grid_budget: int = 5
    grid_domain_family: str = ""
    grid_diversity_threshold: float = 0.15
    grid_max_repair_iterations: int = 5
    grid_sharpe_threshold: float = 0.5
    grid_delta_threshold: float = 0.05
    grid_populate_from_library: bool = True


@dataclass(frozen=True)
class RAGConfig:
    enabled: bool = False
    rag_top_k: int = 5
    rag_max_features: int = 2000


@dataclass(frozen=True)
class LLMGenConfig:
    enabled: bool = False
    provider: str = "heuristic"
    diversity_threshold: float = 0.15
    cot_enabled: bool = True
    cot_temperature: float = 0.7
    rewrite_enabled: bool = True
    rewrite_temperature: float = 0.3


@dataclass(frozen=True)
class AgentConfig:
    budget: int = 24
    max_iterations: int = 12
    seed_fraction: float = 0.7
    refine_top_k: int = 8
    robustness_top_k: int = 3
    robustness_score_threshold: float = 500.0
    family_filter: tuple[str, ...] = ()
    shuffle_seeds: bool = True
    random_seed: int = 7
    max_family_budget_share: float = 0.45
    min_expression_novelty: float = 0.10
    retries: int = 2
    sleep_between: float = 1.0
    max_wait: float = DEFAULT_MAX_WAIT
    poll_interval: float = DEFAULT_POLL_INTERVAL
    allow_pending_checks: bool = False
    submission_mode: str = "disabled"
    workdir: Path = Path(".alpha_agent")
    idea_library: Path = DEFAULT_IDEA_LIBRARY
    fields_summary: Path = DEFAULT_FIELDS_SUMMARY
    datasets_dir: Path = DEFAULT_DATASETS_DIR
    grid: GridConfig = GridConfig()
    rag: RAGConfig = RAGConfig()
    llm_gen: LLMGenConfig = LLMGenConfig()


@dataclass(frozen=True)
class AgentRuntimeConfig:
    auth: AuthConfig
    model: ModelConfig
    agent: AgentConfig
