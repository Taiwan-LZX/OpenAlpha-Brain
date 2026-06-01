from __future__ import annotations

import asyncio
import logging
import re
import sys
import time
import types
from collections import deque
from datetime import UTC, datetime

import httpx

from openalpha_brain.cli.algo_monitor import AlgoMonitor
from openalpha_brain.cli.alpha_channel import AlphaChannel
from openalpha_brain.cli.heartbeat import SessionHeartbeat
from openalpha_brain.config.config import settings
from openalpha_brain.core.pipeline import AlphaCachePool
from openalpha_brain.core.scheduler import ExplorationScheduler
from openalpha_brain.data import get_data_path
from openalpha_brain.evolution.hypothesis_aligner import HypothesisAligner
from openalpha_brain.evolution.quality_diversity import FeatureMap
from openalpha_brain.evolution.semantic_mutator import SemanticMutator
from openalpha_brain.evolution.strategy_classifier import StrategyClassifier
from openalpha_brain.generation.alpha_logics import AlphaLogicLibrary
from openalpha_brain.knowledge.conversation_summarizer import ConversationSummarizer
from openalpha_brain.knowledge.evolution_db import EvolutionDatabase
from openalpha_brain.knowledge.global_knowledge import GlobalKnowledge
from openalpha_brain.knowledge.rag_tools import RAGBudgetTracker
from openalpha_brain.learning.experience_distiller import ExperienceDistiller
from openalpha_brain.learning.mab import save_mab_state
from openalpha_brain.learning.param_optimizer import ParamOptimizer
from openalpha_brain.learning.reflection_engine import ReflectionEngine
from openalpha_brain.services import llm_client
from openalpha_brain.utils.market_state import MarketStateInferencer
from openalpha_brain.utils.pnl_analyzer import PnLAnalyzer
from openalpha_brain.utils.tool_factory import ToolFactory
from openalpha_brain.validation import validator as val
from openalpha_brain.validation.decay_detector import AlphaDecayDetector
from openalpha_brain.validation.signal_arbiter import SignalArbiter

MAX_BRAIN_MUTATIONS = 10

_OPS_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
_FIELDS_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b(?!\s*\()")

logger = logging.getLogger(__name__)

_CTX_ATTR_NAMES = frozenset({
    '_brain_cookies', '_pool', '_brain_cookies_lock', '_rag_engine',
    '_budget_tracker', '_mab', '_scheduler', '_association', '_whitelist_mgr',
    '_heartbeat', '_global_knowledge', '_last_merged_idx', '_summarizer',
    '_param_optimizer', '_monitor', '_successful_brain_expressions', '_logic_library',
    '_success_lib', '_failure_lib', '_strategy_classifier', '_feature_map',
    '_evo_db', '_semantic_mutator', '_hypothesis_aligner', '_reflection_engine',
    '_tool_factory', '_experience_distiller', '_evolution_cycle_count',
    '_last_diversity_stats', '_last_unexplored_directions', '_diversity_last_cycle',
    '_previous_expressions', '_market_state_inferencer', '_pnl_analyzer',
    '_decay_detector', '_decay_state', '_signal_arbiter', '_alpha_channel',
    '_alpha_channel_integrator', '_fastexpr_grammar', '_success_rate_tracker',
    '_algo_call_counts', '_console_pause_event', '_console_stop_event',
    '_brain_feedback_buffer', '_generation_gates', '_crossover_engine',
    '_trajectory_crossover_insights', '_crossover_exploration_proposals',
    '_weak_segment_alerts',
})


class _LoopStateModule(types.ModuleType):
    def __getattr__(self, name):
        _c = self.__dict__.get('_ctx')
        if _c is not None and name in _CTX_ATTR_NAMES:
            return getattr(_c, name)
        raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")

    def __setattr__(self, name, value):
        if name in _CTX_ATTR_NAMES:
            _c = self.__dict__.get('_ctx')
            if _c is not None:
                object.__setattr__(_c, name, value)
        super().__setattr__(name, value)

    def __dir__(self):
        return list(set(super().__dir__()) | _CTX_ATTR_NAMES)


class LoopContext:
    _mod_ref = None

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name in _CTX_ATTR_NAMES and LoopContext._mod_ref is not None:
            LoopContext._mod_ref.__dict__[name] = value

    def __init__(self) -> None:
        self._brain_cookies: httpx.Cookies | None = None
        self._pool = AlphaCachePool()
        self._brain_cookies_lock = asyncio.Lock()
        self._rag_engine = None
        self._budget_tracker: RAGBudgetTracker | None = None
        self._mab = None
        self._scheduler: ExplorationScheduler | None = None
        self._association = None
        self._whitelist_mgr = None
        self._heartbeat: SessionHeartbeat | None = None
        self._global_knowledge: GlobalKnowledge | None = None
        self._last_merged_idx: dict[str, int] = {}
        self._summarizer: ConversationSummarizer | None = None
        self._param_optimizer: ParamOptimizer | None = None
        self._monitor = AlgoMonitor.get_instance()
        self._successful_brain_expressions: list[str] = []
        self._logic_library: AlphaLogicLibrary | None = None
        self._success_lib = None
        self._failure_lib = None
        self._strategy_classifier: StrategyClassifier | None = None
        self._feature_map: FeatureMap | None = None
        self._evo_db: EvolutionDatabase | None = None
        self._semantic_mutator: SemanticMutator | None = None
        self._hypothesis_aligner: HypothesisAligner | None = None
        self._reflection_engine: ReflectionEngine | None = None
        self._tool_factory: ToolFactory | None = None
        self._experience_distiller: ExperienceDistiller | None = None
        self._evolution_cycle_count: int = 0
        self._last_diversity_stats: dict | None = None
        self._last_unexplored_directions: list[str] | None = None
        self._diversity_last_cycle: int = 0
        self._previous_expressions: deque = deque(maxlen=20)
        self._market_state_inferencer: MarketStateInferencer | None = None
        self._pnl_analyzer: PnLAnalyzer | None = None
        self._decay_detector: AlphaDecayDetector | None = None
        self._decay_state: dict = {}
        self._signal_arbiter: SignalArbiter | None = None
        self._alpha_channel: AlphaChannel | None = None
        self._alpha_channel_integrator = None
        self._fastexpr_grammar: str | None = None
        self._success_rate_tracker: deque | None = None
        self._algo_call_counts: dict[str, int] = {}
        self._console_pause_event: asyncio.Event | None = None
        self._console_stop_event: asyncio.Event | None = None
        self._brain_feedback_buffer: list = []
        self._generation_gates = None
        self._crossover_engine = None
        self._trajectory_crossover_insights: list = []
        self._crossover_exploration_proposals: list = []
        self._weak_segment_alerts: list = []

    def _rebuild_successful_expressions(self) -> list[str]:
        if self._evo_db is None:
            return self._successful_brain_expressions
        try:
            cutoff = time.time() - 90 * 86400
            with self._evo_db._lock:
                candidates = [
                    rec for rec in self._evo_db._records.values()
                    if rec.status == "PASS" and rec.timestamp >= cutoff and rec.sharpe is not None
                ]
            candidates.sort(key=lambda r: r.sharpe or 0, reverse=True)
            seen: set[str] = set()
            result: list[str] = []
            for rec in candidates[:400]:
                expr = rec.expr.strip()
                if not expr or expr in seen:
                    continue
                seen.add(expr)
                result.append(expr)
                if len(result) >= 200:
                    break
            self._successful_brain_expressions = result
            logger.info("Rebuilt _successful_brain_expressions: %d expressions from evolution_db (last 90 days, top 200)", len(result))
        except (OSError, ValueError, RuntimeError):
            logger.warning("Failed to rebuild _successful_brain_expressions from evolution_db", exc_info=True)
        return self._successful_brain_expressions

    def algo_tick(self, name: str) -> None:
        self._algo_call_counts[name] = self._algo_call_counts.get(name, 0) + 1

    def get_algo_call_stats(self) -> dict[str, int]:
        return dict(self._algo_call_counts)

    def get_dashboard_state(self) -> dict:
        return {
            "mab": self._mab,
            "whitelist_mgr": self._whitelist_mgr,
            "association": self._association,
            "heartbeat": self._heartbeat,
            "param_optimizer": self._param_optimizer,
            "global_knowledge": self._global_knowledge,
        }

    def init(self) -> None:
        self._pnl_analyzer = PnLAnalyzer()
        self._summarizer = ConversationSummarizer(threshold=20, keep_recent=5)

        if settings.STRATEGY_CLASSIFIER_ENABLED:
            self._strategy_classifier = StrategyClassifier(path=settings.STRATEGY_CLASSIFIER_PATH, embed_fn=llm_client.embed)
        else:
            self._strategy_classifier = None

        self._heartbeat = SessionHeartbeat(
            timeout_seconds=600,
            scan_interval_seconds=60,
        )

        self._global_knowledge = GlobalKnowledge()
        self._global_knowledge.load()

        self._param_optimizer = ParamOptimizer()

        self._logic_library = AlphaLogicLibrary()

        if settings.EVOLUTION_DB_ENABLED:
            self._evo_db = EvolutionDatabase(path=settings.EVOLUTION_DB_PATH, embed_fn=llm_client.embed)
            logger.info("EvolutionDatabase initialized: %s", settings.EVOLUTION_DB_PATH)
            self._rebuild_successful_expressions()
        else:
            self._evo_db = None

        if settings.SEMANTIC_MUTATOR_ENABLED:
            self._semantic_mutator = SemanticMutator(embed_fn=llm_client.embed, llm_generate_fn=llm_client.generate)
            logger.info("SemanticMutator initialized")
        else:
            self._semantic_mutator = None

        if settings.HYPOTHESIS_ALIGNER_ENABLED:
            self._hypothesis_aligner = HypothesisAligner()
            logger.info("HypothesisAligner initialized")
        else:
            self._hypothesis_aligner = None

        if settings.REFLECTION_ENGINE_ENABLED:
            self._reflection_engine = ReflectionEngine(path=settings.REFLECTION_ENGINE_PATH, embed_fn=llm_client.embed)
            self._reflection_engine.set_llm_generate_fn(llm_client.generate)
            logger.info("ReflectionEngine initialized: %s", settings.REFLECTION_ENGINE_PATH)
        else:
            self._reflection_engine = None

        if settings.TOOL_FACTORY_ENABLED:
            self._tool_factory = ToolFactory(path=settings.TOOL_FACTORY_PATH, embed_fn=llm_client.embed)
            logger.info("ToolFactory initialized: %s", settings.TOOL_FACTORY_PATH)
        else:
            self._tool_factory = None

        if settings.EXPERIENCE_DISTILLER_ENABLED:
            self._experience_distiller = ExperienceDistiller(embed_fn=llm_client.embed, path=settings.EXPERIENCE_DISTILLER_PATH)
            logger.info("ExperienceDistiller initialized")
        else:
            self._experience_distiller = None

        if settings.MARKET_STATE_ENABLED:
            self._market_state_inferencer = MarketStateInferencer()
            logger.info("MarketStateInferencer initialized")
        else:
            self._market_state_inferencer = None

        if settings.FEATURE_MAP_ENABLED:
            self._feature_map = FeatureMap(path=settings.FEATURE_MAP_PATH)
            logger.info("FeatureMap initialized: %s", settings.FEATURE_MAP_PATH)
        else:
            self._feature_map = None

        from openalpha_brain.knowledge.rag_engine import FailureFixLibrary, SuccessCaseLibrary

        if settings.SUCCESS_CASE_LIBRARY_ENABLED:
            self._success_lib = SuccessCaseLibrary(embed_fn=llm_client.embed)
            self._success_lib.load()
            logger.info("SuccessCaseLibrary initialized")
        else:
            self._success_lib = None

        if settings.FAILURE_FIX_LIBRARY_ENABLED:
            self._failure_lib = FailureFixLibrary(embed_fn=llm_client.embed)
            self._failure_lib.load()
            logger.info("FailureFixLibrary initialized")
        else:
            self._failure_lib = None

        if getattr(settings, 'ALPHA_CHANNEL_ENABLED', True):
            self._alpha_channel = AlphaChannel(
                stream_threshold=getattr(settings, 'ALPHA_CHANNEL_STREAM_THRESHOLD', 1.0),
                batch_size=getattr(settings, 'ALPHA_CHANNEL_BATCH_SIZE', 5),
                batch_timeout=getattr(settings, 'ALPHA_CHANNEL_BATCH_TIMEOUT', 30.0),
            )
            logger.info("AlphaChannel initialized")

            from openalpha_brain.cli.alpha_channel import AlphaChannelIntegrator

            async def _integrator_mab_update(direction: str, expression: str, reward: float) -> None:
                logger.info(
                    "[integrator] MAB update skipped (handled by main loop): direction=%s reward=%.4f expr=%s",
                    direction, reward, expression[:60],
                )

            async def _integrator_whitelist_update(expression: str, reward: float) -> None:
                if self._whitelist_mgr and expression:
                    _FIELDS_RE_LOCAL = re.compile(r'\b([a-z_]\w{2,})\b')
                    for _f in _FIELDS_RE_LOCAL.findall(expression):
                        self._whitelist_mgr.update_field_reward(_f.lower(), reward=reward)

            async def _integrator_success_lib_add(expression: str, direction: str, sharpe: float) -> None:
                if self._success_lib and expression:
                    try:
                        await self._success_lib.add_case(
                            expr=expression, hypothesis=direction,
                            sharpe=sharpe, fitness=0.0, turnover=0.0,
                            direction=direction, session_id="alpha_channel",
                        )
                    except (OSError, ValueError, RuntimeError):
                        pass

            self._alpha_channel_integrator = AlphaChannelIntegrator(
                channel=self._alpha_channel,
                mab_update_fn=_integrator_mab_update,
                whitelist_update_fn=_integrator_whitelist_update,
                success_lib_fn=_integrator_success_lib_add,
            )
            logger.info("AlphaChannelIntegrator initialized")

        _fpm = None
        try:
            from openalpha_brain.knowledge.field_proxy_map import get_field_proxy_map
            _fpm = get_field_proxy_map()
        except (ImportError, AttributeError, RuntimeError):
            _fpm = None

        if not settings.RAG_ENABLED and not settings.MAB_ENABLED:
            if self._scheduler is None:
                from openalpha_brain.learning.mab import TemplateFamilyBandit
                self._scheduler = ExplorationScheduler(
                    template_bandit=TemplateFamilyBandit(),
                    feature_map=self._feature_map,
                    field_proxy_map=_fpm,
                )
            return
        from openalpha_brain.knowledge.rag_engine import RAGEngine
        from openalpha_brain.learning.mab import (
            AssociationMatrix,
            HierarchicalMAB,
            TemplateFamilyBandit,
            load_mab_state,
        )
        from openalpha_brain.utils.whitelist import WhitelistManager

        self._whitelist_mgr = WhitelistManager()
        val.set_whitelist_manager(self._whitelist_mgr)

        try:
            loaded = load_mab_state()
        except (OSError, FileNotFoundError, ValueError, RuntimeError):
            loaded = None
        if loaded:
            self._mab, self._association, wl_data, _arbiter_data, _tf_bandit, _scheduler_data = loaded
            self._whitelist_mgr = WhitelistManager.from_dict(wl_data)
            val.set_whitelist_manager(self._whitelist_mgr)
            if _scheduler_data is not None:
                self._scheduler = ExplorationScheduler.from_dict(_scheduler_data)
                self._scheduler.feature_map = self._feature_map
                self._scheduler.field_proxy_map = _fpm
            elif _tf_bandit is not None:
                self._scheduler = ExplorationScheduler(template_bandit=_tf_bandit, feature_map=self._feature_map, field_proxy_map=_fpm)
            else:
                logger.warning("MAB state loaded but scheduler_data/tf_bandit missing — creating default Scheduler (data migration)")
                self._scheduler = ExplorationScheduler(
                    template_bandit=_tf_bandit or TemplateFamilyBandit(),
                    feature_map=self._feature_map,
                    field_proxy_map=_fpm,
                )
        else:
            self._mab = HierarchicalMAB()
            self._association = AssociationMatrix()
            _arbiter_data = None
            self._scheduler = ExplorationScheduler(
                template_bandit=TemplateFamilyBandit(),
                feature_map=self._feature_map,
                field_proxy_map=_fpm,
            )

        self._rag_engine = RAGEngine(
            top_k_ops=settings.RAG_TOP_K_OPS,
            top_k_fields=settings.RAG_TOP_K_FIELDS,
        )
        self._rag_engine.load_indexes()
        self._rag_engine.set_embed_fn(llm_client.embed)
        self._rag_engine.set_whitelist_manager(self._whitelist_mgr)
        self._rag_engine.set_eliminated_fields(set(self._whitelist_mgr.eliminated_fields.keys()))
        if self._rag_engine and hasattr(self._rag_engine, 'load_feedback_weights'):
            self._rag_engine.load_feedback_weights("rag_feedback_weights.json")

        self._budget_tracker = RAGBudgetTracker(budget=settings.RAG_BUDGET_PER_CYCLE)
        from openalpha_brain.knowledge import rag_engine as _rag_engine_mod
        from openalpha_brain.knowledge import rag_tools as _rag_tools_mod
        _rag_engine_mod.set_budget_tracker(self._budget_tracker)
        _rag_tools_mod.set_budget_tracker(self._budget_tracker)

        if settings.SIGNAL_ARBITER_ENABLED and (self._rag_engine is not None or self._mab is not None):
            self._success_rate_tracker = deque(maxlen=20)
            if _arbiter_data:
                self._signal_arbiter = SignalArbiter.from_dict(_arbiter_data, success_rate_tracker=self._success_rate_tracker)
            else:
                self._signal_arbiter = SignalArbiter(success_rate_tracker=self._success_rate_tracker)
            logger.info("SignalArbiter initialized")

        if settings.EVIDENCE_MAB_BIAS_ENABLED and self._logic_library and self._mab:
            try:
                direction_weights = self._logic_library.get_direction_weights()
                for direction, weight in direction_weights.items():
                    self._mab.set_initial_bias(direction, weight)
                logger.info("MAB initial bias set from AlphaLogicLibrary: %s", direction_weights)
            except (OSError, ValueError, RuntimeError):
                logger.warning("MAB bias initialization failed", exc_info=True)

        logger.info("Intelligent search initialized: RAG=%s MAB=%s", settings.RAG_ENABLED, settings.MAB_ENABLED)

        if settings.FASTEXPR_GRAMMAR_ENABLED and settings.LLM_PROVIDER.lower() == "lmstudio":
            try:
                _grammar_path = get_data_path("fastexpr_grammar.gbnf")
                if _grammar_path.exists():
                    self._fastexpr_grammar = _grammar_path.read_text(encoding="utf-8")
                    logger.info("FastExpr GBNF grammar loaded (%d bytes)", len(self._fastexpr_grammar))
                else:
                    logger.warning("FastExpr GBNF grammar file not found: %s", _grammar_path)
            except (OSError, FileNotFoundError, ValueError, RuntimeError):
                logger.warning("Failed to load FastExpr grammar", exc_info=True)

        if settings.AUTOBRAIN_SIM_ENABLED and settings.BRAIN_EMAIL and settings.BRAIN_PASSWORD:
            from openalpha_brain.services.brain_data_client import init_brain_data_client
            init_brain_data_client(settings.BRAIN_EMAIL, settings.BRAIN_PASSWORD)

        try:
            from openalpha_brain.services import brain_submitter as _bs_mod
            if getattr(_bs_mod, '_DYNAMIC_SKILL_ENABLED', False) and getattr(_bs_mod, '_dynamic_skill_lib', None) is None:
                from openalpha_brain.knowledge.dynamic_skill_library import DynamicSkillLibrary as _DSL
                _bs_mod._dynamic_skill_lib = _DSL()
                logger.info("DynamicSkillLibrary initialized (instance created, await initialize_from_brain for async init)")
        except ImportError:
            logger.info("DynamicSkillLibrary not available (import failed)")
        except (AttributeError, RuntimeError):
            logger.warning("DynamicSkillLibrary initialization failed", exc_info=True)

        if self._scheduler is not None:
            self._scheduler.initialize()

    def save_state(self) -> None:
        if not self._mab and not self._whitelist_mgr and not self._association:
            return
        try:
            whitelist_data = self._whitelist_mgr.to_dict() if self._whitelist_mgr else {}
            arbiter_data = self._signal_arbiter.to_dict() if self._signal_arbiter else None
            save_mab_state(
                self._mab, self._association, whitelist_data,
                arbiter_data=arbiter_data,
                template_family_bandit=self._scheduler.bandit if self._scheduler else None,
            )
            if self._scheduler is not None:
                import json as _json

                from openalpha_brain.learning.mab import _MAB_STATE_PATH
                _saved = _json.loads(_MAB_STATE_PATH.read_text(encoding="utf-8"))
                _saved["scheduler"] = self._scheduler.to_dict()
                _MAB_STATE_PATH.write_text(_json.dumps(_saved, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("Intelligent search state persisted to mab_state.json")
        except (OSError, RuntimeError):
            logger.warning("Failed to save intelligent search state", exc_info=True)
        if self._rag_engine and hasattr(self._rag_engine, 'save_feedback_weights'):
            try:
                self._rag_engine.save_feedback_weights("rag_feedback_weights.json")
            except (OSError, RuntimeError):
                logger.warning("Failed to save RAG feedback weights", exc_info=True)

    def log(self, state, log_type: str, message: str, detail: dict | None = None) -> None:
        entry = {
            "time": datetime.now(UTC).strftime("%H:%M:%S"),
            "type": log_type,
            "message": message,
        }
        if detail:
            entry["detail"] = detail
        if not hasattr(state, "activity_log") or state.activity_log is None:
            state.activity_log = []
        state.activity_log.append(entry)
        state.activity_log = state.activity_log[-200:]
        logger.info("[activity] [%s] %s", log_type, message)

    def build_blacklist_prompt(self) -> str:
        if not self._global_knowledge:
            return ""
        never_use = self._global_knowledge.get_never_use_list()
        if not never_use:
            return ""
        lines = [
            "",
            "══════════════════════════════════════════════════════════════════════",
            "GLOBAL NEVER USE LIST (confirmed invalid across multiple sessions)",
            "══════════════════════════════════════════════════════════════════════",
            "The following variables/operators have been repeatedly hallucinated",
            "and confirmed INVALID. DO NOT USE them in any expression:",
        ]
        for item in never_use:
            lines.append(f"  ✗ {item}")
        lines.append("══════════════════════════════════════════════════════════════════════")
        return "\n".join(lines)


_ctx = LoopContext()


def get_brain_cookies():
    """[Brief description of function purpose.]
        """
    return _ctx._brain_cookies


def set_brain_cookies(cookies):
    """[Brief description of function purpose.]

        Args:
            cookies: [Description]
        """
    _ctx._brain_cookies = cookies


def _rebuild_successful_expressions() -> list[str]:
    """[Brief description of function purpose.]

        Returns:
            list[str]: [Description]
        """
    return _ctx._rebuild_successful_expressions()


def _algo_tick(name: str) -> None:
    """[Brief description of function purpose.]

        Args:
            name (str): [Description]

        Returns:
            None: [Description]
        """
    _ctx.algo_tick(name)


def get_algo_call_stats() -> dict[str, int]:
    """[Brief description of function purpose.]

        Returns:
            dict[str, int]: [Description]
        """
    return _ctx.get_algo_call_stats()


def get_dashboard_state() -> dict:
    """[Brief description of function purpose.]

        Returns:
            dict: [Description]
        """
    return _ctx.get_dashboard_state()


def init_intelligent_search() -> None:
    """[Brief description of function purpose.]

        Returns:
            None: [Description]
        """
    _ctx.init()


def _save_intelligent_search_state() -> None:
    """[Brief description of function purpose.]

        Returns:
            None: [Description]
        """
    _ctx.save_state()


def _log(state, log_type: str, message: str, detail: dict | None = None) -> None:
    """[Brief description of function purpose.]

        Args:
            state: [Description]
            log_type (str): [Description]
            message (str): [Description]
            detail (dict | None): [Description]

        Returns:
            None: [Description]
        """
    _ctx.log(state, log_type, message, detail)


def _build_global_blacklist_prompt() -> str:
    """[Brief description of function purpose.]

        Returns:
            str: [Description]
        """
    return _ctx.build_blacklist_prompt()


_mod = _LoopStateModule(__name__)
_mod.__dict__.update(globals())
_mod._ctx = _ctx
for _n in _CTX_ATTR_NAMES:
    _mod.__dict__[_n] = getattr(_ctx, _n)
LoopContext._mod_ref = _mod
sys.modules[__name__] = _mod
