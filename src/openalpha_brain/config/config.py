from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

_ENV_PATH = Path(__file__).resolve().parent.parent.parent.parent / ".env"


class LLMSettings(BaseSettings):
    model_config = {"env_file": str(_ENV_PATH), "env_file_encoding": "utf-8", "extra": "ignore"}
    LLM_PROVIDER: str = "anthropic"
    LLM_MODEL: str = "claude-sonnet-4-20250514"
    LLM_API_KEY: str | None = None
    LLM_BASE_URL: str | None = None
    LLM_TEMPERATURE: float = Field(default=0.8, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int = Field(default=4096, ge=100)
    LLM_MAX_CONCURRENT: int = Field(default=4, ge=1, le=20)
    EMBED_MAX_CONCURRENT: int = Field(default=4, ge=1, le=20)
    EMBED_MODEL: str = "llama-nemotron-embed-1b-v2"
    EMBED_BASE_URL: str = "http://localhost:1234/v1/embeddings"
    LMSTUDIO_API_BASE: str = "http://localhost:1234"


class BRAINSettings(BaseSettings):
    model_config = {"env_file": str(_ENV_PATH), "env_file_encoding": "utf-8", "extra": "ignore"}
    BRAIN_EMAIL: str | None = None
    BRAIN_PASSWORD: str | None = None
    BRAIN_SUBMIT_ENABLED: bool = True
    BRAIN_POLL_TIMEOUT: int = Field(default=300, ge=60)
    AUTOBRAIN_SIM_ENABLED: bool = True


class PipelineSettings(BaseSettings):
    model_config = {"env_file": str(_ENV_PATH), "env_file_encoding": "utf-8", "extra": "ignore"}
    PIPELINE_MODE: bool = True
    PIPELINE_MAX_SLOTS: int = Field(default=3, ge=1, le=3)
    PIPELINE_MAX_IMPROVEMENT_WORKERS: int = Field(default=2, ge=1)
    PIPELINE_QUEUE_MAX_SIZE: int = Field(default=100, ge=10)
    PIPELINE_SUBMIT_TIMEOUT: float = Field(default=600.0, ge=60.0)
    PIPELINE_IMPROVE_TIMEOUT: float = Field(default=120.0, ge=30.0)
    GENERATOR_PARALLEL_TASKS: int = Field(default=3, ge=1, le=10)


class MABSettings(BaseSettings):
    model_config = {"env_file": str(_ENV_PATH), "env_file_encoding": "utf-8", "extra": "ignore"}
    MAB_ENABLED: bool = True
    MAB_REWARD_VALIDATOR_PASS: float = 0.1
    MAB_REWARD_BRAIN_SUBMIT: float = 0.3
    MAB_REWARD_SHARPE_05: float = 0.5
    MAB_REWARD_SHARPE_10: float = 1.0
    MAB_PENALTY_BRAIN_FAIL: float = 0.3
    MAB_PENALTY_BRAIN_ERROR: float = 0.5
    MAB_PENALTY_OVERUSE: float = 0.1
    OVERUSE_THRESHOLD: int = 8
    OVERUSE_WINDOW: int = 20
    ELIMINATION_THRESHOLD: float = 0.15
    BETA_DECAY_FACTOR: float = 0.99
    BETA_DECAY_INTERVAL: int = 100
    EVIDENCE_MAB_BIAS_ENABLED: bool = True


class RewardSettings(BaseSettings):
    model_config = {"env_file": str(_ENV_PATH), "env_file_encoding": "utf-8", "extra": "ignore"}
    DRAWDOWN_PENALTY_THRESHOLD: float = 10.0
    DRAWDOWN_PENALTY: float = 0.03
    OVERFITTING_WARNING_PENALTY: float = 0.2
    MARGIN_EFFICIENCY_THRESHOLD: float = 0.5
    HIERARCHICAL_REWARD_ENABLED: bool = True
    HIERARCHICAL_REWARD_QUALITY_THRESHOLD: float = 0.6
    HIERARCHICAL_REWARD_BASIC_THRESHOLD: float = 0.3
    GARCH_CLUSTERING_PENALTY: float = 0.05
    GARCH_MIN_PNL_LENGTH: int = 20
    OVERFIT_DETECTION_PENALTY: float = 0.1
    OVERFIT_MAB_PENALTY: float = 0.3
    HIGH_CORRELATION_THRESHOLD: float = 0.7
    CORRELATION_PENALTY_THRESHOLD: float = 0.3
    CORRELATION_PENALTY_COEFFICIENT: float = 0.1
    PROD_CORRELATION_THRESHOLD: float = 0.7
    PROD_CORRELATION_PENALTY_COEFFICIENT: float = 0.15
    DIVERSITY_BONUS_THRESHOLD: float = 0.5
    DIVERSITY_BONUS: float = 0.05
    DIVERSITY_PENALTY_THRESHOLD: float = 0.2
    DIVERSITY_PENALTY: float = 0.05


class SignalArbiterSettings(BaseSettings):
    model_config = {"env_file": str(_ENV_PATH), "env_file_encoding": "utf-8", "extra": "ignore"}
    SIGNAL_ARBITER_ENABLED: bool = True
    SIGNAL_ARBITER_WEIGHT_LOWER_BOUND: float = 0.05
    SIGNAL_ARBITER_WEIGHT_UPPER_BOUND: float = 0.5
    SIGNAL_ARBITER_ADJUSTMENT_STEP: float = 0.05
    SIGNAL_ARBITER_ADJUSTMENT_INTERVAL: int = 5
    SIGNAL_ARBITER_TRACKER_WINDOW: int = 20
    SIGNAL_ARBITER_DEFAULT_SCORE: float = 0.5
    SIGNAL_ARBITER_LOW_SUCCESS_THRESHOLD: float = 0.3
    SIGNAL_ARBITER_HIGH_SUCCESS_THRESHOLD: float = 0.6


class FeatureSettings(BaseSettings):
    model_config = {"env_file": str(_ENV_PATH), "env_file_encoding": "utf-8", "extra": "ignore"}
    RAG_ENABLED: bool = True
    RAG_TOOL_CALL_ENABLED: bool = True
    RAG_BUDGET_PER_CYCLE: int = 3
    RAG_TOP_K_OPS: int = 15
    RAG_TOP_K_FIELDS: int = 40
    MULTI_AGENT_ENABLED: bool = True
    ORIGINALITY_CHECK_ENABLED: bool = True
    COMPLEXITY_CHECK_ENABLED: bool = True
    CROSSOVER_ENABLED: bool = True
    EVIDENCE_RECORDING_ENABLED: bool = True
    DEFAULT_EXPLORATION_DIRECTION: str = "momentum"
    FACTOR_TEMPLATE_MODE: str = "hybrid"
    TRAJECTORY_MUTATION_ENABLED: bool = False
    DIAGNOSIS_LLM_ENABLED: bool = True
    SEMANTIC_DRIFT_THRESHOLD: float = 0.7
    FASTEXPR_GRAMMAR_ENABLED: bool = True
    SUCCESS_CASE_LIBRARY_ENABLED: bool = True
    FAILURE_FIX_LIBRARY_ENABLED: bool = True
    EVOLUTION_DB_ENABLED: bool = True
    EVOLUTION_DB_PATH: str = "evolution_db.json"
    STRATEGY_CLASSIFIER_ENABLED: bool = True
    STRATEGY_CLASSIFIER_PATH: str = "strategy_profiles.json"
    FEATURE_MAP_ENABLED: bool = True
    FEATURE_MAP_PATH: str = "feature_map.json"
    REFLECTION_ENGINE_ENABLED: bool = True
    REFLECTION_ENGINE_PATH: str = "reflection_log.json"
    TOOL_FACTORY_ENABLED: bool = True
    TOOL_FACTORY_PATH: str = "alpha_tools.json"
    EXPERIENCE_DISTILLER_ENABLED: bool = True
    EXPERIENCE_DISTILLER_PATH: str = "experience_cards.json"
    SEMANTIC_MUTATOR_ENABLED: bool = True
    HYPOTHESIS_ALIGNER_ENABLED: bool = True
    ADAPTIVE_AGENT_ENABLED: bool = True
    MARKET_STATE_ENABLED: bool = True
    PARAM_OPTIMIZATION_ENABLED: bool = True
    ALPHA_CHANNEL_ENABLED: bool = True
    ALPHA_CHANNEL_STREAM_THRESHOLD: float = 1.0
    ALPHA_CHANNEL_BATCH_SIZE: int = 5
    ALPHA_CHANNEL_BATCH_TIMEOUT: float = 30.0
    OVERFIT_IS_OS_DECAY_SEVERE: float = 0.5
    OVERFIT_IS_OS_DECAY_WARNING: float = 0.7
    OVERFIT_YEARLY_SHARPE_CV_SEVERE: float = 1.0
    OVERFIT_YEARLY_SHARPE_CV_WARNING: float = 0.5


class LoopSettings(BaseSettings):
    model_config = {"env_file": str(_ENV_PATH), "env_file_encoding": "utf-8", "extra": "ignore"}
    MAX_CYCLES: int = Field(default=20, ge=1)
    MAX_MUTATIONS: int = Field(default=4, ge=1)
    SESSION_DIR: Path = Path("sessions")
    LOG_LEVEL: str = "INFO"


class Settings:
    def __init__(self) -> None:
        self._llm = LLMSettings()
        self._brain = BRAINSettings()
        self._pipeline = PipelineSettings()
        self._mab = MABSettings()
        self._reward = RewardSettings()
        self._signal_arbiter = SignalArbiterSettings()
        self._feature = FeatureSettings()
        self._loop = LoopSettings()
        self._all = {}
        for sub in (
            self._llm,
            self._brain,
            self._pipeline,
            self._mab,
            self._reward,
            self._signal_arbiter,
            self._feature,
            self._loop,
        ):
            for k, v in sub.model_dump().items():
                self._all[k] = v
                setattr(self, k, v)

    @property
    def llm(self) -> LLMSettings:
        return self._llm

    @property
    def brain(self) -> BRAINSettings:
        return self._brain

    @property
    def pipeline(self) -> PipelineSettings:
        return self._pipeline

    @property
    def mab(self) -> MABSettings:
        return self._mab

    @property
    def reward(self) -> RewardSettings:
        return self._reward

    @property
    def signal_arbiter(self) -> SignalArbiterSettings:
        return self._signal_arbiter

    @property
    def feature(self) -> FeatureSettings:
        return self._feature

    @property
    def loop(self) -> LoopSettings:
        return self._loop


settings = Settings()
