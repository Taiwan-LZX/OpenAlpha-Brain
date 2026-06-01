from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from openalpha_brain.config.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/config", tags=["config"])


class ConfigUpdateRequest(BaseModel):
    LLM_API_KEY: str | None = None
    LLM_BASE_URL: str | None = None
    LLM_MODEL: str | None = None
    LLM_PROVIDER: str | None = None
    LLM_TEMPERATURE: float | None = Field(default=None, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int | None = Field(default=None, ge=100)
    LLM_MAX_CONCURRENT: int | None = Field(default=None, ge=1, le=20)
    EMBED_MODEL: str | None = None
    EMBED_BASE_URL: str | None = None
    EMBED_MAX_CONCURRENT: int | None = Field(default=None, ge=1, le=20)
    BRAIN_EMAIL: str | None = None
    BRAIN_PASSWORD: str | None = None
    BRAIN_SUBMIT_ENABLED: bool | None = None
    BRAIN_POLL_TIMEOUT: int | None = Field(default=None, ge=60)
    AUTOBRAIN_SIM_ENABLED: bool | None = None
    PIPELINE_MODE: bool | None = None
    PIPELINE_MAX_SLOTS: int | None = Field(default=None, ge=1, le=3)
    GENERATOR_PARALLEL_TASKS: int | None = Field(default=None, ge=1, le=10)
    MAX_CYCLES: int | None = Field(default=None, ge=1)
    MAX_MUTATIONS: int | None = Field(default=None, ge=1)
    LOG_LEVEL: str | None = None
    MAB_ENABLED: bool | None = None
    RAG_ENABLED: bool | None = None
    RAG_TOOL_CALL_ENABLED: bool | None = None
    RAG_BUDGET_PER_CYCLE: int | None = None
    RAG_TOP_K_OPS: int | None = None
    RAG_TOP_K_FIELDS: int | None = None
    MULTI_AGENT_ENABLED: bool | None = None
    ORIGINALITY_CHECK_ENABLED: bool | None = None
    COMPLEXITY_CHECK_ENABLED: bool | None = None
    CROSSOVER_ENABLED: bool | None = None
    EVIDENCE_RECORDING_ENABLED: bool | None = None
    SUCCESS_CASE_LIBRARY_ENABLED: bool | None = None
    FAILURE_FIX_LIBRARY_ENABLED: bool | None = None
    EXPERIENCE_DISTILLER_ENABLED: bool | None = None
    STRATEGY_CLASSIFIER_ENABLED: bool | None = None
    FEATURE_MAP_ENABLED: bool | None = None
    REFLECTION_ENGINE_ENABLED: bool | None = None
    TOOL_FACTORY_ENABLED: bool | None = None
    SEMANTIC_MUTATOR_ENABLED: bool | None = None
    HYPOTHESIS_ALIGNER_ENABLED: bool | None = None
    ADAPTIVE_AGENT_ENABLED: bool | None = None
    MARKET_STATE_ENABLED: bool | None = None
    PARAM_OPTIMIZATION_ENABLED: bool | None = None
    SIGNAL_ARBITER_ENABLED: bool | None = None
    EVIDENCE_MAB_BIAS_ENABLED: bool | None = None
    FASTEXPR_GRAMMAR_ENABLED: bool | None = None
    EVOLUTION_DB_ENABLED: bool | None = None
    ALPHA_CHANNEL_ENABLED: bool | None = None
    DIAGNOSIS_LLM_ENABLED: bool | None = None


class ConnectionTestRequest(BaseModel):
    test_type: str = Field(default="llm", pattern="^(llm|brain|embed)$")


def _mask_secret(value: str | None) -> str | None:
    if not value:
        return value
    if len(value) <= 4:
        return "****"
    return value[:4] + "***"


@router.get("")
async def get_config():
    llm = settings.llm
    brain = settings.brain
    pipeline = settings.pipeline
    mab = settings.mab
    feature = settings.feature
    loop_s = settings.loop

    return {
        "llm": {
            "LLM_PROVIDER": llm.LLM_PROVIDER,
            "LLM_MODEL": llm.LLM_MODEL,
            "LLM_API_KEY": _mask_secret(llm.LLM_API_KEY),
            "LLM_BASE_URL": llm.LLM_BASE_URL,
            "LLM_TEMPERATURE": llm.LLM_TEMPERATURE,
            "LLM_MAX_TOKENS": llm.LLM_MAX_TOKENS,
            "LLM_MAX_CONCURRENT": llm.LLM_MAX_CONCURRENT,
            "EMBED_MODEL": llm.EMBED_MODEL,
            "EMBED_BASE_URL": llm.EMBED_BASE_URL,
            "EMBED_MAX_CONCURRENT": llm.EMBED_MAX_CONCURRENT,
            "LMSTUDIO_API_BASE": llm.LMSTUDIO_API_BASE,
        },
        "brain": {
            "BRAIN_EMAIL": brain.BRAIN_EMAIL,
            "BRAIN_PASSWORD": _mask_secret(brain.BRAIN_PASSWORD),
            "BRAIN_SUBMIT_ENABLED": brain.BRAIN_SUBMIT_ENABLED,
            "BRAIN_POLL_TIMEOUT": brain.BRAIN_POLL_TIMEOUT,
            "AUTOBRAIN_SIM_ENABLED": brain.AUTOBRAIN_SIM_ENABLED,
        },
        "pipeline": {
            "PIPELINE_MODE": pipeline.PIPELINE_MODE,
            "PIPELINE_MAX_SLOTS": pipeline.PIPELINE_MAX_SLOTS,
            "PIPELINE_MAX_IMPROVEMENT_WORKERS": pipeline.PIPELINE_MAX_IMPROVEMENT_WORKERS,
            "PIPELINE_QUEUE_MAX_SIZE": pipeline.PIPELINE_QUEUE_MAX_SIZE,
            "PIPELINE_SUBMIT_TIMEOUT": pipeline.PIPELINE_SUBMIT_TIMEOUT,
            "PIPELINE_IMPROVE_TIMEOUT": pipeline.PIPELINE_IMPROVE_TIMEOUT,
            "GENERATOR_PARALLEL_TASKS": pipeline.GENERATOR_PARALLEL_TASKS,
        },
        "mab": {
            "MAB_ENABLED": mab.MAB_ENABLED,
            "MAB_REWARD_VALIDATOR_PASS": mab.MAB_REWARD_VALIDATOR_PASS,
            "MAB_REWARD_BRAIN_SUBMIT": mab.MAB_REWARD_BRAIN_SUBMIT,
            "MAB_REWARD_SHARPE_05": mab.MAB_REWARD_SHARPE_05,
            "MAB_REWARD_SHARPE_10": mab.MAB_REWARD_SHARPE_10,
            "MAB_PENALTY_BRAIN_FAIL": mab.MAB_PENALTY_BRAIN_FAIL,
            "MAB_PENALTY_BRAIN_ERROR": mab.MAB_PENALTY_BRAIN_ERROR,
            "MAB_PENALTY_OVERUSE": mab.MAB_PENALTY_OVERUSE,
            "EVIDENCE_MAB_BIAS_ENABLED": mab.EVIDENCE_MAB_BIAS_ENABLED,
        },
        "feature": {
            "RAG_ENABLED": feature.RAG_ENABLED,
            "RAG_TOOL_CALL_ENABLED": feature.RAG_TOOL_CALL_ENABLED,
            "RAG_BUDGET_PER_CYCLE": feature.RAG_BUDGET_PER_CYCLE,
            "RAG_TOP_K_OPS": feature.RAG_TOP_K_OPS,
            "RAG_TOP_K_FIELDS": feature.RAG_TOP_K_FIELDS,
            "MULTI_AGENT_ENABLED": feature.MULTI_AGENT_ENABLED,
            "ORIGINALITY_CHECK_ENABLED": feature.ORIGINALITY_CHECK_ENABLED,
            "COMPLEXITY_CHECK_ENABLED": feature.COMPLEXITY_CHECK_ENABLED,
            "CROSSOVER_ENABLED": feature.CROSSOVER_ENABLED,
            "EVIDENCE_RECORDING_ENABLED": feature.EVIDENCE_RECORDING_ENABLED,
            "SUCCESS_CASE_LIBRARY_ENABLED": feature.SUCCESS_CASE_LIBRARY_ENABLED,
            "FAILURE_FIX_LIBRARY_ENABLED": feature.FAILURE_FIX_LIBRARY_ENABLED,
            "EXPERIENCE_DISTILLER_ENABLED": feature.EXPERIENCE_DISTILLER_ENABLED,
            "STRATEGY_CLASSIFIER_ENABLED": feature.STRATEGY_CLASSIFIER_ENABLED,
            "FEATURE_MAP_ENABLED": feature.FEATURE_MAP_ENABLED,
            "REFLECTION_ENGINE_ENABLED": feature.REFLECTION_ENGINE_ENABLED,
            "TOOL_FACTORY_ENABLED": feature.TOOL_FACTORY_ENABLED,
            "SEMANTIC_MUTATOR_ENABLED": feature.SEMANTIC_MUTATOR_ENABLED,
            "HYPOTHESIS_ALIGNER_ENABLED": feature.HYPOTHESIS_ALIGNER_ENABLED,
            "ADAPTIVE_AGENT_ENABLED": feature.ADAPTIVE_AGENT_ENABLED,
            "MARKET_STATE_ENABLED": feature.MARKET_STATE_ENABLED,
            "PARAM_OPTIMIZATION_ENABLED": feature.PARAM_OPTIMIZATION_ENABLED,
            "SIGNAL_ARBITER_ENABLED": settings.signal_arbiter.SIGNAL_ARBITER_ENABLED,
            "FASTEXPR_GRAMMAR_ENABLED": feature.FASTEXPR_GRAMMAR_ENABLED,
            "EVOLUTION_DB_ENABLED": feature.EVOLUTION_DB_ENABLED,
            "ALPHA_CHANNEL_ENABLED": feature.ALPHA_CHANNEL_ENABLED,
            "DIAGNOSIS_LLM_ENABLED": feature.DIAGNOSIS_LLM_ENABLED,
        },
        "loop": {
            "MAX_CYCLES": loop_s.MAX_CYCLES,
            "MAX_MUTATIONS": loop_s.MAX_MUTATIONS,
            "LOG_LEVEL": loop_s.LOG_LEVEL,
        },
    }


@router.put("")
async def update_config(req: ConfigUpdateRequest):
    updated_keys: list[str] = []

    def _apply(sub_settings: Any, field_name: str, new_value: Any) -> bool:
        if new_value is None:
            return False
        current = getattr(sub_settings, field_name, None)
        if current is None and not hasattr(sub_settings, field_name):
            return False
        setattr(sub_settings, field_name, new_value)
        os.environ[field_name] = str(new_value)
        settings._all[field_name] = new_value
        return True

    llm_fields = {
        "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LLM_PROVIDER",
        "LLM_TEMPERATURE", "LLM_MAX_TOKENS", "LLM_MAX_CONCURRENT",
        "EMBED_MODEL", "EMBED_BASE_URL", "EMBED_MAX_CONCURRENT",
    }
    brain_fields = {
        "BRAIN_EMAIL", "BRAIN_PASSWORD", "BRAIN_SUBMIT_ENABLED",
        "BRAIN_POLL_TIMEOUT", "AUTOBRAIN_SIM_ENABLED",
    }
    pipeline_fields = {
        "PIPELINE_MODE", "PIPELINE_MAX_SLOTS", "GENERATOR_PARALLEL_TASKS",
    }
    loop_fields = {"MAX_CYCLES", "MAX_MUTATIONS", "LOG_LEVEL"}
    mab_fields = {"MAB_ENABLED", "EVIDENCE_MAB_BIAS_ENABLED"}
    feature_fields = {
        "RAG_ENABLED", "RAG_TOOL_CALL_ENABLED", "RAG_BUDGET_PER_CYCLE",
        "RAG_TOP_K_OPS", "RAG_TOP_K_FIELDS", "MULTI_AGENT_ENABLED",
        "ORIGINALITY_CHECK_ENABLED", "COMPLEXITY_CHECK_ENABLED",
        "CROSSOVER_ENABLED", "EVIDENCE_RECORDING_ENABLED",
        "SUCCESS_CASE_LIBRARY_ENABLED", "FAILURE_FIX_LIBRARY_ENABLED",
        "EXPERIENCE_DISTILLER_ENABLED", "STRATEGY_CLASSIFIER_ENABLED",
        "FEATURE_MAP_ENABLED", "REFLECTION_ENGINE_ENABLED",
        "TOOL_FACTORY_ENABLED", "SEMANTIC_MUTATOR_ENABLED",
        "HYPOTHESIS_ALIGNER_ENABLED", "ADAPTIVE_AGENT_ENABLED",
        "MARKET_STATE_ENABLED", "PARAM_OPTIMIZATION_ENABLED",
        "SIGNAL_ARBITER_ENABLED", "FASTEXPR_GRAMMAR_ENABLED",
        "EVOLUTION_DB_ENABLED", "ALPHA_CHANNEL_ENABLED",
        "DIAGNOSIS_LLM_ENABLED",
    }

    sub_map: list[tuple[Any, set[str]]] = [
        (settings._llm, llm_fields),
        (settings._brain, brain_fields),
        (settings._pipeline, pipeline_fields),
        (settings._loop, loop_fields),
        (settings._mab, mab_fields),
        (settings._feature, feature_fields),
    ]

    data = req.model_dump(exclude_none=True)
    for key, value in data.items():
        for sub_obj, fields in sub_map:
            if key in fields:
                if _apply(sub_obj, key, value):
                    updated_keys.append(key)
                break

    logger.info("Config updated: %s", ", ".join(updated_keys) if updated_keys else "no changes")

    return {
        "updated": updated_keys,
        "count": len(updated_keys),
    }


@router.post("/test-connection")
async def test_connection(req: ConnectionTestRequest):
    if req.test_type == "llm":
        return await _test_llm_connection()
    if req.test_type == "brain":
        return await _test_brain_connection()
    if req.test_type == "embed":
        return await _test_embed_connection()
    return {"status": "error", "message": f"Unknown test_type: {req.test_type}"}


async def _test_llm_connection() -> dict:
    if not settings.LLM_API_KEY and settings.LLM_PROVIDER.lower() != "lmstudio":
        return {"status": "error", "message": "LLM_API_KEY is not set"}
    try:
        from openalpha_brain.services.llm_client import generate
        result = await generate(
            system_prompt="You are a test assistant.",
            history=[],
            user_msg="Reply with exactly: OK",
            session_id="connection-test",
            cycle=0,
        )
        return {
            "status": "ok",
            "provider": settings.LLM_PROVIDER,
            "model": settings.LLM_MODEL,
            "response_preview": result[:100] if result else "",
        }
    except (ConnectionError, OSError, TimeoutError) as exc:
        return {"status": "error", "message": str(exc)}


async def _test_brain_connection() -> dict:
    email = settings.BRAIN_EMAIL
    password = settings.BRAIN_PASSWORD
    if not email or not password:
        return {"status": "error", "message": "BRAIN_EMAIL or BRAIN_PASSWORD is not set"}
    try:
        from openalpha_brain.services.http_pool import get_client
        client = get_client()
        resp = await client.post(
            "https://api.worldquantbrain.com/authentication",
            json={"email": email, "password": password},
            timeout=15.0,
        )
        if resp.status_code == 201:
            return {"status": "ok", "message": "BRAIN authentication successful"}
        return {
            "status": "error",
            "message": f"BRAIN auth returned HTTP {resp.status_code}",
            "detail": resp.text[:200],
        }
    except (ConnectionError, OSError, TimeoutError) as exc:
        return {"status": "error", "message": str(exc)}


async def _test_embed_connection() -> dict:
    embed_url = settings.EMBED_BASE_URL
    if not embed_url:
        return {"status": "error", "message": "EMBED_BASE_URL is not set"}
    try:
        from openalpha_brain.services.llm_client import embed
        result = await embed("test connection")
        return {
            "status": "ok",
            "embed_url": embed_url,
            "embed_model": settings.EMBED_MODEL,
            "vector_dim": len(result) if result else 0,
        }
    except (ConnectionError, OSError, TimeoutError) as exc:
        return {"status": "error", "message": str(exc)}


@router.get("/models")
async def get_available_models():
    provider = settings.LLM_PROVIDER.lower()
    models: list[dict[str, str]] = []

    if provider == "lmstudio":
        try:
            from openalpha_brain.services.http_pool import get_client
            client = get_client()
            resp = await client.get(
                f"{settings.LMSTUDIO_API_BASE}/v1/models",
                timeout=10.0,
            )
            data = resp.json()
            for m in data.get("data", []):
                models.append({"id": m.get("id", ""), "provider": "lmstudio"})
        except (ConnectionError, OSError, TimeoutError) as exc:
            return {"status": "error", "message": str(exc), "models": []}
    elif provider == "openai":
        models = [
            {"id": "gpt-4o", "provider": "openai"},
            {"id": "gpt-4o-mini", "provider": "openai"},
            {"id": "gpt-4-turbo", "provider": "openai"},
            {"id": "o1", "provider": "openai"},
            {"id": "o3-mini", "provider": "openai"},
        ]
    elif provider == "anthropic":
        models = [
            {"id": "claude-sonnet-4-20250514", "provider": "anthropic"},
            {"id": "claude-3-5-sonnet-20241022", "provider": "anthropic"},
            {"id": "claude-3-5-haiku-20241022", "provider": "anthropic"},
            {"id": "claude-3-opus-20240229", "provider": "anthropic"},
        ]
    elif provider == "groq":
        models = [
            {"id": "llama-3.3-70b-versatile", "provider": "groq"},
            {"id": "llama-3.1-8b-instant", "provider": "groq"},
            {"id": "mixtral-8x7b-32768", "provider": "groq"},
        ]
    elif provider == "gemini":
        models = [
            {"id": "gemini-2.0-flash", "provider": "gemini"},
            {"id": "gemini-1.5-flash", "provider": "gemini"},
            {"id": "gemini-1.5-pro", "provider": "gemini"},
        ]

    return {
        "provider": provider,
        "current_model": settings.LLM_MODEL,
        "models": models,
    }
