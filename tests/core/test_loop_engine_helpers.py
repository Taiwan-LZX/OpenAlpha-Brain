import json
import pytest
from pathlib import Path
from openalpha_brain.config.config import settings
from openalpha_brain.core.loop_engine import _sync_mab_bias_from_evidence


class TestRecordEvidenceIntegration:
    def test_sync_mab_bias_function_exists(self):
        assert callable(_sync_mab_bias_from_evidence)

    def test_config_has_new_fields(self):
        assert hasattr(settings, 'DEFAULT_EXPLORATION_DIRECTION')
        assert hasattr(settings, 'FACTOR_TEMPLATE_MODE')
        assert hasattr(settings, 'TRAJECTORY_MUTATION_ENABLED')
        assert hasattr(settings, 'EVIDENCE_MAB_BIAS_ENABLED')
        assert hasattr(settings, 'DIAGNOSIS_LLM_ENABLED')
        assert hasattr(settings, 'SEMANTIC_DRIFT_THRESHOLD')


class TestBrainSubmitParams:
    def test_brain_submit_params_json_exists(self):
        p = Path(__file__).resolve().parent.parent.parent / "src" / "openalpha_brain" / "data" / "brain_submit_params.json"
        assert p.exists(), "brain_submit_params.json not found"
        with open(p) as f:
            params = json.load(f)
        assert "instrumentType" in params
        assert "delay" in params
        assert "decay" in params
        assert "neutralization" in params


class TestDirectionMaps:
    def test_direction_operator_map_json(self):
        p = Path(__file__).resolve().parent.parent.parent / "src" / "openalpha_brain" / "data" / "direction_operator_map.json"
        assert p.exists(), "direction_operator_map.json not found"
        with open(p) as f:
            data = json.load(f)
        assert "momentum" in data
        assert len(data) >= 4

    def test_direction_field_map_json(self):
        p = Path(__file__).resolve().parent.parent.parent / "src" / "openalpha_brain" / "data" / "direction_field_map.json"
        assert p.exists(), "direction_field_map.json not found"
        with open(p) as f:
            data = json.load(f)
        assert "momentum" in data
        assert len(data) >= 4
