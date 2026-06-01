from openalpha_brain.generation.alpha_logics import AlphaLogicLibrary


class TestAlphaLogicLibrary:
    def test_get_logic_for_direction(self):
        lib = AlphaLogicLibrary()
        logics = lib.get_logic_for_direction("momentum")
        assert isinstance(logics, list)

    def test_get_direction_weights(self):
        lib = AlphaLogicLibrary()
        weights = lib.get_direction_weights()
        assert isinstance(weights, dict)

    def test_accumulate_diagnosis(self):
        lib = AlphaLogicLibrary()
        lib.accumulate_diagnosis(
            "momentum",
            {
                "failure_type": "signal_too_weak",
                "root_cause": "momentum signal arbitraged away",
                "suggested_fix": "use ts_zscore instead of ts_delta",
                "confidence": 0.8,
            },
        )

    def test_get_templates_for_direction(self):
        lib = AlphaLogicLibrary()
        templates = lib.get_templates_for_direction("momentum")
        assert isinstance(templates, list)

    def test_record_evidence(self):
        lib = AlphaLogicLibrary()
        logics = lib.get_logic_for_direction("momentum")
        if logics:
            lib.record_evidence(logics[0].logic_id, True)
