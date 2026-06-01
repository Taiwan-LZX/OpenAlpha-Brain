from openalpha_brain.validation.ast_repair import repair_expression, _VALID_OPS_CACHE, _VALID_VARS_CACHE


class TestRepairExpression:
    def test_valid_expression_no_repair(self):
        expr = "group_neutralize(rank(close), industry)"
        repaired, entries = repair_expression(expr)
        assert len(entries) == 0

    def test_invalid_var_repair(self):
        expr = "group_neutralize(rank(short_ratio), industry)"
        repaired, entries = repair_expression(expr)
        assert len(entries) > 0
        assert any("short_ratio" in e.get("variable", "") for e in entries)

    def test_unknown_operator_repair(self):
        expr = "group_neutralize(unknown_func(close), industry)"
        repaired, entries = repair_expression(expr)
        assert len(entries) > 0
        assert any("unknown_func" in e.get("variable", "") for e in entries)

    def test_cache_loaded(self):
        assert len(_VALID_OPS_CACHE) > 0
        assert len(_VALID_VARS_CACHE) > 0

    def test_close_var_not_flagged(self):
        expr = "group_neutralize(ts_decay_linear(close, 10), industry)"
        repaired, entries = repair_expression(expr)
        close_entries = [e for e in entries if e.get("variable") == "close"]
        assert len(close_entries) == 0, "close should be a valid variable"
