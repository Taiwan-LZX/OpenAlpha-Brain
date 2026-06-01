from openalpha_brain.generation import alpha_parser as parser

_V1_TEMPLATE = (
    "[1] ECONOMIC RATIONALE\n{rationale}\n"
    "[2] ALPHA EXPRESSION\n{expression}\n"
    "[3] ESTIMATED METRICS\n{metrics}\n"
    "[4] STRUCTURAL FINGERPRINT\nDataset: Price/Vol\n"
    "[5] REFINEMENT LOG\nTest refinement\n"
    "[6] DECISION\nSUBMIT CANDIDATE\n"
    "[7] MUTATION PATHS\n- Test path"
)


class TestParseRange:
    def test_plain_number(self):
        metrics = "Sharpe: 1.35-1.60\nFitness: 1.00-1.50\nTurnover: 15%-25%\nReturns: 19%"
        raw = _V1_TEMPLATE.format(
            rationale="Test rationale",
            expression="group_neutralize(rank(close), industry)",
            metrics=metrics,
        )
        result = parser.parse_alpha_output(raw)
        assert result is not None
        assert result["metrics"]["sharpe_min"] == 1.35

    def test_tilde_prefix(self):
        metrics = "Sharpe: ~1.35-~1.60\nFitness: ~1.00-~1.50\nTurnover: ~15%-~25%\nReturns: ~19%"
        raw = _V1_TEMPLATE.format(
            rationale="Test rationale",
            expression="group_neutralize(rank(close), industry)",
            metrics=metrics,
        )
        result = parser.parse_alpha_output(raw)
        assert result is not None
        assert result["metrics"]["sharpe_min"] == 1.35, f"Got sharpe_min={result['metrics'].get('sharpe_min')}"

    def test_approx_symbol(self):
        metrics = "Sharpe: ≈1.35\nFitness: ≈1.00\nTurnover: ≈15%\nReturns: ≈19%"
        raw = _V1_TEMPLATE.format(
            rationale="Test rationale",
            expression="group_neutralize(rank(close), industry)",
            metrics=metrics,
        )
        result = parser.parse_alpha_output(raw)
        assert result is not None
        assert result["metrics"]["sharpe_min"] == 1.35
