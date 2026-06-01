import json
import os
import tempfile
import unittest
from pathlib import Path

from alpha_agent.rag_spec import (
    RAGSpecEncoder,
    DeterministicValidator,
    VALID_OPERATORS,
)


_FIELDS_JSON = json.dumps([
    {"id": "close"},
    {"id": "open"},
    {"id": "high"},
    {"id": "low"},
    {"id": "volume"},
    {"id": "vwap"},
    {"id": "returns"},
    {"id": "adv20"},
    {"id": "cap"},
    {"id": "sharesout"},
    {"id": "sector"},
    {"id": "bookvalue_ps"},
    {"id": "return_equity"},
    {"id": "return_assets"},
    {"id": "ebit"},
    {"id": "cashflow_op"},
    {"id": "sales"},
    {"id": "enterprise_value"},
])


class TestRAGSpecEncoder(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        self.tmp.write(_FIELDS_JSON)
        self.tmp.close()
        self.fields_path = Path(self.tmp.name)

    def tearDown(self) -> None:
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)

    def test_from_fields_summary(self) -> None:
        encoder = RAGSpecEncoder.from_fields_summary(self.fields_path)
        self.assertGreater(len(encoder.corpus), 0)
        self.assertGreater(len(encoder.field_index), 0)
        self.assertTrue(encoder._fitted)

    def test_encode(self) -> None:
        encoder = RAGSpecEncoder.from_fields_summary(self.fields_path)
        vec = encoder.encode("price momentum using close and volume")
        self.assertIsNotNone(vec)
        self.assertEqual(vec.shape[0], 1)
        self.assertGreater(vec.shape[1], 0)

    def test_similarity(self) -> None:
        encoder = RAGSpecEncoder.from_fields_summary(self.fields_path)
        sim = encoder.similarity("close", "close")
        self.assertAlmostEqual(sim, 1.0, places=4)

    def test_similarity_different(self) -> None:
        encoder = RAGSpecEncoder.from_fields_summary(self.fields_path)
        sim = encoder.similarity("close", "bookvalue_ps")
        self.assertLess(sim, 1.0)

    def test_similarity_empty(self) -> None:
        encoder = RAGSpecEncoder.from_fields_summary(self.fields_path)
        sim = encoder.similarity("", "close")
        self.assertEqual(sim, 0.0)

    def test_recommend_fields(self) -> None:
        encoder = RAGSpecEncoder.from_fields_summary(self.fields_path)
        recs = encoder.recommend_fields("stock price", top_k=3)
        self.assertLessEqual(len(recs), 3)
        for field, score in recs:
            self.assertIsInstance(field, str)
            self.assertIsInstance(score, float)
            self.assertGreater(score, 0)

    def test_recommend_fields_empty_corpus(self) -> None:
        encoder = RAGSpecEncoder()
        recs = encoder.recommend_fields("test", top_k=3)
        self.assertEqual(recs, [])

    def test_spec_decoder(self) -> None:
        encoder = RAGSpecEncoder.from_fields_summary(self.fields_path)
        decoded = encoder.spec_decoder("price momentum using close and volume")
        self.assertIn("feature_count", decoded)
        self.assertIn("top_features", decoded)
        self.assertGreater(decoded["feature_count"], 0)


class TestDeterministicValidator(unittest.TestCase):
    def setUp(self) -> None:
        self.known = {"close", "volume", "high", "low", "open", "returns",
                      "vwap", "adv20", "cap", "sector"}
        self.validator = DeterministicValidator(known_fields=self.known)

    def test_d61_empty_expression_fails(self) -> None:
        report = self.validator.validate("")
        self.assertFalse(report.passed)
        self.assertFalse(report.rule_results["D6.1"])
        self.assertIn("D6.1", str(report.errors))

    def test_d62_no_field_refs_fails(self) -> None:
        report = self.validator.validate("rank(123)")
        self.assertFalse(report.passed)
        self.assertFalse(report.rule_results["D6.2"])

    def test_d63_no_operator_fails(self) -> None:
        report = self.validator.validate("close")
        self.assertFalse(report.passed)
        self.assertFalse(report.rule_results["D6.3"])

    def test_d64_complexity_exceeded(self) -> None:
        many_ops = "rank(close) + rank(close) + rank(close) + rank(close) + rank(close) + rank(close) + rank(close) + rank(close) + rank(close) + rank(close) + rank(close) + rank(close) + rank(close)"
        report = self.validator.validate(many_ops)
        self.assertFalse(report.passed)
        self.assertFalse(report.rule_results["D6.4"])

    def test_d65_unknown_field(self) -> None:
        report = self.validator.validate("rank(zzz_invalid_field)")
        self.assertFalse(report.passed)
        self.assertFalse(report.rule_results["D6.5"])

    def test_d66_unmatched_parens(self) -> None:
        report = self.validator.validate("rank(close, volume")
        self.assertFalse(report.passed)
        self.assertFalse(report.rule_results["D6.6"])

    def test_valid_expression_passes_all(self) -> None:
        expr = "rank(ts_mean(close, 20))"
        report = self.validator.validate(expr)
        self.assertTrue(report.passed, msg=f"errors: {report.errors}")

    def test_all_rules_present(self) -> None:
        expr = "rank(close)"
        report = self.validator.validate(expr)
        for rule in self.validator.RULES:
            self.assertIn(rule, report.rule_results)

    def test_to_dict(self) -> None:
        report = self.validator.validate("rank(close)")
        d = report.to_dict()
        self.assertIn("passed", d)
        self.assertIn("rule_results", d)
        self.assertIn("errors", d)

    def test_from_fields_summary(self) -> None:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        tmp.write(_FIELDS_JSON)
        tmp.close()
        try:
            v = DeterministicValidator.from_fields_summary(Path(tmp.name))
            self.assertGreater(len(v.known_fields), 0)
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

    def test_parse_tokens(self) -> None:
        tokens = self.validator.parse_tokens("rank(ts_mean(close, 20))")
        self.assertIn("rank", tokens)
        self.assertIn("close", tokens)
        self.assertIn("ts_mean", tokens)

    def test_detect_operators(self) -> None:
        tokens = self.validator.parse_tokens("rank(ts_mean(close, 20))")
        ops = self.validator.detect_operators(tokens)
        self.assertIn("rank", ops)
        self.assertIn("ts_mean", ops)

    def test_detect_field_refs(self) -> None:
        tokens = self.validator.parse_tokens("rank(close, volume)")
        refs = self.validator.detect_field_refs(tokens)
        self.assertIn("close", refs)
        self.assertIn("volume", refs)

    def test_check_parentheses_valid(self) -> None:
        self.assertTrue(self.validator.check_parentheses("rank(close)"))

    def test_check_parentheses_invalid(self) -> None:
        self.assertFalse(self.validator.check_parentheses("rank(close"))
        self.assertFalse(self.validator.check_parentheses("rank(close))"))

    def test_valid_operators_set_not_empty(self) -> None:
        self.assertGreater(len(VALID_OPERATORS), 10)


if __name__ == "__main__":
    unittest.main()
