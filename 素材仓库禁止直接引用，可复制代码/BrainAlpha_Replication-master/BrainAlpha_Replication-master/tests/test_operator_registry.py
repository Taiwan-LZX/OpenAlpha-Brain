import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alpha_agent.operator_registry import OperatorRegistry, get_operator_registry


class TestOperatorRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reg = get_operator_registry()

    def test_load_operators(self) -> None:
        names = self.reg.get_valid_names()
        self.assertGreaterEqual(len(names), 60)
        for name in ["rank", "ts_mean", "group_neutralize", "zscore", "abs", "log"]:
            self.assertIn(name, names)
        for name in names:
            op = self.reg.get_operator(name)
            self.assertIsNotNone(op, f"Operator {name} returned None")
            self.assertIn("name", op)
            self.assertIn("definition", op)
            self.assertIn("description", op)
            self.assertIn("category", op)

    def test_get_valid_names(self) -> None:
        names = self.reg.get_valid_names()
        self.assertGreaterEqual(len(names), 60)
        self.assertIn("rank", names)
        self.assertIn("ts_mean", names)
        self.assertIn("group_neutralize", names)

    def test_get_operator_with_doc(self) -> None:
        op = self.reg.get_operator("ts_corr")
        self.assertIsNotNone(op)
        self.assertEqual(op["name"], "ts_corr")
        self.assertIn("definition", op)
        self.assertIn("ts_corr(", op["definition"])
        self.assertIn("doc_text", op)
        self.assertGreater(len(op["doc_text"]), 10)

    def test_get_operator_no_doc(self) -> None:
        op = self.reg.get_operator("and")
        self.assertIsNotNone(op)
        self.assertEqual(op["name"], "and")
        self.assertIn("definition", op)
        self.assertIn("doc_text", op)
        self.assertEqual(op["doc_text"], "")

    def test_get_by_category(self) -> None:
        ts_ops = self.reg.get_by_category("Time Series")
        self.assertGreaterEqual(len(ts_ops), 15)
        ts_names = {op["name"] for op in ts_ops}
        self.assertIn("ts_mean", ts_names)
        self.assertIn("ts_delta", ts_names)
        self.assertIn("ts_rank", ts_names)

        arith_ops = self.reg.get_by_category("Arithmetic")
        self.assertGreaterEqual(len(arith_ops), 8)
        arith_names = {op["name"] for op in arith_ops}
        self.assertIn("abs", arith_names)
        self.assertIn("log", arith_names)

    def test_get_categories(self) -> None:
        cats = self.reg.get_categories()
        self.assertIn("Time Series", cats)
        self.assertIn("Arithmetic", cats)
        self.assertIn("Cross Sectional", cats)
        self.assertIn("Logical", cats)

    def test_get_relevant(self) -> None:
        results = self.reg.get_relevant("correlation")
        self.assertGreaterEqual(len(results), 1)
        names = {op["name"] for op in results}
        self.assertTrue(names & {"ts_corr", "ts_covariance", "ts_regression"})

    def test_doc_text_no_html(self) -> None:
        op = self.reg.get_operator("ts_corr")
        doc = op["doc_text"]
        self.assertNotIn("<p>", doc)
        self.assertNotIn("<ul>", doc)
        self.assertNotIn("<li>", doc)
        self.assertNotIn("<b>", doc)
        self.assertNotIn("</p>", doc)
        self.assertNotIn("</ul>", doc)
        self.assertNotIn("</li>", doc)

    def test_existing_operators_subset(self) -> None:
        old_hardcoded = {
            "rank", "zscore", "scale", "winsorize", "group_neutralize",
            "ts_sum", "ts_mean", "ts_std_dev", "ts_delta", "ts_min", "ts_max",
            "ts_argmax", "ts_argmin", "ts_rank", "ts_product", "ts_correlation",
            "ts_covariance", "ts_regression", "ts_quantile",
            "ts_zscore",
            "group_rank", "group_mean", "group_zscore",
            "decay_linear",
            "log", "abs", "sign", "sqrt", "power",
            "max", "min", "if_else",
        }
        official = self.reg.get_valid_names()
        old_that_exist = old_hardcoded & official
        self.assertGreaterEqual(len(old_that_exist), 20)
        for name in ["rank", "zscore", "ts_mean", "ts_std_dev", "log", "abs", "max", "min"]:
            self.assertIn(name, official, f"Core operator {name} missing from official set")
        official_contains_new = official - old_hardcoded
        self.assertGreaterEqual(len(official_contains_new), 20)


class TestOperatorRegistryFallback(unittest.TestCase):
    def test_fallback_no_operators_dir(self) -> None:
        with patch("alpha_agent.operator_registry._OPERATORS_JSON", Path(tempfile.gettempdir()) / "_nonexistent_ops.json"):
            with patch("alpha_agent.operator_registry._DOCS_DIR", Path(tempfile.gettempdir()) / "_nonexistent_docs"):
                reg = OperatorRegistry()
                self.assertEqual(len(reg.get_valid_names()), 0)
                self.assertIsNone(reg.get_operator("rank"))
                self.assertEqual(reg.get_by_category("Time Series"), [])
                self.assertEqual(reg.format_operators_for_prompt(["rank"]), "")


if __name__ == "__main__":
    unittest.main()
