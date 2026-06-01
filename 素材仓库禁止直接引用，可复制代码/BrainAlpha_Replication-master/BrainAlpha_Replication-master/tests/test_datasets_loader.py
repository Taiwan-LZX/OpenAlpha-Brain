import os
import tempfile
import unittest

from alpha_agent.datasets_loader import DatasetsLoader, FieldMetadata


def _make_csv(path: str, rows: list[list[str]]) -> None:
    lines = ['Field,Description,Type,Coverage,Users,Alphas']
    for r in rows:
        quoted = ['"' + c.replace('"', '""') + '"' for c in r]
        lines.append(",".join(quoted))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


class TestDatasetsLoader(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # dataset_a: 2 clean fields
        _make_csv(
            os.path.join(self.tmpdir, "dataset_a_fields_formatted.csv"),
            [
                ["field_alpha", "Alpha description", "MATRIX", "85%", "100", "500"],
                ["field_beta", "Beta description", "VECTOR", "60%", "50", "200"],
            ],
        )
        # dataset_b: 1 field with multi-line description
        _make_csv(
            os.path.join(self.tmpdir, "dataset_b_fields_formatted.csv"),
            [
                [
                    "field_gamma",
                    "Multi-line\ndescription\nhere",
                    "MATRIX",
                    "90%",
                    "30",
                    "100",
                ],
            ],
        )
        # dataset_c: 1 field with 0% coverage
        _make_csv(
            os.path.join(self.tmpdir, "dataset_c_fields_formatted.csv"),
            [
                ["field_delta", "Zero coverage", "GROUP", "0%", "0", "0"],
            ],
        )

    def test_load_counts(self):
        loader = DatasetsLoader(self.tmpdir)
        loader.load()
        self.assertEqual(len(loader.all_metadata()), 4)
        self.assertEqual(len(loader.all_dataset_ids()), 3)

    def test_get_metadata(self):
        loader = DatasetsLoader(self.tmpdir)
        loader.load()
        meta = loader.get_metadata("field_alpha")
        self.assertIsNotNone(meta)
        self.assertEqual(meta.field_id, "field_alpha")
        self.assertEqual(meta.description, "Alpha description")
        self.assertEqual(meta.type, "MATRIX")
        self.assertEqual(meta.coverage, 85.0)
        self.assertEqual(meta.users, 100)
        self.assertEqual(meta.alphas, 500)
        self.assertEqual(meta.dataset_id, "dataset_a")

    def test_get_metadata_nonexistent(self):
        loader = DatasetsLoader(self.tmpdir)
        loader.load()
        self.assertIsNone(loader.get_metadata("nonexistent"))

    def test_get_fields_by_dataset(self):
        loader = DatasetsLoader(self.tmpdir)
        loader.load()
        fields = loader.get_fields_by_dataset("dataset_a")
        self.assertEqual(sorted(fields), ["field_alpha", "field_beta"])
        self.assertEqual(loader.get_fields_by_dataset("nonexistent"), [])

    def test_search_fields_by_keyword(self):
        loader = DatasetsLoader(self.tmpdir)
        loader.load()
        results = loader.search_fields("alpha")
        self.assertTrue(any(m.field_id == "field_alpha" for m in results))
        results2 = loader.search_fields("BETA")
        self.assertTrue(any(m.field_id == "field_beta" for m in results2))

    def test_search_fields_by_description(self):
        loader = DatasetsLoader(self.tmpdir)
        loader.load()
        results = loader.search_fields("multi")
        self.assertTrue(any(m.field_id == "field_gamma" for m in results))

    def test_search_fields_empty_keyword(self):
        loader = DatasetsLoader(self.tmpdir)
        loader.load()
        results = loader.search_fields("")
        # all fields match empty string (score > 0 due to id/desc matching)
        self.assertEqual(len(results), 4)

    def test_skip_invalid_field_names(self):
        _make_csv(
            os.path.join(self.tmpdir, "dataset_d_fields_formatted.csv"),
            [
                ["123_invalid", "Starts with digit", "MATRIX", "50%", "1", "1"],
                ["valid_field", "Valid name", "MATRIX", "50%", "1", "1"],
            ],
        )
        loader = DatasetsLoader(self.tmpdir)
        loader.load()
        self.assertIsNone(loader.get_metadata("123_invalid"))
        self.assertIsNotNone(loader.get_metadata("valid_field"))

    def test_zero_coverage(self):
        loader = DatasetsLoader(self.tmpdir)
        loader.load()
        meta = loader.get_metadata("field_delta")
        self.assertEqual(meta.coverage, 0.0)

    def test_all_dataset_ids(self):
        loader = DatasetsLoader(self.tmpdir)
        loader.load()
        self.assertEqual(loader.all_dataset_ids(), ["dataset_a", "dataset_b", "dataset_c"])

    def test_lazy_load(self):
        loader = DatasetsLoader(self.tmpdir)
        meta = loader.get_metadata("field_alpha")  # triggers load()
        self.assertIsNotNone(meta)
        self.assertIn("field_beta", loader.all_field_ids())

    def test_field_id_uniqueness(self):
        loader = DatasetsLoader(self.tmpdir)
        loader.load()
        # all field_ids should be unique
        ids = list(loader.all_metadata().keys())
        self.assertEqual(len(ids), len(set(ids)))

    def test_all_metadata_immutable_copy(self):
        loader = DatasetsLoader(self.tmpdir)
        loader.load()
        d1 = loader.all_metadata()
        d2 = loader.all_metadata()
        self.assertEqual(len(d1), len(d2))
        d1["new_key"] = None  # should not affect loader
        self.assertNotIn("new_key", loader.all_metadata())

    def test_parse_coverage_malformed(self):
        _make_csv(
            os.path.join(self.tmpdir, "dataset_e_fields_formatted.csv"),
            [
                ["field_epsilon", "Bad coverage", "MATRIX", "abc", "1", "1"],
            ],
        )
        loader = DatasetsLoader(self.tmpdir)
        loader.load()
        meta = loader.get_metadata("field_epsilon")
        self.assertEqual(meta.coverage, 0.0)

    def test_parse_users_alphas_malformed(self):
        _make_csv(
            os.path.join(self.tmpdir, "dataset_f_fields_formatted.csv"),
            [
                ["field_zeta", "Bad ints", "MATRIX", "50%", "not_a_number", "also_bad"],
            ],
        )
        loader = DatasetsLoader(self.tmpdir)
        loader.load()
        meta = loader.get_metadata("field_zeta")
        self.assertEqual(meta.users, 0)
        self.assertEqual(meta.alphas, 0)


if __name__ == "__main__":
    unittest.main()
