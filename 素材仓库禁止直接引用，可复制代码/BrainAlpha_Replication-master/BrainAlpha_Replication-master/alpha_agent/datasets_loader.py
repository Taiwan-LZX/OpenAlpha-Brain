from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set


_SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass
class FieldMetadata:
    field_id: str
    description: str
    type: str
    coverage: float
    users: int
    alphas: int
    dataset_id: str


EMPTY_META = FieldMetadata(
    field_id="",
    description="",
    type="",
    coverage=0.0,
    users=0,
    alphas=0,
    dataset_id="",
)


class DatasetsLoader:
    def __init__(self, datasets_dir: str | Path) -> None:
        self._datasets_dir = Path(datasets_dir)
        self._metadata: dict[str, FieldMetadata] = {}
        self._dataset_fields: dict[str, list[str]] = {}
        self._loaded = False

    def load(self) -> None:
        self._metadata.clear()
        self._dataset_fields.clear()
        csv_files = sorted(self._datasets_dir.glob("*_fields_formatted.csv"))
        for fpath in csv_files:
            dataset_id = fpath.stem.replace("_fields_formatted", "")
            with fpath.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    raw_field = (row.get("Field") or "").strip()
                    if not raw_field or not _SNAKE_CASE_RE.match(raw_field):
                        continue
                    coverage_str = ((row.get("Coverage") or "0%").strip()).rstrip("%")
                    try:
                        coverage = float(coverage_str) if coverage_str else 0.0
                    except ValueError:
                        coverage = 0.0
                    try:
                        users = int(row.get("Users") or 0)
                    except (ValueError, TypeError):
                        users = 0
                    try:
                        alphas = int(row.get("Alphas") or 0)
                    except (ValueError, TypeError):
                        alphas = 0
                    meta = FieldMetadata(
                        field_id=raw_field,
                        description=(row.get("Description") or "").strip(),
                        type=(row.get("Type") or "").strip(),
                        coverage=coverage,
                        users=users,
                        alphas=alphas,
                        dataset_id=dataset_id,
                    )
                    self._metadata[raw_field] = meta
                    self._dataset_fields.setdefault(dataset_id, []).append(raw_field)
        self._loaded = True

    def get_metadata(self, field_id: str) -> Optional[FieldMetadata]:
        if not self._loaded:
            self.load()
        return self._metadata.get(field_id)

    def get_fields_by_dataset(self, dataset_id: str) -> List[str]:
        if not self._loaded:
            self.load()
        return list(self._dataset_fields.get(dataset_id, []))

    def search_fields(self, keyword: str, top_k: int = 20) -> List[FieldMetadata]:
        if not self._loaded:
            self.load()
        lower = keyword.lower()
        results: list[tuple[FieldMetadata, int]] = []
        for meta in self._metadata.values():
            score = 0
            if lower in meta.field_id.lower():
                score += 10
            if lower in meta.description.lower():
                score += 5
            if meta.type.lower() == lower:
                score += 3
            if score > 0:
                results.append((meta, score))
        results.sort(key=lambda x: -x[1])
        return [m for m, _ in results[:top_k]]

    def all_field_ids(self) -> Set[str]:
        if not self._loaded:
            self.load()
        return set(self._metadata.keys())

    def all_metadata(self) -> Dict[str, FieldMetadata]:
        if not self._loaded:
            self.load()
        return dict(self._metadata)

    def all_dataset_ids(self) -> List[str]:
        if not self._loaded:
            self.load()
        return sorted(self._dataset_fields.keys())
