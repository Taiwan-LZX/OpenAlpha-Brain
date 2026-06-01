from __future__ import annotations

import json
import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any

import numpy as np

from openalpha_brain.utils.algo_logger import Timer, algo_log, log_call
from openalpha_brain.utils.paper_edge_enhancements import compute_structural_novelty_score

logger = logging.getLogger(__name__)

_DIRECTIONS = ["momentum", "mean_reversion", "volatility", "statistical", "volume", "interaction"]
_TIME_HORIZONS = ["short", "medium", "long"]
_MECHANISMS = ["signal", "normalized", "conditional", "interaction"]

DIRECTION_ENUM = ["momentum", "reversal", "value", "growth", "volatility", "liquidity"]
TIME_HORIZON_ENUM = ["short", "medium", "long"]
MECHANISM_ENUM = ["cross_sectional", "time_series", "statistical", "fundamental"]


def behavior_to_index(direction: str, time_horizon: str, mechanism: str) -> tuple[int, int, int]:
    return (
        DIRECTION_ENUM.index(direction),
        TIME_HORIZON_ENUM.index(time_horizon),
        MECHANISM_ENUM.index(mechanism),
    )


@dataclass
class FeatureCell:
    direction: str = ""
    time_horizon: str = ""
    mechanism: str = ""
    best_expr: str = ""
    best_fitness: float = 0.0
    best_sharpe: float = 0.0
    best_turnover: float | None = None
    update_count: int = 0
    elites: list[dict] = dataclass_field(default_factory=list)
    decay_state: str = "active"
    admission_paused: bool = False

    def _update_best_from_elites(self) -> None:
        if not self.elites:
            return
        best = max(self.elites, key=lambda e: e.get("fitness", 0.0))
        self.best_expr = best.get("expr", "")
        self.best_fitness = best.get("fitness", 0.0)
        self.best_sharpe = best.get("sharpe", 0.0)
        self.best_turnover = best.get("turnover")

    def worst_elite_fitness(self) -> float:
        if not self.elites:
            return -float("inf")
        return min(e.get("fitness", 0.0) for e in self.elites)


@dataclass
class StrategyFeatures:
    direction: str = "momentum"
    time_horizon: str = "medium"
    mechanism: str = "signal"


class GridArchive:
    def __init__(self,
                 dims: tuple[int, int, int] = (6, 3, 4),
                 elite_capacity: int = 3,
                 behavior_ranges: list[tuple[float, float]] | None = None):
        self._dims = dims
        self._elite_capacity = elite_capacity
        self._behavior_ranges = behavior_ranges
        with Timer("GridArchive.init"):
            self._occupied: np.ndarray = np.zeros(dims, dtype=bool)
            self._best_fitness: np.ndarray = np.full(dims, -np.inf, dtype=np.float32)
            self._solution_count: np.ndarray = np.zeros(dims, dtype=np.int32)
            self._cells: dict[tuple[int, ...], FeatureCell] = {}
            self._max_observed_fitness: float = 0.0

    @algo_log()
    def add(self, behavior: tuple[int, int, int], fitness: float,
            metadata: dict) -> bool:
        with Timer("GridArchive.add"):
            if not self._valid_behavior(behavior):
                return False
            cell = self._get_or_create_cell(behavior)
            if len(cell.elites) < self._elite_capacity:
                admitted = True
            elif fitness > cell.worst_elite_fitness():
                admitted = True
            else:
                return False
            new_elite = {
                "expr": metadata.get("expr", ""),
                "fitness": fitness,
                "sharpe": metadata.get("sharpe", 0.0),
                "turnover": metadata.get("turnover"),
                "added_at": time.time(),
                "behavior": behavior,
            }
            if len(cell.elites) >= self._elite_capacity:
                removed = min(cell.elites, key=lambda e: e.get("fitness", 0.0))
                cell.elites.remove(removed)
            cell.elites.append(new_elite)
            cell.elites.sort(key=lambda e: e.get("fitness", 0.0), reverse=True)
            cell.update_count += 1
            cell._update_best_from_elites()
            idx = tuple(behavior)
            self._occupied[idx] = True
            self._best_fitness[idx] = cell.best_fitness
            self._solution_count[idx] = len(cell.elites)
            if fitness > self._max_observed_fitness:
                self._max_observed_fitness = fitness
            return True

    @algo_log()
    def batch_add(self, behaviors: np.ndarray, fitnesses: np.ndarray,
                  metadatas: list[dict]) -> int:
        with Timer("GridArchive.batch_add"):
            n_accepted = 0
            for i in range(len(behaviors)):
                behavior = tuple(behaviors[i])
                if self.add(behavior, float(fitnesses[i]), metadatas[i]):
                    n_accepted += 1
            return n_accepted

    @algo_log()
    def sample_elite(self, k: int = 1, method: str = "uniform") -> list[dict]:
        with Timer("GridArchive.sample_elite"):
            all_elites = []
            for cell in self._cells.values():
                for elite in cell.elites:
                    elite_copy = dict(elite)
                    elite_copy["cell_behavior"] = (
                        DIRECTION_ENUM[cell.direction] if isinstance(cell.direction, int) else cell.direction,
                        TIME_HORIZON_ENUM[cell.time_horizon] if isinstance(cell.time_horizon, int) else cell.time_horizon,
                        MECHANISM_ENUM[cell.mechanism] if isinstance(cell.mechanism, int) else cell.mechanism,
                    )
                    all_elites.append(elite_copy)
            if not all_elites:
                return []
            if method == "uniform":
                selected = random.sample(all_elites, min(k, len(all_elites)))
            elif method == "fitness_proportional":
                weights = [max(e.get("fitness", 0.0), 0.01) for e in all_elites]
                total = sum(weights)
                weights = [w / total for w in weights]
                indices = random.choices(range(len(all_elites)), weights=weights, k=k)
                selected = [all_elites[i] for i in indices]
            elif method == "novelty":
                selected = random.sample(all_elites, min(k, len(all_elites)))
            else:
                selected = random.sample(all_elites, min(k, len(all_elites)))
            return selected

    def as_array(self) -> dict[str, np.ndarray]:
        return {
            "occupied": self._occupied.copy(),
            "best_fitness": self._best_fitness.copy(),
            "solution_count": self._solution_count.copy(),
        }

    @property
    def qd_score(self) -> float:
        occupied_fitness = self._best_fitness[self._occupied]
        if len(occupied_fitness) == 0:
            return 0.0
        return float(np.sum(occupied_fitness))

    @property
    def coverage(self) -> float:
        return float(np.sum(self._occupied)) / max(self.n_cells, 1)

    @property
    def max_fitness(self) -> float:
        if not np.any(self._occupied):
            return 0.0
        return float(np.max(self._best_fitness[self._occupied]))

    @property
    def normalized_qd_score(self) -> float:
        max_possible = self._max_observed_fitness if self._max_observed_fitness > 0 else 1.0
        return self.qd_score / (max_possible * self.n_cells)

    @property
    def n_occupied(self) -> int:
        return int(np.sum(self._occupied))

    @property
    def n_cells(self) -> int:
        dims = self._dims
        result = 1
        for d in dims:
            result *= d
        return result

    def get_empty_cells(self) -> list[tuple[int, ...]]:
        empty_indices = np.where(~self._occupied)
        return list(zip(*empty_indices))

    def get_frontier_cells(self) -> list[tuple[int, ...]]:
        frontier = []
        for idx in np.argwhere(self._occupied):
            idx_tuple = tuple(idx)
            neighbors = self._get_neighbors(idx_tuple)
            has_empty_neighbor = any(not self._occupied[n] for n in neighbors)
            if has_empty_neighbor:
                frontier.append(idx_tuple)
        return frontier

    def get_cell(self, behavior: tuple[int, ...]) -> FeatureCell | None:
        return self._cells.get(tuple(behavior))

    @algo_log()
    def save_state(self, path: str) -> None:
        with Timer("GridArchive.save_state"):
            arr_data = self.as_array()
            np.savez_compressed(path, **arr_data)
            meta = {
                "dims": self._dims,
                "elite_capacity": self._elite_capacity,
                "max_observed_fitness": self._max_observed_fitness,
                "cells": {},
            }
            for k, cell in self._cells.items():
                meta["cells"][str(k)] = {
                    "direction": cell.direction,
                    "time_horizon": cell.time_horizon,
                    "mechanism": cell.mechanism,
                    "best_expr": cell.best_expr,
                    "best_fitness": cell.best_fitness,
                    "best_sharpe": cell.best_sharpe,
                    "update_count": cell.update_count,
                    "elites": cell.elites,
                    "decay_state": cell.decay_state,
                    "admission_paused": cell.admission_paused,
                }
            meta_path = path.rsplit(".", 1)[0] + "_meta.json"
            Path(meta_path).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    @algo_log()
    def load_state(cls, path: str) -> GridArchive:
        with Timer("GridArchive.load_state"):
            meta_path = path.rsplit(".", 1)[0] + "_meta.json"
            meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
            archive = cls(
                dims=tuple(meta["dims"]),
                elite_capacity=meta["elite_capacity"],
            )
            archive._max_observed_fitness = meta.get("max_observed_fitness", 0.0)
            arr_data = np.load(path)
            archive._occupied = arr_data["occupied"]
            archive._best_fitness = arr_data["best_fitness"]
            archive._solution_count = arr_data["solution_count"]
            for k_str, cell_data in meta.get("cells", {}).items():
                k = eval(k_str)
                cell = FeatureCell(
                    direction=cell_data["direction"],
                    time_horizon=cell_data["time_horizon"],
                    mechanism=cell_data["mechanism"],
                    best_expr=cell_data.get("best_expr", ""),
                    best_fitness=cell_data.get("best_fitness", 0.0),
                    best_sharpe=cell_data.get("best_sharpe", 0.0),
                    update_count=cell_data.get("update_count", 0),
                    elites=cell_data.get("elites", []),
                    decay_state=cell_data.get("decay_state", "active"),
                    admission_paused=cell_data.get("admission_paused", False),
                )
                archive._cells[k] = cell
            return archive

    def _valid_behavior(self, behavior: tuple[int, ...]) -> bool:
        for i, b in enumerate(behavior):
            if b < 0 or b >= self._dims[i]:
                return False
        return True

    def _get_or_create_cell(self, behavior: tuple[int, ...]) -> FeatureCell:
        key = tuple(behavior)
        if key not in self._cells:
            dir_idx, time_idx, mech_idx = behavior
            self._cells[key] = FeatureCell(
                direction=DIRECTION_ENUM[dir_idx],
                time_horizon=TIME_HORIZON_ENUM[time_idx],
                mechanism=MECHANISM_ENUM[mech_idx],
            )
        return self._cells[key]

    def _get_neighbors(self, idx: tuple[int, ...]) -> list[tuple[int, ...]]:
        neighbors = []
        for dim in range(len(self._dims)):
            for delta in (-1, 1):
                neighbor = list(idx)
                neighbor[dim] += delta
                neighbor = tuple(neighbor)
                if self._valid_behavior(neighbor):
                    neighbors.append(neighbor)
        return neighbors

    @property
    def dims(self) -> tuple[int, int, int]:
        return self._dims

    def get_occupied_cells(self) -> list[tuple[int, int, int]]:
        indices = np.argwhere(self._occupied)
        return [tuple(map(int, idx)) for idx in indices]

    def get_occupied_cells_with_fitness(self) -> list[tuple[tuple[int, int, int], float]]:
        result = []
        indices = np.argwhere(self._occupied)
        for idx in indices:
            key = tuple(map(int, idx))
            fit = float(self._best_fitness[key])
            result.append((key, fit))
        return result

    def get_low_fitness_cells(self, threshold: float = 0.3) -> list[tuple[int, int, int]]:
        occupied = self.get_occupied_cells()
        return [cell for cell in occupied if self._best_fitness[cell] < threshold]


class FeatureMap:
    ELITE_CAPACITY: int = 3
    EXPLORE_WEIGHT: float = 0.6
    EXPLORE_WEIGHT_MIN: float = 0.3
    EXPLOIT_BONUS_PER_ELITE: float = 0.1

    def __init__(self, path: str = "feature_map.json"):
        self._path = Path(path)
        self._archive = GridArchive(
            dims=(len(_DIRECTIONS), len(_TIME_HORIZONS), len(_MECHANISMS)),
            elite_capacity=self.ELITE_CAPACITY,
        )
        self._lock = threading.Lock()
        self._generation: int = 0
        self._cells: dict[str, FeatureCell] = {}
        self._init_cells()
        self._load()

    @property
    def archive(self) -> GridArchive:
        return self._archive

    @property
    def qd_score(self) -> float:
        return self._archive.qd_score

    @property
    def coverage(self) -> float:
        return self._archive.coverage

    @property
    def normalized_qd_score(self) -> float:
        return self._archive.normalized_qd_score

    def _cell_key(self, direction: str, time_horizon: str, mechanism: str) -> str:
        return f"{direction}__{time_horizon}__{mechanism}"

    def _init_cells(self) -> None:
        for d in _DIRECTIONS:
            for t in _TIME_HORIZONS:
                for m in _MECHANISMS:
                    key = self._cell_key(d, t, m)
                    self._cells[key] = FeatureCell(direction=d, time_horizon=t, mechanism=m)

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            for key, cell_data in data.get("cells", {}).items():
                if key not in self._cells:
                    continue
                cell = self._cells[key]
                cell.best_expr = cell_data.get("best_expr", "")
                cell.best_fitness = cell_data.get("best_fitness", 0.0)
                cell.best_sharpe = cell_data.get("best_sharpe", 0.0)
                cell.best_turnover = cell_data.get("best_turnover")
                cell.update_count = cell_data.get("update_count", 0)
                cell.elites = cell_data.get("elites", [])
                cell.decay_state = cell_data.get("decay_state", "active")
                cell.admission_paused = cell_data.get("admission_paused", False)
            logger.info("FeatureMap: loaded %d cells from %s", len(self._cells), self._path)
        except OSError:
            logger.warning("FeatureMap: failed to load", exc_info=True)

    def _save(self) -> None:
        try:
            cells_data = {}
            for k, v in self._cells.items():
                cells_data[k] = {
                    "direction": v.direction,
                    "time_horizon": v.time_horizon,
                    "mechanism": v.mechanism,
                    "best_expr": v.best_expr,
                    "best_fitness": v.best_fitness,
                    "best_sharpe": v.best_sharpe,
                    "best_turnover": v.best_turnover,
                    "update_count": v.update_count,
                    "elites": v.elites,
                    "decay_state": v.decay_state,
                    "admission_paused": v.admission_paused,
                }
            data = {"cells": cells_data, "generation": self._generation}
            self._path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError:
            logger.warning("FeatureMap: failed to save", exc_info=True)

    def advance_generation(self) -> int:
        self._generation += 1
        return self._generation

    def mark_cell_decay(self, direction: str, time_horizon: str, mechanism: str,
                        decay_level: str) -> None:
        """Two-level decay response for MAP-Elites cells.

        L3 (observing): pause admission but keep elites for potential recovery
        L4 (critical): clear all elites and blacklist permanently
        """
        key = self._cell_key(direction, time_horizon, mechanism)
        with self._lock:
            cell = self._cells.get(key)
            if cell is None:
                return

            if decay_level == "L3_DIR_LIGHT":
                if cell.decay_state == "blacklisted":
                    return
                cell.decay_state = "observing"
                cell.admission_paused = True
                logger.warning(
                    "FeatureMap: cell %s -> L3 observing (paused admission, %d elites preserved)",
                    key, len(cell.elites),
                )
            elif decay_level == "L4_DIR_HEAVY":
                cell.decay_state = "blacklisted"
                cell.admission_paused = True
                cell.elites.clear()
                cell.best_expr = ""
                cell.best_fitness = 0.0
                cell.best_sharpe = 0.0
                cell.best_turnover = None
                logger.warning(
                    "FeatureMap: cell %s -> L4 blacklisted (elites cleared)",
                    key,
                )
            self._save()

    def add_candidate(
        self,
        expr: str,
        features: StrategyFeatures,
        fitness_score: float,
        sharpe: float = 0.0,
        turnover: float | None = None,
    ) -> bool:
        key = self._cell_key(features.direction, features.time_horizon, features.mechanism)
        with self._lock:
            cell = self._cells.get(key)
            if cell is None:
                return False

            if cell.decay_state == "blacklisted":
                return False

            if cell.admission_paused and cell.decay_state == "observing":
                return False

            worst_fitness = cell.worst_elite_fitness()
            if len(cell.elites) < self.ELITE_CAPACITY:
                admitted = True
            elif fitness_score > worst_fitness:
                admitted = True
            else:
                admitted = False

            if not admitted:
                return False

            new_elite = {
                "expr": expr,
                "fitness": fitness_score,
                "sharpe": sharpe,
                "turnover": turnover,
                "added_at": time.time(),
            }

            if len(cell.elites) >= self.ELITE_CAPACITY:
                removed = min(cell.elites, key=lambda e: e.get("fitness", 0.0))
                cell.elites.remove(removed)
                logger.info(
                    "FeatureMap: cell %s evicted elite fitness=%.4f, admitted new fitness=%.4f",
                    key, removed.get("fitness", 0.0), fitness_score,
                )

            cell.elites.append(new_elite)
            cell.elites.sort(key=lambda e: e.get("fitness", 0.0), reverse=True)
            cell.update_count += 1
            cell._update_best_from_elites()

            try:
                dir_idx = _DIRECTIONS.index(features.direction)
                time_idx = _TIME_HORIZONS.index(features.time_horizon)
                mech_idx = _MECHANISMS.index(features.mechanism)
                behavior = (dir_idx, time_idx, mech_idx)
                self._archive.add(behavior, fitness_score, {
                    "expr": expr,
                    "sharpe": sharpe,
                    "turnover": turnover,
                })
            except ValueError:
                pass

            self._save()

            logger.info(
                "FeatureMap: cell %s admitted elite #%d fitness=%.4f sharpe=%.4f",
                key, len(cell.elites), fitness_score, sharpe,
            )
            return True

    @algo_log()
    def batch_add_candidates(self, candidates: list[dict]) -> int:
        with Timer("FeatureMap.batch_add_candidates"):
            n_accepted = 0
            for cand in candidates:
                expr = cand.get("expr", "")
                direction = cand.get("direction", "momentum")
                time_horizon = cand.get("time_horizon", "medium")
                mechanism = cand.get("mechanism", "signal")
                fitness = cand.get("fitness", 0.0)
                sharpe = cand.get("sharpe", 0.0)
                turnover = cand.get("turnover")
                features = StrategyFeatures(
                    direction=direction,
                    time_horizon=time_horizon,
                    mechanism=mechanism,
                )
                if self.add_candidate(expr, features, fitness, sharpe, turnover):
                    n_accepted += 1
            return n_accepted

    def get_explore_exploit_schedule(self) -> dict:
        """Compute adaptive explore/exploit schedule based on MAP-Elites coverage.

        Returns a dict with:
        - strategy: "explore" | "exploit"
        - explore_weight: float (0.0~1.0), probability of preferring empty cells
        - coverage: current grid coverage ratio
        - unexplored_count: number of empty cells
        - elite_density: avg elites per filled cell
        """
        with self._lock:
            total = len(self._cells)
            active_cells = [
                c for c in self._cells.values()
                if c.decay_state not in ("blacklisted",)
            ]
            filled = sum(1 for c in active_cells if c.elites)
            coverage = filled / max(total, 1)

        elite_count = sum(len(c.elites) for c in active_cells if c.elites)
        elite_density = elite_count / max(filled, 1)

        if coverage < self.EXPLORE_WEIGHT_MIN:
            explore_weight = self.EXPLORE_WEIGHT
        elif coverage < self.EXPLORE_WEIGHT:
            explore_weight = self.EXPLORE_WEIGHT - (coverage - self.EXPLORE_WEIGHT_MIN)
        else:
            explore_weight = max(0.05, 1.0 - coverage)

        elite_density_bonus = min(0.15, elite_density * self.EXPLOIT_BONUS_PER_ELITE)
        explore_weight = max(0.05, explore_weight - elite_density_bonus)

        strategy = "explore" if explore_weight >= 0.5 else "exploit"

        return {
            "strategy": strategy,
            "explore_weight": round(explore_weight, 4),
            "coverage": round(coverage, 4),
            "unexplored_count": total - filled,
            "elite_density": round(elite_density, 4),
        }

    def sample_parent(self, exclude_key: str | None = None) -> FeatureCell | None:
        """Fitness-proportional parent selection from MAP-Elites grid.

        Uses the best elite fitness per cell as selection weight.
        Skips blacklisted cells.
        """
        with self._lock:
            non_empty = [
                cell for key, cell in self._cells.items()
                if cell.best_expr and key != exclude_key
                and cell.decay_state != "blacklisted"
            ]
        if not non_empty:
            return None
        weights = [max(cell.best_fitness, 0.01) for cell in non_empty]
        total = sum(weights)
        if total == 0:
            return random.choice(non_empty)
        weights = [w / total for w in weights]
        return random.choices(non_empty, weights=weights, k=1)[0]

    def sample_elite(self, direction: str | None = None) -> dict | None:
        """Sample a single elite entry from the MAP-Elites grid.

        Can optionally filter by direction.
        Uses fitness-proportional sampling across all elites.
        """
        with self._lock:
            all_elites: list[tuple[dict, FeatureCell]] = []
            for cell in self._cells.values():
                if cell.decay_state == "blacklisted":
                    continue
                if direction and cell.direction != direction:
                    continue
                for elite in cell.elites:
                    all_elites.append((elite, cell))

        if not all_elites:
            return None

        weights = [max(e[0].get("fitness", 0.0), 0.01) for e in all_elites]
        total = sum(weights)
        if total == 0:
            return random.choice(all_elites)[0]
        weights = [w / total for w in weights]
        return random.choices([e[0] for e in all_elites], weights=weights, k=1)[0]

    def sample_distant_parent(self, current_key: str, n: int = 1) -> list[FeatureCell]:
        with self._lock:
            parts = current_key.split("__") if current_key else []
            distant = []
            for key, cell in self._cells.items():
                if not cell.best_expr:
                    continue
                if key == current_key:
                    continue
                if cell.decay_state == "blacklisted":
                    continue
                k_parts = key.split("__")
                diff_count = sum(1 for a, b in zip(parts, k_parts) if a != b)
                if diff_count >= 2:
                    distant.append(cell)
        if not distant:
            with self._lock:
                distant = [c for k, c in self._cells.items()
                           if c.best_expr and k != current_key
                           and c.decay_state != "blacklisted"]
        if not distant:
            return []
        return random.sample(distant, min(n, len(distant)))

    def get_diversity_stats(self) -> dict:
        with self._lock:
            total = len(self._cells)
            active_cells = [
                c for c in self._cells.values()
                if c.decay_state != "blacklisted"
            ]
            filled = sum(1 for c in active_cells if c.elites)
        direction_coverage: dict[str, float] = {}
        for d in _DIRECTIONS:
            d_cells = [c for k, c in self._cells.items() if c.direction == d]
            d_active = [c for c in d_cells if c.decay_state != "blacklisted"]
            d_filled = sum(1 for c in d_active if c.elites)
            direction_coverage[d] = round(d_filled / len(d_cells), 2) if d_cells else 0.0

        filled_exprs = [c.best_expr for c in active_cells if c.best_expr]
        avg_novelty = 0.0
        if len(filled_exprs) >= 2:
            novelty_scores = [
                compute_structural_novelty_score(expr, [e for e in filled_exprs if e != expr])
                for expr in filled_exprs
            ]
            avg_novelty = round(sum(novelty_scores) / len(novelty_scores), 4) if novelty_scores else 0.0

        decay_cells = {
            "observing": sum(1 for c in self._cells.values() if c.decay_state == "observing"),
            "blacklisted": sum(1 for c in self._cells.values() if c.decay_state == "blacklisted"),
        }

        return {
            "total_cells": total,
            "filled_cells": filled,
            "coverage": round(filled / total, 4) if total else 0.0,
            "direction_coverage": direction_coverage,
            "avg_structural_novelty": avg_novelty,
            "decay_cells": decay_cells,
        }

    def get_cell(self, direction: str, time_horizon: str, mechanism: str) -> FeatureCell | None:
        key = self._cell_key(direction, time_horizon, mechanism)
        with self._lock:
            return self._cells.get(key)

    def get_cell_elites(self, direction: str, time_horizon: str,
                        mechanism: str) -> list[dict]:
        key = self._cell_key(direction, time_horizon, mechanism)
        with self._lock:
            cell = self._cells.get(key)
            if cell is None:
                return []
            return list(cell.elites)

    def get_unexplored_directions(self) -> list[str]:
        with self._lock:
            unexplored = []
            for d in _DIRECTIONS:
                d_cells = [c for k, c in self._cells.items() if c.direction == d]
                active = [c for c in d_cells if c.decay_state != "blacklisted"]
                if not any(c.elites for c in active):
                    unexplored.append(d)
        return unexplored

    def get_explore_targets(self, top_k: int = 3) -> list[dict]:
        """Return top-k empty cells prioritized for exploration."""
        with self._lock:
            empty = []
            for key, cell in self._cells.items():
                if cell.decay_state == "blacklisted":
                    continue
                if cell.admission_paused:
                    continue
                if not cell.elites:
                    empty.append({
                        "key": key,
                        "direction": cell.direction,
                        "time_horizon": cell.time_horizon,
                        "mechanism": cell.mechanism,
                    })

        random.shuffle(empty)
        return empty[:top_k]

    def get_frontier_targets(self, top_k: int = 3) -> list[dict]:
        frontier_indices = self._archive.get_frontier_cells()
        targets = []
        for idx in frontier_indices[:top_k]:
            dir_idx, time_idx, mech_idx = idx
            targets.append({
                "behavior_index": idx,
                "direction": _DIRECTIONS[dir_idx] if dir_idx < len(_DIRECTIONS) else DIRECTION_ENUM[dir_idx],
                "time_horizon": _TIME_HORIZONS[time_idx] if time_idx < len(_TIME_HORIZONS) else TIME_HORIZON_ENUM[time_idx],
                "mechanism": _MECHANISMS[mech_idx] if mech_idx < len(_MECHANISMS) else MECHANISM_ENUM[mech_idx],
            })
        return targets


@dataclass
class EmitterOutput:
    target_behavior: tuple[int, int, int]
    mutation_hints: dict


class BaseEmitter(ABC):
    """MAP-Elites Emitter 抽象基类。"""

    @abstractmethod
    def ask(self, archive: GridArchive, n: int = 1) -> list[EmitterOutput]:
        ...

    @abstractmethod
    def tell(self, archive: GridArchive, behaviors: list[tuple],
             fitnesses: list[float], metadatas: list[dict]) -> None:
        ...


class ExploreEmitter(BaseEmitter):
    """面向未探索区域的 Emitter。

    [DEFENSIVE_LOG] ExploreEmitter 的职责是选择「目标行为坐标」(target_behavior)，
    而非生成 alpha 表达式。实际的表达式生成由上游调度器完成：
      1. 根据 target_behavior 对应的 (direction, time_horizon, mechanism)
      2. 从 TemplateFamilyBandit 的未访问 arm 中选取模板
      3. 由模板 + 字段代理生成具体表达式

    绝不允许 ExploreEmitter 自行随机拼接算子/字段生成表达式，
    那样会产生非法组合（如 ts_rank(volume_field, close_price) 的语义错误）。
    """

    def __init__(self, exploration_rate: float = 0.5,
                 boundary_preference: float = 0.7):
        self.exploration_rate = exploration_rate
        self.boundary_preference = boundary_preference
        self._recent_targets: list[tuple[int, int, int]] = []

    @algo_log()
    def ask(self, archive: GridArchive, n: int = 1) -> list[EmitterOutput]:
        with Timer("ExploreEmitter.ask"):
            empty_cells = archive.get_empty_cells()
            occupied_cells = archive.get_occupied_cells()

            if not empty_cells:
                fallback = random.sample(occupied_cells, min(n, len(occupied_cells)))
                return [
                EmitterOutput(
                    target_behavior=cell,
                    mutation_hints={"strategy": "explore", "direction_bias": "fallback"},
                )
                for cell in fallback
            ]

            boundary_cells = self._find_boundary_cells(empty_cells, occupied_cells)
            n_boundary = int(n * self.boundary_preference)
            n_boundary = min(n_boundary, len(boundary_cells))
            n_random = n - n_boundary

            selected: list[tuple[int, int, int]] = []
            if n_boundary > 0 and boundary_cells:
                selected.extend(random.sample(boundary_cells, min(n_boundary, len(boundary_cells))))
            if n_random > 0:
                remaining = [c for c in empty_cells if c not in selected]
                if remaining:
                    selected.extend(random.sample(remaining, min(n_random, len(remaining))))

            while len(selected) < n and empty_cells:
                extra = random.choice(empty_cells)
                if extra not in selected:
                    selected.append(extra)

            self._recent_targets = selected[:]

            logger.info(
                "ExploreEmitter.ask: 选择 %d 个目标 (boundary=%d, random=%d), "
                "提示上游使用 TemplateFamilyBandit 未访问 arm",
                len(selected), n_boundary, n_random,
            )

            return [
                EmitterOutput(
                    target_behavior=cell,
                    mutation_hints={
                        "strategy": "explore",
                        "direction_bias": "large",
                        "field_family_suggest": "novel",
                        "template_source": "unvisited_bandit_arm",
                    },
                )
                for cell in selected[:n]
            ]

    @algo_log()
    def tell(self, archive: GridArchive, behaviors: list[tuple],
             fitnesses: list[float], metadatas: list[dict]) -> None:
        with Timer("ExploreEmitter.tell"):
            logger.info(
                "ExploreEmitter: 收到 %d 个反馈结果, avg_fitness=%.4f",
                len(behaviors),
                sum(fitnesses) / max(len(fitnesses), 1),
            )

    def _find_boundary_cells(
        self,
        empty_cells: list[tuple[int, int, int]],
        occupied_cells: list[tuple[int, int, int]],
    ) -> list[tuple[int, int, int]]:
        boundary = []
        occupied_set = set(occupied_cells)
        for cell in empty_cells:
            d, t, m = cell
            neighbors = [
                (d + dx, t + dy, m + dz)
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                for dz in (-1, 0, 1)
                if (dx, dy, dz) != (0, 0, 0)
            ]
            if any(n in occupied_set for n in neighbors):
                boundary.append(cell)
        return boundary


class ExploitEmitter(BaseEmitter):
    """面向高适应度区域的 Emitter。

    [DEFENSIVE_LOG] ExploitEmitter 的采样策略合法性说明：
      - 在 behavior space (direction, time_horizon, mechanism) 中做邻域偏移是安全的，
        因为这是 MAP-Elites grid 的坐标空间，不是表达式空间
      - softmax 权重基于已有 elite 的 fitness（来自 WQ 真实 Sharpe），
        这是对「哪个行为区域更可能产出好因子」的粗粒度估计
      - _neighbor_offset 限制在 ±1 grid unit 内，确保不会跳到远距离无关区域
      - ⚠️ 不生成新表达式，仅建议目标区域。实际参数微调由 DecayParameterTuner 负责
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self._selection_history: list[tuple[int, int, int]] = []

    @algo_log()
    def ask(self, archive: GridArchive, n: int = 1) -> list[EmitterOutput]:
        with Timer("ExploitEmitter.ask"):
            occupied = archive.get_occupied_cells_with_fitness()
            if not occupied:
                dims = archive.dims
                return [
                    EmitterOutput(
                        target_behavior=(
                            random.randint(0, dims[0] - 1),
                            random.randint(0, dims[1] - 1),
                            random.randint(0, dims[2] - 1),
                        ),
                        mutation_hints={"strategy": "exploit", "direction_bias": "fallback"},
                    )
                    for _ in range(n)
                ]

            cells = [c for c, _ in occupied]
            fitnesses = np.array([f for _, f in occupied], dtype=np.float64)

            if self.temperature > 0:
                scaled = fitnesses / max(self.temperature, 1e-8)
                scaled -= scaled.max()
                probs = np.exp(scaled)
                probs /= probs.sum()
            else:
                probs = np.ones(len(cells)) / len(cells)

            indices = np.random.choice(len(cells), size=min(n, len(cells)), replace=True, p=probs)
            results: list[EmitterOutput] = []
            for idx in indices:
                base = cells[idx]
                offset = self._neighbor_offset(base, archive.dims)
                self._selection_history.append(base)
                results.append(EmitterOutput(
                    target_behavior=offset,
                    mutation_hints={
                        "strategy": "exploit",
                        "direction_bias": "small",
                        "field_family_suggest": "refine",
                        "block_c_tuning": "use_decay_parameter_tuner",
                    },
                ))

            logger.info(
                "ExploitEmitter.ask: 从 %d 个已占用 cell 中采样 %d 个邻域目标, temperature=%.2f",
                len(cells), len(results), self.temperature,
            )
            return results

    @algo_log()
    def tell(self, archive: GridArchive, behaviors: list[tuple],
             fitnesses: list[float], metadatas: list[dict]) -> None:
        with Timer("ExploitEmitter.tell"):
            logger.info(
                "ExploitEmitter: 收到 %d 个反馈结果, avg_fitness=%.4f",
                len(behaviors),
                sum(fitnesses) / max(len(fitnesses), 1),
            )

    def _neighbor_offset(
        self,
        base: tuple[int, int, int],
        dims: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        d_off = random.randint(-1, 1)
        t_off = random.randint(-1, 1)
        m_off = random.randint(-1, 1)
        return (
            max(0, min(dims[0] - 1, base[0] + d_off)),
            max(0, min(dims[1] - 1, base[1] + t_off)),
            max(0, min(dims[2] - 1, base[2] + m_off)),
        )


ExploitTemperature = ExploitEmitter


PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    'decay_window': (5.0, 30.0),
    'decay_weight': (0.5, 1.0),
    'rank_threshold': (0.0, 1.0),
}
_EPSILON = 1e-6


class CMAEvolutionStrategy:
    """CMA-ES 适配器 — 仅用于 Block C（衰减段）连续参数微调。

    [DEFENSIVE_LOG] WQ 平台返回 PASS/FAIL 离散反馈 + Sharpe/Turnover 等少数指标。
    这是一个无梯度、高噪音、离散的黑箱环境。CMA-ES 在此环境下只能用于
    **低维连续参数空间的微调**（Block C 的 decay_window / decay_weight / rank_threshold），
    绝不能用于算子组合或字段选择等离散变量的进化搜索。
    """

    def __init__(self, sigma: float = 0.5,
                 population_size: int = 8):
        self._param_names = list(PARAMETER_BOUNDS.keys())
        self.dim = len(self._param_names)
        self.sigma = sigma
        self.population_size = population_size
        bounds_arr = np.array([PARAMETER_BOUNDS[k] for k in self._param_names], dtype=np.float64)
        self._lower = bounds_arr[:, 0]
        self._upper = bounds_arr[:, 1]
        self.mean = (self._lower + self._upper) / 2.0
        self.C = np.eye(self.dim) * ((self._upper - self._lower) ** 2 / 16.0)
        self.generation = 0
        self._ps = np.zeros(self.dim)
        self._pc = np.zeros(self.dim)
        self._cc = (4.0 + self.dim / self.dim) / (self.dim + 4.0 + 2.0 * self.dim / self.dim)
        self._cs = (self.dim + 2.0) / (3.0 * self.dim + 6.0)
        self._c1 = 2.0 / ((self.dim + 1.3) ** 2 + self._mu_eff())
        self._cmu = min(1.0 - self._c1, 2.0 * (self._mu_eff() - 2.0 + 1.0 / self._mu_eff()) / ((self.dim + 2.0) ** 2 + self._mu_eff()))
        self._damps = 1.0 + 2.0 * max(0.0, np.sqrt((self._mu_eff() - 1.0) / max(self.dim, 1)) - 1.0) + self._cs

    def _mu_eff(self) -> float:
        return max(self.population_size / 2.0, _EPSILON)

    @algo_log()
    def ask(self) -> dict[str, float]:
        with Timer("CMAEvolutionStrategy.ask"):
            try:
                L = np.linalg.cholesky(self.C)
            except np.linalg.LinAlgError:
                L = np.eye(self.dim) * _EPSILON
            z = np.random.randn(1, self.dim)
            raw_sample = self.mean + self.sigma * (z @ L.T)[0]
            clipped = np.clip(raw_sample, self._lower + _EPSILON, self._upper - _EPSILON)
            result = {
                k: float(clipped[i])
                for i, k in enumerate(self._param_names)
            }
            logger.info(
                "CMAEvolutionStrategy.ask: gen=%d, sample=%s",
                self.generation,
                {k: round(v, 4) for k, v in result.items()},
            )
            return result

    @algo_log()
    def tell(self, params: dict[str, float], fitness: float) -> None:
        with Timer("CMAEvolutionStrategy.tell"):
            vector = np.array([params.get(k, self.mean[i]) for i, k in enumerate(self._param_names)], dtype=np.float64)
            old_mean = self.mean.copy()
            self.mean = vector
            mean_delta = self.mean - old_mean
            self._ps = (1 - self._cs) * self._ps + \
                       np.sqrt(self._cs * (2 - self._cs) * self._mu_eff()) * \
                       (np.linalg.solve(np.linalg.cholesky(self.C + _EPSILON * np.eye(self.dim)).T, mean_delta / max(self.sigma, _EPSILON)) if self.sigma > _EPSILON else np.zeros(self.dim))

            ps_norm_denom = np.sqrt(max(1.0 - (1 - self._cs) ** (2 * (self.generation + 1)), _EPSILON))
            ps_norm = np.linalg.norm(self._ps) / ps_norm_denom
            threshold = (1.4 + 2.0 / (max(self.dim, 1) + 1)) * np.sqrt(max(self.dim, 1))
            hsig = float(ps_norm < threshold)

            self._pc = (1 - self._cc) * self._pc + \
                       hsig * np.sqrt(self._cc * (2 - self._cc) * self._mu_eff()) * (mean_delta / max(self.sigma, _EPSILON))

            rank_one = np.outer(self._pc, self._pc)
            y = (vector - old_mean) / max(self.sigma, _EPSILON)
            rank_mu = np.outer(y, y)

            self.C = (1 - self._c1 - self._cmu) * self.C + self._c1 * rank_one + self._cmu * rank_mu
            self.C = np.clip(self.C, -1e6, 1e6)
            self.C = (self.C + self.C.T) / 2.0 + _EPSILON * np.eye(self.dim)

            self.sigma *= np.exp((np.linalg.norm(self._ps) / self._dim_expected_norm() - 1) * (self._cs / self._damps))
            self.sigma = max(self.sigma, _EPSILON)
            self.sigma = min(self.sigma, (self._upper[0] - self._lower[0]) / 2.0)
            self.generation += 1
            logger.info(
                "CMAEvolutionStrategy.tell: gen=%d, sigma=%.6f, fitness=%.4f, params=%s",
                self.generation, round(self.sigma, 6), fitness,
                {k: round(v, 4) for k, v in params.items()},
            )

    def _dim_expected_norm(self) -> float:
        return np.sqrt(max(self.dim, 1)) * (1 - 1.0 / (4 * max(self.dim, 1)) + 1.0 / (21 * max(self.dim, 1) ** 2))


class DecayParameterTuner:
    """Block C 连续参数微调器（替代原 CMAEmitter）。

    [DEFENSIVE_LOG] 此类不再继承 BaseEmitter，不再生成 EmitterOutput。
    它的唯一职责是：在 Block C 的低维连续参数空间中，
    利用 CMA-ES 根据真实 WQ 平台反馈（Sharpe 等）进行参数微调。

    用法：
        tuner = DecayParameterTuner()
        params = tuner.ask()       # -> {'decay_window': 12.3, 'decay_weight': 0.75, ...}
        # ... 将 params 注入 ThreeBlockTemplate.assemble() 并提交 WQ ...
        tuner.tell(params, sharpe)  # 用 WQ 返回的 Sharpe 更新 CMA 分布
    """

    def __init__(self, cma: CMAEvolutionStrategy | None = None):
        self._cma = cma or CMAEvolutionStrategy()
        self._history: list[dict[str, Any]] = []
        self._consecutive_no_improvement: int = 0
        self._best_fitness: float = -float("inf")
        self._best_params: dict[str, float] | None = None

    @algo_log()
    def ask(self) -> dict[str, float]:
        """仅生成 Block C 参数候选集，不生成算子组合或字段选择。"""
        with Timer("DecayParameterTuner.ask"):
            params = self._cma.ask()
            self._history.append({"params": params.copy(), "stage": "ask"})
            logger.info(
                "DecayParameterTuner.ask: proposed params=%s",
                {k: round(v, 4) for k, v in params.items()},
            )
            return params

    @algo_log()
    def tell(self, params: dict[str, float], fitness: float) -> None:
        """接收 WQ 平台真实反馈（Sharpe 等），更新 CMA 分布。"""
        with Timer("DecayParameterTuner.tell"):
            if fitness > self._best_fitness + _EPSILON:
                self._consecutive_no_improvement = 0
                self._best_fitness = fitness
                self._best_params = params.copy()
            else:
                self._consecutive_no_improvement += 1

            self._cma.tell(params, fitness)
            self._history.append({
                "params": params.copy(),
                "fitness": fitness,
                "stage": "tell",
                "gen": self._cma.generation,
            })
            logger.info(
                "DecayParameterTuner.tell: fitness=%.4f, best=%.4f, no_improve=%d",
                fitness, self._best_fitness, self._consecutive_no_improvement,
            )

    @property
    def best_params(self) -> dict[str, float] | None:
        return self._best_params

    @property
    def best_fitness(self) -> float:
        return self._best_fitness

    @property
    def is_stagnant(self) -> bool:
        return self._consecutive_no_improvement >= 10

    @property
    def generation(self) -> int:
        return self._cma.generation


CMAEmitter = DecayParameterTuner


class IsoLineDirectionCalculator:
    """计算 MAP-Elites 的 IsoLine（等值线）方向。"""

    def __init__(self):
        self._cached_direction: np.ndarray | None = None
        self._cache_generation: int = -1

    @algo_log()
    def compute_direction(self, archive: GridArchive) -> np.ndarray:
        with Timer("IsoLineDirectionCalculator.compute_direction"):
            arr_dict = archive.as_array()
            arr = arr_dict.get("best_fitness", None)
            if arr is None or arr.size == 0:
                return np.array([1.0, 0.5, 0.3])

            occupied_mask = arr_dict.get("occupied", np.zeros_like(arr, dtype=bool))
            n_occupied = int(np.sum(occupied_mask))
            if n_occupied < 3:
                center = np.array([
                    (archive.dims[0] - 1) / 2.0,
                    (archive.dims[1] - 1) / 2.0,
                    (archive.dims[2] - 1) / 2.0,
                ])
                occupied = archive.get_occupied_cells_with_fitness()
                if not occupied:
                    return np.array([1.0, 0.5, 0.3])
                weighted_dir = np.zeros(3)
                total_w = 0.0
                for cell, fit in occupied:
                    vec = np.array(cell, dtype=np.float64) - center
                    w = max(fit, 0.01)
                    weighted_dir += w * vec
                    total_w += w
                if total_w > 0:
                    weighted_dir /= total_w
                norm = np.linalg.norm(weighted_dir)
                if norm < 1e-10:
                    weighted_dir = np.array([1.0, 0.5, 0.3])
                    norm = np.linalg.norm(weighted_dir)
                self._cached_direction = weighted_dir / norm
                return self._cached_direction

            safe_arr = np.where(occupied_mask, arr, np.nan)
            grad_d, grad_t, grad_m = np.gradient(safe_arr, edge_order=1)
            grad_d = np.nan_to_num(grad_d, nan=0.0)
            grad_t = np.nan_to_num(grad_t, nan=0.0)
            grad_m = np.nan_to_num(grad_m, nan=0.0)

            empty_mask = ~occupied_mask
            if not np.any(empty_mask):
                total_grad_d = float(np.sum(grad_d[occupied_mask]))
                total_grad_t = float(np.sum(grad_t[occupied_mask]))
                total_grad_m = float(np.sum(grad_m[occupied_mask]))
            else:
                total_grad_d = float(np.sum(np.where(empty_mask, grad_d, 0.0)[empty_mask])) if np.any(empty_mask) else 0.0
                total_grad_t = float(np.sum(np.where(empty_mask, grad_t, 0.0)[empty_mask])) if np.any(empty_mask) else 0.0
                total_grad_m = float(np.sum(np.where(empty_mask, grad_m, 0.0)[empty_mask])) if np.any(empty_mask) else 0.0

            direction = np.array([total_grad_d, total_grad_t, total_grad_m], dtype=np.float64)
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                direction = np.array([1.0, 0.5, 0.3], dtype=np.float64)
                norm = np.linalg.norm(direction)
            direction /= norm
            self._cached_direction = direction
            self._cache_generation = getattr(archive, 'generation', -1)
            logger.info(
                "IsoLineDirectionCalculator: 方向向量=%s",
                np.round(direction, 4).tolist(),
            )
            return direction

    @algo_log()
    def suggest_target(self, archive: GridArchive,
                       n: int = 1) -> list[tuple[int, int, int]]:
        with Timer("IsoLineDirectionCalculator.suggest_target"):
            direction = self.compute_direction(archive)
            empty_cells = archive.get_empty_cells()
            low_fitness_cells = archive.get_low_fitness_cells(threshold=0.3)
            candidates = list(set(empty_cells) | set(low_fitness_cells))
            if not candidates:
                occupied = archive.get_occupied_cells()
                if occupied:
                    return random.sample(occupied, min(n, len(occupied)))
                dims = archive.dims
                return [(0, 0, 0)] * min(n, 1)

            center = np.array([
                (archive.dims[0] - 1) / 2.0,
                (archive.dims[1] - 1) / 2.0,
                (archive.dims[2] - 1) / 2.0,
            ])

            scored: list[tuple[float, tuple[int, int, int]]] = []
            for cell in candidates:
                cell_vec = np.array(cell, dtype=np.float64)
                to_cell = cell_vec - center
                proj = np.dot(to_cell, direction)
                dist = np.linalg.norm(to_cell)
                score = proj / max(dist, 0.1)
                scored.append((score, cell))

            scored.sort(key=lambda x: x[0], reverse=True)
            targets = [cell for _, cell in scored[:n]]
            logger.info(
                "IsoLineDirectionCalculator: 推荐 %d 个目标, top_score=%.3f",
                len(targets), scored[0][0] if scored else 0,
            )
            return targets


class EmitterOrchestrator:
    """管理多个 Emitter 的调度和权重分配。

    [DEFENSIVE_LOG] ⚠️ UCB 权重在离散反馈下的局限性警告：
      WQ 平台仅返回 PASS/FAIL + Sharpe/Turnover，这是一个高噪音离散黑箱。
      UCB 公式 (avg_reward + c * sqrt(log(N)/n)) 假设 reward 有统计意义，
      但在 PASS/FAIL 离散反馈下：
        - avg_reward 只是 Sharpe 的粗糙代理，不是真实梯度信号
        - exploration_bonus 项只能做粗粒度调度（"这个 emitter 最近产出还行"）
        - **绝不能**将 UCB 权重解读为"该方向有更高梯度"
      正确理解：UCB 在此处仅用于避免某个 emitter 饿死或垄断，
      真正的优化能力来自 TemplateFamilyBandit（离散 arm 选择）+ DecayParameterTuner（连续参数微调）
    """

    _STAGNANT_THRESHOLD: int = 5

    def __init__(self):
        self.emitters: dict[str, BaseEmitter] = {}
        self.weights: dict[str, float] = {}
        self._rewards: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._positive_counts: dict[str, int] = {}
        self._last_positive_gen: dict[str, int] = {}
        self.history: list[dict] = []
        self._ucb_c: float = 1.0
        self._total_rounds: int = 0

    @algo_log()
    def register(self, name: str, emitter: BaseEmitter,
                 weight: float = 1.0) -> None:
        self.emitters[name] = emitter
        self.weights[name] = weight
        self._rewards[name] = 0.0
        self._counts[name] = 0
        self._positive_counts[name] = 0
        self._last_positive_gen[name] = 0
        logger.info(
            "EmitterOrchestrator: 注册 emitter '%s', 初始权重=%.2f",
            name, weight,
        )

    @algo_log()
    def ask(self, archive: GridArchive, n: int = 1) -> list[EmitterOutput]:
        with Timer("EmitterOrchestrator.ask"):
            if not self.emitters:
                return []
            names = list(self.emitters.keys())
            w = np.array([self.weights.get(name, 1.0) for name in names], dtype=np.float64)
            total_w = w.sum()
            if total_w <= 0:
                w = np.ones(len(names)) / len(names)
            else:
                w /= total_w
            quotas = (w * n + 0.5).astype(int)
            diff = n - quotas.sum()
            if diff != 0:
                idx = np.random.choice(len(names), p=w)
                quotas[idx] += diff

            all_outputs: list[EmitterOutput] = []
            for name, quota in zip(names, quotas):
                if quota <= 0:
                    continue
                emitter = self.emitters[name]
                try:
                    outputs = emitter.ask(archive, n=quota)
                    for out in outputs:
                        out.mutation_hints["_emitter_name"] = name
                    all_outputs.extend(outputs)
                    self._counts[name] = self._counts.get(name, 0) + quota
                except (OSError, ValueError, RuntimeError) as e:
                    logger.warning(
                        "EmitterOrchestrator: emitter '%s' ask 失败: %s",
                        name, e,
                    )

            random.shuffle(all_outputs)
            logger.info(
                "EmitterOrchestrator: 分配 %d 个配额, 返回 %d 个输出",
                n, len(all_outputs),
            )
            return all_outputs

    @algo_log()
    def tell_all(self, archive: GridArchive, behaviors: list[tuple],
                 fitnesses: list[float], metadatas: list[dict]) -> None:
        with Timer("EmitterOrchestrator.tell_all"):
            self._total_rounds += 1
            grouped: dict[str, list[tuple[tuple, float, dict]]] = {}
            for beh, fit, meta in zip(behaviors, fitnesses, metadatas):
                emitter_name = meta.get("_emitter_name", "unknown") if isinstance(meta, dict) else "unknown"
                if emitter_name not in grouped:
                    grouped[emitter_name] = []
                grouped[emitter_name].append((beh, fit, meta))

            for name, items in grouped.items():
                emitter = self.emitters.get(name)
                if emitter is None:
                    continue
                b_items = [it[0] for it in items]
                f_items = [it[1] for it in items]
                m_items = [it[2] for it in items]
                try:
                    emitter.tell(archive, b_items, f_items, m_items)
                    avg_fit = sum(f_items) / max(len(f_items), 1)
                    self._rewards[name] = self._rewards.get(name, 0.0) + avg_fit
                    n_positive = sum(1 for f in f_items if f > 0.0)
                    self._positive_counts[name] = self._positive_counts.get(name, 0) + n_positive
                    if n_positive > 0:
                        self._last_positive_gen[name] = self._total_rounds
                except (OSError, ValueError, RuntimeError) as e:
                    logger.warning(
                        "EmitterOrchestrator: emitter '%s' tell 失敗: %s",
                        name, e,
                    )

            logger.info(
                "EmitterOrchestrator: 已反馈给 %d 个 emitters (round=%d)",
                len(grouped), self._total_rounds,
            )

    @algo_log()
    def adapt_weights(self, archive: GridArchive,
                      recent_qd_scores: list[float]) -> None:
        with Timer("EmitterOrchestrator.adapt_weights"):
            total_count = sum(self._counts.values())
            if total_count == 0:
                return
            new_weights: dict[str, float] = {}
            for name in self.emitters:
                count = self._counts.get(name, 1)
                reward = self._rewards.get(name, 0.0)
                avg_reward = reward / count
                exploration_bonus = self._ucb_c * np.sqrt(
                    np.log(max(total_count, 1)) / max(count, 1)
                )
                ucb_value = avg_reward + exploration_bonus

                since_positive = self._total_rounds - self._last_positive_gen.get(name, self._total_rounds)

                if since_positive >= self._STAGNANT_THRESHOLD:

                    decay_factor = 0.5 ** (since_positive - self._STAGNANT_THRESHOLD + 1)
                    ucb_value *= decay_factor
                    logger.warning(
                        "[DEFENSIVE_LOG] EmitterOrchestrator: emitter '%s' 已 %d 轮无正向反馈, "
                        "权重衰减 factor=%.4f (UCB前=%.4f → 衰减后=%.4f)",
                        name, since_positive, decay_factor,
                        avg_reward + exploration_bonus, ucb_value,
                    )

                new_weights[name] = max(0.001, ucb_value)

            total_new = sum(new_weights.values())
            if total_new > 0:
                for name in new_weights:
                    self.weights[name] = new_weights[name] / total_new

            self.history.append({
                "qd_scores": recent_qd_scores[-5:] if recent_qd_scores else [],
                "weights": dict(self.weights),
                "counts": dict(self._counts),
                "positive_counts": dict(self._positive_counts),
                "round": self._total_rounds,
            })
            logger.info(
                "EmitterOrchestrator: 权重自适应完成 (round=%d), weights=%s",
                self._total_rounds,
                {k: round(v, 3) for k, v in self.weights.items()},
            )
