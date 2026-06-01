from __future__ import annotations

import logging
import math
import random
from collections import defaultdict

import numpy as np

from openalpha_brain.utils.algo_logger import algo_log
from openalpha_brain.utils.paper_edge_enhancements import compute_adaptive_simulation_budget

logger = logging.getLogger(__name__)


class _MCTSNode:
    """MCTS 樹節點，代表搜索空間中的一個決策點。

    在 alpha 挖掘語境下，每個節點代表一個「結構決策」：
      - 根節點：起始狀態（無表達式）
      - 中間節點：已選擇 root_op / 已選擇主要 ts_op / 已選擇欄位組合 ...
      - 葉節點：完整表達式候選
    """

    __slots__ = (
        "children",
        "decision",
        "decision_type",
        "depth",
        "expr_hint",
        "parent",
        "reward",
        "untried_actions",
        "visits",
    )

    def __init__(
        self,
        decision: str = "",
        decision_type: str = "root",
        parent: _MCTSNode | None = None,
        untried_actions: list[str] | None = None,
        expr_hint: str | None = None,
        depth: int = 0,
    ):
        self.decision = decision
        self.decision_type = decision_type
        self.parent = parent
        self.children: list[_MCTSNode] = []
        self.visits: int = 0
        self.reward: float = 0.0
        self.untried_actions: list[str] = untried_actions if untried_actions is not None else []
        self.expr_hint = expr_hint
        self.depth = depth

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    @property
    def is_terminal(self) -> bool:
        return self.decision_type == "full_expr"

    def get_path_to_root(self) -> list[str]:
        path = []
        node: _MCTSNode | None = self
        while node is not None and node.decision:
            path.append(node.decision)
            node = node.parent
        return list(reversed(path))


class SemanticMutator:
    ROOT_OPS = ["rank", "zscore", "scale", "sign", "abs"]
    TS_OPS = ["ts_delta", "ts_mean", "ts_decay_linear", "ts_std_dev", "ts_corr", "ts_rank", "ts_skewness"]
    FIELDS = ["close", "volume", "returns", "earnings", "revenue", "vwap", "high", "low", "open", "sharesout"]
    WINDOWS = ["5", "10", "20", "60", "120", "250"]
    COMPOSITIONS = ["ratio", "multiplicative", "additive"]

    def __init__(self, embed_fn=None, llm_generate_fn=None):
        self._embed_fn = embed_fn
        self._llm_generate_fn = llm_generate_fn
        self._path_history: dict[str, int] = defaultdict(int)

    def interpolate_embeddings(
        self,
        emb_a: list[float],
        emb_b: list[float],
        alpha: float = 0.5,
    ) -> list[float]:
        """Interpolate two embedding vectors via linear combination.
        NOTE: The interpolated vector is NOT decoded back to expression space.
        It is only used to generate a natural language prompt that guides the LLM
        toward blending aspects of both parent expressions. This is prompt
        decoration, not true semantic-space mutation."""
        a = np.array(emb_a, dtype=np.float64)
        b = np.array(emb_b, dtype=np.float64)
        result = (1 - alpha) * a + alpha * b
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
        return result.tolist()

    async def _compute_embedding(self, text: str) -> list[float] | None:
        if self._embed_fn is None:
            return None
        try:
            vec = await self._embed_fn(text)
            if isinstance(vec, list):
                if len(vec) > 0 and isinstance(vec[0], list):
                    return vec[0]
                return vec
            if isinstance(vec, np.ndarray):
                return vec.tolist()
        except (ValueError, TypeError, OSError):
            logger.warning("SemanticMutator: embed failed for text")
        return None

    async def decode_to_expression(
        self,
        parent_a_expr: str,
        parent_b_expr: str,
        interpolation_ratio: float = 0.5,
        direction: str = "",
    ) -> str | None:
        if self._llm_generate_fn is None:
            return None

        interpolation_context = ""
        if self._embed_fn is not None:
            try:
                emb_a = await self._compute_embedding(parent_a_expr)
                emb_b = await self._compute_embedding(parent_b_expr)
                if emb_a is not None and emb_b is not None:
                    interpolated = self.interpolate_embeddings(emb_a, emb_b, alpha=interpolation_ratio)
                    interpolation_context = (
                        "\n\nAdditional semantic guidance: The interpolated semantic vector "
                        "suggests combining aspects of both parents. The blend ratio indicates "
                        f"the new expression should lean {interpolation_ratio * 100:.0f}% toward "
                        f"Expression A's semantic characteristics and {(1 - interpolation_ratio) * 100:.0f}% "
                        "toward Expression B's. Focus on the shared semantic dimensions while "
                        "preserving the unique signal from each parent."
                    )
                    try:
                        from openalpha_brain.core.loop_state import _algo_tick
                        _algo_tick("semantic_interpolation")
                    except (ImportError, AttributeError):
                        pass
            except (OSError, ValueError, RuntimeError):
                logger.warning("SemanticMutator: interpolation context generation failed, proceeding without")

        prompt = (
            "You are an alpha factor engineer. Create a NEW WorldQuant BRAIN FastExpr expression "
            "that combines the financial logic of these two successful alpha expressions:\n\n"
            f"Expression A: {parent_a_expr}\n"
            f"Expression B: {parent_b_expr}\n\n"
            f"The new expression should be approximately {interpolation_ratio * 100:.0f}% influenced "
            f"by Expression A and {(1 - interpolation_ratio) * 100:.0f}% by Expression B.\n"
            f"Direction: {direction}\n\n"
            "Rules:\n"
            "- Must use group_neutralize() as the outermost function\n"
            "- Must be valid FastExpr syntax\n"
            "- Must combine the CORE IDEAS of both expressions, not just concatenate them\n"
            "- Output ONLY the expression, nothing else"
            + interpolation_context
        )

        try:
            response = await self._llm_generate_fn(prompt)
        except (ConnectionError, OSError, TimeoutError) as exc:
            logger.warning("decode_to_expression: LLM generation failed: %s", exc)
            return None

        if not response:
            return None

        expr = response.strip()
        if expr.startswith("```"):
            lines = expr.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            expr = "\n".join(lines).strip()

        return expr if expr else None

    def _get_child_actions(self, decision_type: str) -> list[str]:
        """根據決策類型返回可用的子動作空間"""
        action_map = {
            "root": self.ROOT_OPS,
            "root_op": self.TS_OPS,
            "ts_op": self.FIELDS,
            "field": self.WINDOWS,
            "window": self.COMPOSITIONS,
            "composition": ["finalize"],
        }
        return action_map.get(decision_type, [])

    def _get_next_decision_type(self, current_type: str) -> str:
        type_order = {
            "root": "root_op",
            "root_op": "ts_op",
            "ts_op": "field",
            "field": "window",
            "window": "composition",
            "composition": "full_expr",
        }
        return type_order.get(current_type, "full_expr")

    def _mcts_select(self, node: _MCTSNode, c: float = 1.41) -> _MCTSNode:
        """UCB1 選擇階段：從根節點出發，沿 UCB1 值最大的路徑走到葉節點。

        UCB1公式: UCB1(child) = Q(child)/N(child) + c * sqrt(ln(N(parent))/N(child))

        其中：
            Q(child) = child.reward / max(1, child.visits)  (平均獎值)
            N(child) = child.visits
            c = 1.41 (標準 UCB1 constant，平衡探索與利用)

        Args:
            node: 起始節點（通常是根節點）
            c: 探索常數，較大值傾向探索未知區域

        Returns:
            最值得進一步探索的葉節點（可能是未完全展開的節點）
        """
        while node.is_fully_expanded and not node.is_terminal and node.children:
            best_child = None
            best_ucb = float("-inf")
            log_parent_visits = math.log(max(1, node.visits))

            for child in node.children:
                if child.visits == 0:
                    ucb_value = float("inf")
                else:
                    exploitation = child.reward / child.visits
                    exploration = c * math.sqrt(log_parent_visits / child.visits)
                    ucb_value = exploitation + exploration

                if ucb_value > best_ucb:
                    best_ucb = ucb_value
                    best_child = child

            if best_child is None:
                break

            node = best_child

        return node

    def _mcts_expand(
        self,
        node: _MCTSNode,
        feature_map=None,
    ) -> _MCTSNode | None:
        """展開階段：為選中的葉節點添加一個子節點。

        邏輯：
        1. 從 node.untried_actions 中取出一個動作
        2. 建立新的子節點
        3. 根據 decision_type 初始化子節點的 untried_actions
        4. 如果 feature_map 可用，優先展開低覆蓋率的動作（引導探索未開墾區域）

        Args:
            node: 要展開的葉節點
            feature_map: FeatureMap 實例（可選，用於引導探索）

        Returns:
            新展開的子節點，或 None（如果無法展開）
        """
        if not node.untried_actions:
            return None

        action = node.untried_actions.pop(0)
        next_type = self._get_next_decision_type(node.decision_type)
        next_actions = [] if next_type == "full_expr" else self._get_child_actions(next_type)

        if feature_map is not None and len(node.untried_actions) > 0 and random.random() < 0.3:
            low_coverage_actions = []
            for remaining_action in node.untried_actions:
                coverage_score = self._estimate_action_coverage(action, feature_map)
                if coverage_score < 0.5:
                    low_coverage_actions.append((coverage_score, remaining_action))

            if low_coverage_actions:
                low_coverage_actions.sort(key=lambda x: x[0])
                chosen_action_data = low_coverage_actions[0][1]
                node.untried_actions.insert(0, action)
                action = node.untried_actions.pop(node.untried_actions.index(chosen_action_data))
                next_actions = [] if next_type == "full_expr" else self._get_child_actions(next_type)

        new_hint = f"{node.expr_hint} → {action}" if node.expr_hint else action
        child_node = _MCTSNode(
            decision=f"{node.decision_type}={action}" if node.decision_type != "root" else f"root_op={action}",
            decision_type=next_type,
            parent=node,
            untried_actions=next_actions[:],
            expr_hint=new_hint,
            depth=node.depth + 1,
        )

        node.children.append(child_node)
        return child_node

    def _estimate_action_coverage(self, action: str, feature_map) -> float:
        """估計某個動作在 FeatureMap 中的覆蓋率（啟發式）"""
        try:
            stats = feature_map.get_diversity_stats()
            base_coverage = stats.get("coverage", 0.0)
            filled_cells = stats.get("filled_cells", 0)
            total_cells = stats.get("total_cells", 1)

            action_lower = action.lower()
            direction_bonus = 0.0
            if any(kw in action_lower for kw in ["delta", "rank", "momentum"]):
                direction_bonus += 0.15
            elif any(kw in action_lower for kw in ["mean", "decay", "reversion"]):
                direction_bonus += 0.12
            elif any(kw in action_lower for kw in ["std_dev", "volatility", "skewness"]):
                direction_bonus += 0.10

            coverage = min(1.0, base_coverage + direction_bonus)
            return coverage
        except (ValueError, TypeError):
            pass

    def _mcts_simulate(
        self,
        node: _MCTSNode,
        direction: str = "",
        known_good_exprs: list[str] | None = None,
    ) -> float:
        """模擬/評估階段：從當前節點快速估計潛在價值。

        使用結構啟發式評分（方案 A）：
        - 基於到目前為止的決策路徑計算啟發式分數
        - 分數組成：
          * 多樣性加成：此路徑在歷史中少見 → 高分
          * 特徵匹配：與 direction 一致 → 高分
          * 複雜度適中（depth 3-5）→ 高分，太淺或太深 → 扣分
          * 已知好模式匹配 → 加分

        Args:
            node: 當前節點
            direction: 探索方向（momentum/mean_reversion/value/volatility 等）
            known_good_exprs: 已知高 Sharpe 表達式列表（用於模式匹配）

        Returns:
            模擬獎值（float，通常 0.0~1.0）
        """
        path = node.get_path_to_root()
        path_key = "|".join(path)
        self._path_history[path_key] += 1
        path_frequency = self._path_history[path_key]

        diversity_score = 1.0 / (1.0 + path_frequency)

        direction_keywords = {
            "momentum": ["rank", "delta", "ts_delta"],
            "mean_reversion": ["zscore", "scale", "ts_mean", "decay"],
            "value": ["earnings", "revenue", "sharesout"],
            "volatility": ["std_dev", "ts_std_dev", "skewness"],
            "volume": ["volume", "vwap"],
        }

        direction_match = 0.0
        if direction:
            keywords = direction_keywords.get(direction.lower(), [])
            matching_decisions = sum(1 for p in path if any(kw in p.lower() for kw in keywords))
            direction_match = min(1.0, matching_decisions / max(1, len(keywords))) if keywords else 0.5

        depth = node.depth
        optimal_depth_range = (3, 6)
        if optimal_depth_range[0] <= depth <= optimal_depth_range[1]:
            complexity_score = 1.0
        elif depth < optimal_depth_range[0]:
            complexity_score = 0.4 + (depth / optimal_depth_range[0]) * 0.6
        else:
            complexity_score = max(0.2, 1.0 - (depth - optimal_depth_range[1]) * 0.15)

        pattern_match_score = 0.0
        if known_good_exprs:
            path_str = " ".join(path).lower()
            pattern_matches = sum(
                1 for expr in known_good_exprs
                if any(partial in expr.lower() for partial in path_str.split())
            )
            pattern_match_score = min(0.3, pattern_matches * 0.05)

        total_reward = (
            diversity_score * 0.30 +
            direction_match * 0.25 +
            complexity_score * 0.30 +
            pattern_match_score * 0.15
        )

        noise = random.gauss(0, 0.05)
        total_reward = max(0.0, min(1.0, total_reward + noise))

        return total_reward

    def _mcts_backpropagate(self, node: _MCTSNode, reward: float) -> None:
        """反向傳播階段：將模擬結果沿路徑向上更新所有祖先節點。

        更新內容：
            node.visits += 1
            node.reward += reward
        對從 node 到 root 的所有祖先遞歸執行。

        Args:
            node: 開始反向傳播的節點（通常是剛展開或模擬的葉節點）
            reward: 模擬階段計算出的獎值
        """
        current: _MCTSNode | None = node
        while current is not None:
            current.visits += 1
            current.reward += reward
            depth_penalty = 1.0 - (current.depth * 0.02)
            adjusted_reward = reward * max(0.5, depth_penalty)
            current.reward += adjusted_reward * 0.5
            current = current.parent

    @algo_log(log_args_to_skip=("self", "feature_map", "evolution_db", "known_good_exprs"))
    async def mcts_explore(
        self,
        direction: str = "momentum",
        iterations: int = 20,
        exploration_budget: int = 3,
        feature_map=None,
        evolution_db=None,
        known_good_exprs: list[str] | None = None,
    ) -> list[dict]:
        """MCTS 主搜尋入口 — 在 alpha 搜索空間中執行 Monte Carlo Tree Search。

        完整流程：
        1. 初始化根節點（decision="root", decision_type="root"）
        2. 初始化根節點的 untried_actions = 所有可能的 root_op 選項
        3. 迴圈 iterations 次：
           a. select: UCB1 選擇最有潛力的葉節點
           b. expand: 展開一個新子節點（如果葉節點未完全展開）
           c. simulate: 估算該節點的潛在價值
           d. backpropagate: 更新路徑上所有節點的統計資料
        4. 從根節點選擇 visits 最高（最被驗證）的前 exploration_budget 條路徑
        5. 將每條路徑轉換為「建議的表達式結構」提示
        6. 對每條路徑，如果需要則調用 LLM 生成完整表達式

        Args:
            direction: 探索方向（momentum/mean_reversion/value/volatility 等）
            iterations: MCTS 迴圈次數（預設 20，每次約 0.01s 無 LLM）
            exploration_budget: 回傳的探索建議數量上限
            feature_map: FeatureMap 實例（可選，用於引導探索低覆蓋區域）
            evolution_db: EvolutionDB 實例（可選，用於已知好模式的加分）
            known_good_exprs: 已知高 Sharpe 表達式列表（可選，用於 simulate 階段的模式匹配）

        Returns:
            包含探索結果的字典列表，每個字典包含：
            - expression: 生成的表達式（如果有 LLM）
            - structure_path: 決策路徑
            - ucb1_score: 最終 UCB1 分數
            - visits: 訪問次數
            - avg_reward: 平均獎值
            - diversity_bonus: 多樣性加成
            - exploration_reason: 人類可讀的探索原因
        """
        root = _MCTSNode(
            decision="root",
            decision_type="root",
            untried_actions=self.ROOT_OPS[:],
            expr_hint="MCTS root",
            depth=0,
        )

        logger.info(
            "MCTS explore started: direction=%s, iterations=%d, budget=%d",
            direction, iterations, exploration_budget,
        )

        try:
            for i in range(iterations):
                leaf = self._mcts_select(root)

                if leaf.is_terminal or not leaf.is_fully_expanded:
                    expanded = self._mcts_expand(leaf, feature_map=feature_map)
                    node_to_simulate = expanded if expanded is not None else leaf
                else:
                    node_to_simulate = leaf

                reward = self._mcts_simulate(
                    node_to_simulate,
                    direction=direction,
                    known_good_exprs=known_good_exprs,
                )

                self._mcts_backpropagate(node_to_simulate, reward)

                if (i + 1) % 5 == 0:
                    logger.debug(
                        "MCTS iteration %d/%d completed, root visits=%d",
                        i + 1, iterations, root.visits,
                    )
        except Exception as e:
            logger.error("MCTS explore failed at iteration: %s", e, exc_info=True)
            return []

        candidate_nodes: list[_MCTSNode] = []
        self._collect_best_paths(root, candidate_nodes, max_depth=6)

        candidate_nodes.sort(key=lambda n: (-n.visits, -n.reward / max(1, n.visits)))
        top_candidates = candidate_nodes[:exploration_budget]

        results = []
        for idx, candidate in enumerate(top_candidates):
            structure_path = candidate.get_path_to_root()
            path_key = "|".join(structure_path)
            path_freq = self._path_history.get(path_key, 0)
            diversity_bonus = 1.0 / (1.0 + path_freq)

            avg_reward = candidate.reward / max(1, candidate.visits)
            ucb1_score = avg_reward + (1.41 * math.sqrt(math.log(max(1, root.visits)) / max(1, candidate.visits)))

            expression = None
            if self._llm_generate_fn is not None and candidate.depth >= 3:
                try:
                    expression = await self._generate_expression_from_path(
                        structure_path, direction,
                    )
                except (ConnectionError, OSError, TimeoutError) as e:
                    logger.warning("MCTS LLM generation failed for path %d: %s", idx, e)

            reason_parts = [
                f"深度={candidate.depth}",
                f"訪問={candidate.visits}次",
                f"平均獎值={avg_reward:.3f}",
            ]

            adaptive_budget = compute_adaptive_simulation_budget(
                node_visits=candidate.visits,
                node_reward_variance=max(0.0, (avg_reward * (1 - avg_reward)) / max(1, candidate.visits)),
                parent_visits=root.visits,
            )
            reason_parts.append(f"自適應預算={adaptive_budget}")
            if diversity_bonus > 0.7:
                reason_parts.append(f"高多樣性({diversity_bonus:.2f})")
            if direction and any(kw in " ".join(structure_path).lower() for kw in self.DIRECTION_KEYWORDS.get(direction, [])):
                reason_parts.append(f"符合{direction}方向")

            result_entry = {
                "expression": expression,
                "structure_path": structure_path,
                "ucb1_score": round(ucb1_score, 4),
                "visits": candidate.visits,
                "avg_reward": round(avg_reward, 4),
                "diversity_bonus": round(diversity_bonus, 4),
                "adaptive_simulation_budget": adaptive_budget,
                "exploration_reason": ", ".join(reason_parts),
            }
            results.append(result_entry)

            logger.info(
                "MCTS result %d: visits=%d, avg_reward=%.3f, ucb1=%.3f, reason=%s",
                idx + 1, candidate.visits, avg_reward, ucb1_score,
                result_entry["exploration_reason"],
            )

        logger.info("MCTS explore completed: generated %d candidates", len(results))
        return results

    def _collect_best_paths(
        self,
        node: _MCTSNode,
        candidates: list[_MCTSNode],
        max_depth: int = 6,
    ) -> None:
        """遞歸收集最佳候選路徑（基於 visits 和 depth）"""
        if node.depth >= 2 and node.visits > 0:
            should_add = True
            for existing in candidates:
                if existing.get_path_to_root() == node.get_path_to_root():
                    if node.visits > existing.visits:
                        candidates.remove(existing)
                    else:
                        should_add = False
                    break

            if should_add and node.depth <= max_depth:
                candidates.append(node)

        for child in node.children:
            self._collect_best_paths(child, candidates, max_depth)

    async def _generate_expression_from_path(
        self,
        structure_path: list[str],
        direction: str = "",
    ) -> str | None:
        """根據 MCTS 決策路徑生成完整的 alpha 表達式"""
        if self._llm_generate_fn is None:
            return None

        path_description = "\n".join(f"  Step {i+1}: {step}" for i, step in enumerate(structure_path))

        prompt = (
            "You are an alpha factor engineer. Based on the following structural decisions "
            "from a Monte Carlo Tree Search exploration, create a valid WorldQuant BRAIN "
            "FastExpr expression.\n\n"
            f"Structural Path:\n{path_description}\n\n"
            f"Target Direction: {direction}\n\n"
            "Requirements:\n"
            "- Must use group_neutralize() as the outermost function\n"
            "- Must be valid FastExpr syntax\n"
            "- The expression should reflect the structural choices above\n"
            "- Output ONLY the expression, nothing else"
        )

        try:
            response = await self._llm_generate_fn(prompt)
            if response:
                expr = response.strip()
                if expr.startswith("```"):
                    lines = expr.split("\n")
                    lines = [l for l in lines if not l.startswith("```")]
                    expr = "\n".join(lines).strip()
                return expr if expr else None
        except (ValueError, TypeError, OSError) as e:
            logger.warning("_generate_expression_from_path failed: %s", e)

        return None

    DIRECTION_KEYWORDS = {
        "momentum": ["rank", "delta", "ts_delta"],
        "mean_reversion": ["zscore", "scale", "ts_mean", "decay"],
        "value": ["earnings", "revenue", "sharesout"],
        "volatility": ["std_dev", "ts_std_dev", "skewness"],
        "volume": ["volume", "vwap"],
    }

    @algo_log(log_args_to_skip=("self", "feature_map", "evolution_db"))
    async def explore_unexplored_regions(
        self,
        feature_map,
        evolution_db,
        top_k: int = 3,
    ) -> list[dict]:
        results: list[dict] = []
        stats = feature_map.get_diversity_stats()
        coverage = stats.get("coverage", 0.0)
        filled_cells = stats.get("filled_cells", 0)

        if coverage >= 0.8:
            return []

        empty_cells = []
        with feature_map._lock:
            for key, cell in feature_map._cells.items():
                if not cell.best_expr:
                    empty_cells.append((key, cell))

        if not empty_cells:
            return []

        filled_list = []
        with feature_map._lock:
            for key, cell in feature_map._cells.items():
                if cell.best_expr:
                    filled_list.append((key, cell))

        if len(filled_list) < 2:
            return []

        direction_map = {"momentum": 0, "mean_reversion": 1, "volatility": 2, "statistical": 3, "volume": 4, "interaction": 5}
        horizon_map = {"short": 0, "medium": 1, "long": 2}
        mechanism_map = {"signal": 0, "normalized": 1, "conditional": 2, "interaction": 3}

        def _cell_vec(cell):
            return np.array([
                direction_map.get(cell.direction, 0),
                horizon_map.get(cell.time_horizon, 0),
                mechanism_map.get(cell.mechanism, 0),
            ], dtype=np.float64)

        filled_vecs = [(key, cell, _cell_vec(cell)) for key, cell in filled_list]

        known_good_exprs = [cell.best_expr for _, cell in filled_list[:10]] if len(filled_list) >= 3 else None

        if len(filled_list) >= 3 and known_good_exprs is not None:
            try:
                sample_empty_cell = empty_cells[0][1] if empty_cells else None
                direction_hint = sample_empty_cell.direction if sample_empty_cell else "momentum"

                logger.info(
                    "explore_unexplored_regions: 使用 MCTS 模式 (filled=%d, direction=%s)",
                    len(filled_list), direction_hint,
                )

                mcts_results = await self.mcts_explore(
                    direction=direction_hint,
                    iterations=20,
                    exploration_budget=top_k,
                    feature_map=feature_map,
                    evolution_db=evolution_db,
                    known_good_exprs=known_good_exprs,
                )

                if mcts_results:
                    for idx, mcts_result in enumerate(mcts_results):
                        if idx >= top_k - len(results):
                            break

                        result_entry = {
                            "cell_key": f"mcts_{idx}",
                            "expression": mcts_result.get("expression"),
                            "feature_description": (
                                f"[MCTS 探索] 路徑: {' → '.join(mcts_result['structure_path'])}. "
                                f"{mcts_result['exploration_reason']}"
                            ),
                            "structure_path": mcts_result.get("structure_path", []),
                            "ucb1_score": mcts_result.get("ucb1_score", 0),
                            "visits": mcts_result.get("visits", 0),
                            "avg_reward": mcts_result.get("avg_reward", 0),
                        }
                        results.append(result_entry)

                    if len(results) >= top_k:
                        logger.info("MCTS 已生成 %d 個候選，跳過傳統插值", len(results))
                        return results

            except (OSError, ValueError, RuntimeError) as e:
                logger.warning("MCTS 探索失敗，回退到傳統插值模式: %s", e, exc_info=True)

        filled_embeddings: dict[str, list[float] | None] = {}
        if self._embed_fn is not None:
            for key, cell in filled_list:
                emb = await self._compute_embedding(cell.best_expr)
                filled_embeddings[key] = emb

        for empty_key, empty_cell in empty_cells[:top_k]:
            empty_vec = _cell_vec(empty_cell)
            dists = []
            for fkey, fcell, fvec in filled_vecs:
                d = float(np.linalg.norm(empty_vec - fvec))
                dists.append((d, fkey, fcell))
            dists.sort(key=lambda x: x[0])

            nearest = dists[:2]
            if len(nearest) < 2:
                continue

            parent_a = nearest[0][2].best_expr
            parent_b = nearest[1][2].best_expr
            parent_a_key = nearest[0][1]
            parent_b_key = nearest[1][1]
            feature_desc = f"direction={empty_cell.direction}, time_horizon={empty_cell.time_horizon}, mechanism={empty_cell.mechanism}"

            interpolation_hint = ""
            if self._embed_fn is not None:
                emb_a = filled_embeddings.get(parent_a_key)
                emb_b = filled_embeddings.get(parent_b_key)
                if emb_a is not None and emb_b is not None:
                    try:
                        interpolated = self.interpolate_embeddings(emb_a, emb_b, alpha=0.5)
                        interpolation_hint = (
                            " [semantic interpolation: vector computed from nearby explored cells "
                            "to guide exploration of this unexplored region]"
                        )
                        try:
                            from openalpha_brain.core.loop_state import _algo_tick
                            _algo_tick("semantic_interpolation")
                        except (ImportError, AttributeError):
                            pass
                    except (OSError, ValueError, RuntimeError):
                        pass

            expr = await self.decode_to_expression(
                parent_a, parent_b,
                interpolation_ratio=0.5,
                direction=empty_cell.direction,
            )

            if expr:
                results.append({
                    "cell_key": empty_key,
                    "expression": expr,
                    "feature_description": feature_desc + interpolation_hint,
                })

            if len(results) >= top_k:
                break

        return results
