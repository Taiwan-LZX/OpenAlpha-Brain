from openalpha_brain.knowledge.rag_engine import RAGEngine


class TestRAGFeedbackWeights:
    def test_update_weights_low_sharpe(self):
        engine = RAGEngine()
        engine.update_weights_from_feedback("momentum", [{"name": "LOW_SHARPE", "result": "FAIL"}])
        weights = engine._feedback_weights.get("momentum", {})
        assert weights.get("ts_ops_boost", 1.0) > 1.0

    def test_update_weights_high_turnover(self):
        engine = RAGEngine()
        engine.update_weights_from_feedback("momentum", [{"name": "HIGH_TURNOVER", "result": "FAIL"}])
        weights = engine._feedback_weights.get("momentum", {})
        assert weights.get("smoothing_ops_boost", 1.0) > 1.0

    def test_update_weights_self_correlation(self):
        engine = RAGEngine()
        engine.update_weights_from_feedback("momentum", [{"name": "SELF_CORRELATION", "result": "FAIL"}])
        weights = engine._feedback_weights.get("momentum", {})
        assert weights.get("alternative_ops_boost", 1.0) > 1.0

    def test_no_update_on_pass(self):
        engine = RAGEngine()
        engine.update_weights_from_feedback("momentum", [{"name": "LOW_SHARPE", "result": "PASS"}])
        weights = engine._feedback_weights.get("momentum", {})
        assert "ts_ops_boost" not in weights

    def test_rerank_preserves_result_without_weights(self):
        engine = RAGEngine()
        result = {"top_ops_detailed": [{"name": "rank", "category": "ranking"}]}
        reranked = engine._rerank_with_feedback(result, "momentum")
        assert reranked == result
